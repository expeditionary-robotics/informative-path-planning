# !/usr/bin/python

'''
This library allows access to the Monte Carlo Tree Search class used in the PLUMES framework.
A MCTS allows for performing many forward simulation of multiple-chained actions in order to 
select the single most promising action to take at some time t. We have presented a variation
of the MCTS by forward simulating within an incrementally updated GP belief world.

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''

import numpy as np
import scipy as sp
import math
import os
import GPy as GPy
import time
from itertools import chain
import pdb
import logging
logger = logging.getLogger('robot')
from aq_library import *
import copy
import random

class MCTS(object):
    '''Class that establishes a MCTS for nonmyopic planning'''

    def __init__(self, computation_budget, belief, initial_pose, rollout_length, path_generator, aquisition_function, f_rew, T, aq_param = None, use_cost = False, tree_type = None, num_samples=20):
        '''
        Initialize with constraints for the planning, including whether there is a budget or planning horizon
        Inputs:
            computation_budget (float) number of seconds to run the tree building procedure
            belief (GP model) current belief of the vehicle
            initial_pose (tuple of floats) location of the vehicle in world coordinates
            rollout_length (int) number of actions to rollout after selecting a child (tree depth)
            frontier_size (int) number of options for each action in the tree (tree breadth)
            path_generator (string) how action sets should be developed
            aquisition_function (function) the criteria to make decisions
            f_rew (string) the name of the function used to make decisions
            T (float) time in the global world used for aquisition weighting
        '''
        # Status of the robot
        self.GP = belief
        self.cp = initial_pose
        self.path_generator = path_generator

        # Parameterization for the search
        self.comp_budget = computation_budget
        self.rl = rollout_length

        # The tree
        self.tree = None
        
        # Elements which are relevant for some acquisition functions
        self.aquisition_function = aquisition_function
        self.params = None
        self.max_val = None
        self.max_locs = None
        self.target = None

        self.current_max = aq_param
        self.f_rew = f_rew
        self.t = T
        self.use_cost = use_cost
        self.num_samples = num_samples

        # constants for the UCT selection in the MCTS
        # determined through empirical observation
        if self.f_rew == 'mean':
            self.c = 300
        elif self.f_rew == 'exp_improve':
            self.c = 200
        elif self.f_rew == 'mes':
            self.c = 1.0 / np.sqrt(2.0)
        else:
            self.c = 1.0

    def choose_trajectory(self, t):
        ''' 
        Main function loop which makes the tree and selects the best child
        Output: path to take, cost of that path
        '''
        # initialize tree
        self.tree = self.initialize_tree() 
        i = 0 #iteration count

        # randomly sample the world for entropy search function
        if self.f_rew == 'mes' or self.f_rew == 'maxs-mes':
            self.max_val, self.max_locs, self.target  = sample_max_vals(self.GP, t = t)
            
        time_start = time.time()            
        # while we still have time to compute, generate the tree
        while time.time() - time_start < self.comp_budget:#i < self.comp_budget:
            i += 1
            current_node = self.tree_policy()
            sequence = self.rollout_policy(current_node)
            reward, cost = self.get_reward(sequence)
            self.update_tree(reward, cost, sequence)
            print(1)

        time_end = time.time()
        print "Rollouts completed in", str(time_end - time_start) +  "s", 

        # get the best action to take with most promising futures, base best on whether to
        # consider cost
        best_sequence, best_val, all_vals = self.get_best_child()
        paths, dense_paths = self.path_generator.get_path_set(self.cp)

        #Document the information
        print "Number of rollouts:", i, "\t Size of tree:", len(self.tree)
        logger.info("Number of rollouts: {} \t Size of tree: {}".format(i, len(self.tree)))
        np.save('./figures/' + self.f_rew + '/tree_' + str(t) + '.npy', self.tree)
        return self.tree[best_sequence][0], self.tree[best_sequence][1], best_val, paths, all_vals, self.max_locs, self.max_val

    def initialize_tree(self):
        '''
        Creates a tree instance, which is a dictionary, that keeps track of the nodes in the world
        Output: tree (dictionary) an initial tree
        '''
        tree = {}
        # root of the tree is current location of the vehicle
        tree['root'] = (self.cp, 0) #(pose, number of queries)
        actions, dense_paths = self.path_generator.get_path_set(self.cp)
        for action in actions.keys():
             #(samples robot observes, path, cost, reward, number of times queried)
            tree['child '+str(action)] = (actions[action], dense_paths[action], 0, 0, 0)
        return tree

    def tree_policy(self):
        '''
        Implements the UCB policy to select the child to expand and forward simulate. From Arora paper, the following is defined:
            avg_r - average reward of all rollouts that have passed through node n
            c_p - some arbitrary constant, they use 0.1
            N - number of times parent has been evaluated
            n - number of times that node has been evaluated
            the formula: avg_r + c_p * np.sqrt(2*np.log(N)/n)
        '''
        leaf_eval = {}
        actions, dense_paths = self.path_generator.get_path_set(self.cp)
        for i, val in actions.items():
            node = 'child '+ str(i)
            if self.tree[node][4] == 0:
                return node
            else:
                leaf_eval[node] = self.tree[node][3] + self.c*np.sqrt(2*(np.log(self.tree['root'][1]))/self.tree[node][4])
        return random.choice([key for key in leaf_eval.keys() if leaf_eval[key] == max(leaf_eval.values())])

    def rollout_policy(self, node):
        '''
        Select random actions to expand the child node
        Input: node (the name of the child node that is to be expanded)
        Output: sequence (list of names of nodes that make the sequence in the tree)
        '''
        sequence = [node] #include the child node
        for i in xrange(self.rl):
            actions, dense_paths = self.path_generator.get_path_set(self.tree[node][0][-1]) #plan from the last point in the sample
            #check that paths were generated; if not, roll back if possible
            try:
                keys = actions.keys()
            except:
                print 'No actions were viably generated; rolling back'
                sequence.remove(node)
                if len(sequence) == 0:
                    print "Empty sequence ", sequence, node
                    logger.warning("Bad Sequence")
            #select a random action
            try: 
                a = np.random.randint(0,len(actions)-1)
            except:
                a = 0

            #create the sequence and add to the tree
            try:
                self.tree[node + ' child ' + str(keys[a])] = (actions[keys[a]], dense_paths[keys[a]], 0, 0, 0) #add random path to the tree
                node = node + ' child ' + str(keys[a])
                sequence.append(node)
            except:
                # This seems like this should never happen?!
                #pdb.set_trace()
                pass

        return sequence

    def get_reward(self, sequence):
        '''
        Evaluate the sequence to get the reward, defined by the percentage of entropy reduction.
        Input: sequence (list of strings) names of the nodes in the tree
        Outut: reward value from the aquisition function of choice
        '''
        sim_world = copy.copy(self.GP) #TODO try selecting a simulated world from spectral sampling
        samples = []
        obs = []
        cost = 0
        reward = 0

        for seq in sequence:
            samples.append(self.tree[seq][0])
            if self.use_cost == True:
                cost += self.path_generator.path_cost(self.tree[seq][1])
        
        obs = list(chain.from_iterable(samples))

        if self.f_rew == 'maxs-mes':
            reward = self.aquisition_function(time = self.t, xvals = obs, robot_model = self.GP, param = (self.max_val, self.max_locs, self.target))
            return reward, cost

        for s in samples:
            obs = np.array(s)
            xobs = np.vstack([obs[:,0], obs[:,1]]).T
            if self.f_rew == 'mes' or self.f_rew == 'maxs-mes':
                reward += self.aquisition_function(time = self.t, xvals = xobs, robot_model = sim_world, param = (self.max_val, self.max_locs, self.target))
            elif self.f_rew == 'exp_improve':
                reward += self.aquisition_function(time=self.t, xvals = xobs, robot_model = sim_world, param = [self.current_max])
            elif self.f_rew == 'naive':
                reward += self.aquisition_function(time = self.t, xvals = xobs, robot_model = sim_world, param = (self.num_samples))
            elif self.f_rew == 'naive_value':
                reward += self.aquisition_function(time = self.t, xvals = xobs, robot_model = sim_world, param = (self.num_samples))
            else:
                reward += self.aquisition_function(time=self.t, xvals = xobs, robot_model = sim_world)

            if sim_world.model is None:
                n_points, input_dim = xobs.shape
                zmean, zvar = np.zeros((n_points, )), np.eye(n_points) * self.GP.variance
                zobs = np.random.multivariate_normal(mean = zmean, cov = zvar)
                zobs = np.reshape(zobs, (n_points, 1))
            else:
                zobs = sim_world.posterior_samples(xobs, full_cov = False, size=1)
                #print zobs
            sim_world.add_data(xobs, zobs)
        return reward, cost
    
    def update_tree(self, reward, cost, sequence):
        '''Propogate the reward for the sequence
        Input:
            reward (float) the reward or utility value of the sequence
            sequence (list of strings) the names of nodes that form the sequence
        '''
        self.tree['root'] = (self.tree['root'][0], self.tree['root'][1]+1)
        for seq in sequence:
            samples, path, cos, rew, queries = self.tree[seq]
            queries += 1
            n = queries
            rew = ((n-1)*rew+reward)/n
            cos = ((n-1)*cos+cost)/n
            self.tree[seq] = (samples, path, cos, rew, queries)

    def get_best_child(self):
        '''Query the tree for the best child in the actions
        Output:
            (string, float) node name of the best child, the cost of that child
        '''
        best = -float('inf')
        best_child = None
        value = {}
        actions, dense_paths = self.path_generator.get_path_set(self.cp)
        keys = actions.keys()
        for i in keys:
            try:
                if self.use_cost == False:
                    r = self.tree['child '+ str(i)][3]
                    value[i] = r
                else:
                    if self.tree['child ' + str(i)][2] == 0.0:
                        r = self.tree['child '+ str(i)][3]/100.
                    else:
                        r = self.tree['child '+ str(i)][3]/self.tree['child ' + str(i)][2]
                    value[i] = r
                if r > best: 
                    best = r
                    best_child = 'child '+ str(i)
            except:
                pass
        return best_child, best, value


class Node(object):
    def __init__(self, pose, parent, name, action = None, dense_path = None, zvals = None):
        self.pose = pose
        self.name = name
        self.zvals = zvals
        self.reward = 0.0
        self.nqueries = 0
        
        # Parent will be none if the node is a root
        self.parent = parent
        self.children = None

        # Set belief or belief action node
        if action is None:
            self.node_type = 'B'
            self.action = None 
            self.dense_path = None

            # If the root node, depth is 0
            if parent is None:
                self.depth = 0
            else:
                self.depth = parent.depth + 1
        else:
            self.node_type = 'BA'
            self.action = action
            self.dense_path = dense_path
            self.depth = parent.depth

    def add_children(self, child_node):
        if self.children is None:
            self.children = []
        self.children.append(child_node)
    
    def print_self(self):
        print self.name

class Tree(object):
    def __init__(self, f_rew, f_aqu,  belief, pose, path_generator, t, depth, param, c):
        self.path_generator = path_generator
        self.max_depth = depth
        self.param = param
        self.t = t
        self.f_rew = f_rew
        self.aquisition_function = f_aqu
        self.c = c

        self.root = Node(pose, parent = None, name = 'root', action = None, dense_path = None, zvals = None)  
        #self.build_action_children(self.root) 

    def get_best_child(self):
        return self.root.children[np.argmax([node.nqueries for node in self.root.children])]

    def backprop(self, leaf_node, reward):
        if leaf_node.parent is None:
            leaf_node.nqueries += 1
            leaf_node.reward += reward
            #print "Calling backprop on:",
            #leaf_node.print_self()
            #print "nqueries:", leaf_node.nqueries, "reward:", leaf_node.reward
            return
        else:
            leaf_node.nqueries += 1
            leaf_node.reward += reward
            #print "Calling backprop on:",
            #leaf_node.print_self()
            #print "nqueries:", leaf_node.nqueries, "reward:", leaf_node.reward
            self.backprop(leaf_node.parent, reward)
            return
    
    def get_next_leaf(self, belief):
        #print "Calling next with root"
        next_leaf, reward = self.leaf_helper(self.root, reward = 0.0,  belief = belief) 
        #print "Next leaf:", next_leaf
        #print "Reward:", reward
        self.backprop(next_leaf, reward)

    def leaf_helper(self, current_node, reward, belief):
        if current_node.node_type == 'B':
            # Root belief node
            if current_node.depth == self.max_depth:
                #print "Returning leaf node:", current_node.name, "with reward", reward
                return current_node, reward
            # Intermediate belief node
            else:
                if current_node.children is None:
                    self.build_action_children(current_node)

                # If no viable actions are avaliable
                if current_node.children is None:
                    return current_node, reward

                child = self.get_next_child(current_node)
                #print "Selecting next action child:", child.name

                # Recursive call
                return self.leaf_helper(child, reward, belief)

        # At random node, after selected action from a specific node
        elif current_node.node_type == 'BA':
            # Copy old belief
            #gp_new = copy.copy(current_node.belief) 
            #gp_new = current_node.belief

            # Sample a new set of observations and form a new belief
            #xobs = current_node.action
            obs = np.array(current_node.action)
            print(obs)
            xobs = np.vstack([obs[:,0], obs[:,1]]).T

            if self.f_rew == 'mes' or self.f_rew == 'maxs-mes':
                r = self.aquisition_function(time = self.t, xvals = xobs, robot_model = belief, param = self.param)
            elif self.f_rew == 'exp_improve':
                r = self.aquisition_function(time = self.t, xvals = xobs, robot_model = belief, param = self.param)
            elif self.f_rew == 'naive':
                # param = sample_max_vals(belief, t=self.t, nK=int(self.param[0]))
                r = self.aquisition_function(time = self.t, xvals = xobs, robot_model = belief, param = self.param)#(param, self.param[1]))
            elif self.f_rew == 'naive_value':
                r = self.aquisition_function(time = self.t, xvals = xobs, robot_model = belief, param = self.param)
            else:
                r = self.aquisition_function(time = self.t, xvals = xobs, robot_model = belief)

            if current_node.children is not None:
                alpha = 3.0 / (10.0 * (self.max_depth - current_node.depth) - 3.0)
                nchild = len(current_node.children)
                #print "Current depth:", current_node.depth, "alpha:", alpha
                #print "First:", np.floor(nchild ** alpha)
                #print "Second:", np.floor((nchild - 1) ** alpha)
                if current_node.depth < self.max_depth - 1 and np.floor(nchild ** alpha) == np.floor((nchild - 1) ** alpha):
                    #print "Choosing from among current nodes"
                    #child = random.choice(current_node.children)
                    #print "number quieres:", nqueries
                    child = random.choice(current_node.children)
                    nqueries = [node.nqueries for node in current_node.children]
                    child = random.choice([node for node in current_node.children if node.nqueries == min(nqueries)])

                    if True:
                        belief.add_data(xobs, child.zvals)
                    #print "Selcted child:", child.nqueries
                    return self.leaf_helper(child, reward + r, belief)

            if True:
                if belief.model is None:
                    n_points, input_dim = xobs.shape
                    zmean, zvar = np.zeros((n_points, )), np.eye(n_points) * belief.variance
                    zobs = np.random.multivariate_normal(mean = zmean, cov = zvar)
                    zobs = np.reshape(zobs, (n_points, 1))
                else:
                    zobs = belief.posterior_samples(xobs, full_cov = False, size = 1)
                    n_points, input_dim = xobs.shape
                    zobs = np.reshape(zobs, (n_points,1))

                # print(xobs)
                # print(type(zobs))
                belief.add_data(xobs, zobs)
            else:
                zobs = belief.posterior_samples(xobs, full_cov = False, size = 1)
                n_points, input_dim = xobs.shape
                zobs = np.reshape(zobs, (n_points,1))

            belief.add_data(xobs, zobs)
            pose_new = current_node.dense_path[-1]
            child = Node(pose = pose_new, 
                         parent = current_node, 
                         name = current_node.name + '_belief' + str(current_node.depth + 1), 
                         action = None, 
                         dense_path = None, 
                         zvals = zobs)
            #print "Adding next belief child:", child.name
            current_node.add_children(child)

            # Recursive call
            return self.leaf_helper(child, reward + r, belief)

    def get_next_child(self, current_node):
        vals = {}
        # e_d = 0.5 * (1.0 - (3.0/10.0*(self.max_depth - current_node.depth)))
        e_d = 0.5 * (1.0 - (3.0/(10.0*(self.max_depth - current_node.depth))))
        for i, child in enumerate(current_node.children):
            #print "Considering child:", child.name, "with queries:", child.nqueries
            if child.nqueries == 0:
                return child
            vals[child] = child.reward/float(child.nqueries) + self.c * np.sqrt((float(current_node.nqueries) ** e_d)/float(child.nqueries)) 
            #vals[child] = child.reward/float(child.nqueries) + self.c * np.sqrt(np.log(float(current_node.nqueries))/float(child.nqueries)) 
        # Return the max node, or a random node if the value is equal
        return random.choice([key for key in vals.keys() if vals[key] == max(vals.values())])
        
    def build_action_children(self, parent):
        # print(self.path_generator.get_path_set(parent.pose))
        # print(self.path_generator)
        actions, dense_paths = self.path_generator.get_path_set(parent.pose)
        # actions = self.path_generator.get_path_set(parent.pose)
        # dense_paths = [0]
        if len(actions) == 0:
            print "No actions!", 
            return
        
        #print "Creating children for:", parent.name
        for i, action in enumerate(actions.keys()):
            #print "Action:", i
            parent.add_children(Node(pose = parent.pose, 
                                    parent = parent, 
                                    name = parent.name + '_action' + str(i), 
                                    action = actions[action], 
                                    dense_path = dense_paths[action],
                                    zvals = None))

    def print_tree(self):
        counter = self.print_helper(self.root)
        print "# nodes in tree:", counter

    def print_helper(self, cur_node):
        if cur_node.children is None:
            #cur_node.print_self()
            #print cur_node.name
            return 1
        else:
            #cur_node.print_self()
            #print "\n"
            counter = 0
            for child in cur_node.children:
                counter += self.print_helper(child)
            return counter

''' Inherit class, that implements more standard MCTS, and assumes MLE observation to deal with continuous spaces '''
class BeliefTree(Tree):
    def __init__(self, f_rew, f_aqu,  belief, pose, path_generator, t, depth, param, c):
        super(BeliefTree, self).__init__(f_rew, f_aqu,  belief, pose, path_generator, t, depth, param, c)

    # Max Reward-based node selection
    def get_best_child(self):
        return self.root.children[np.argmax([node.nqueries for node in self.root.children])]

    def random_rollouts(self, current_node, reward, belief):
        cur_depth = current_node.depth
        pose = current_node.pose
        while cur_depth <= self.max_depth:
            actions, dense_paths = self.path_generator.get_path_set(pose)
            keys = actions.keys()
            # No viable trajectories from current location
            if len(actions) <= 1:
                return reward

            #select a random action
            a = np.random.randint(0, len(actions) - 1)
            obs = np.array(actions[keys[a]])
            xobs = np.vstack([obs[:,0], obs[:,1]]).T

            if self.f_rew == 'mes' or self.f_rew == 'maxs-mes':
                r = self.aquisition_function(time = self.t, xvals = xobs, robot_model = belief, param = self.param)
            elif self.f_rew == 'exp_improve':
                r = self.aquisition_function(time = self.t, xvals = xobs, robot_model = belief, param = self.param)
            elif self.f_rew == 'naive':
                r = self.aquisition_function(time = self.t, xvals = xobs, robot_model = belief, param = self.param)
            elif self.f_rew == 'naive_value':
                r = self.aquisition_function(time = self.t, xvals = xobs, robot_model = belief, param = self.param)
            else:
                r = self.aquisition_function(time = self.t, xvals = xobs, robot_model = belief)

            if True:
                # ''Simulate'' the maximum likelihood observation
                if belief.model is None:
                    n_points, input_dim = xobs.shape
                    zobs = np.zeros((n_points, ))
                    zobs = np.reshape(zobs, (n_points, 1))
                else:
                    zobs, _= belief.predict_value(xobs)

                belief.add_data(xobs, zobs)
            else:
                zobs, _= belief.predict_value(xobs)

            belief.add_data(xobs, zobs)
            pose = dense_paths[keys[a]][-1]
            reward += r
            cur_depth += 1

        return reward

    def leaf_helper(self, current_node, reward, belief):
        if current_node.node_type == 'B':
            # belief node
            if current_node.depth == self.max_depth:
                #print "Returning leaf node:", current_node.name, "with reward", reward
                return current_node, reward
            # Intermediate belief node
            else:
                if current_node.children is None:
                    self.build_action_children(current_node)

                # If no viable actions are avaliable
                if current_node.children is None:
                    return current_node, reward

                child, full_action_set  = self.get_next_child(current_node)
                #print "Selecting next action child:", child.name
                #print "Full action set?", full_action_set

                if full_action_set:
                    # Recursive call
                    return self.leaf_helper(child, reward, belief)
                else:
                    # Do random rollouts
                    #print "Doing random rollouts!"
                    rollout_reward = self.random_rollouts(current_node, reward, belief) 
                    #print "Rollout reward:", rollout_reward
                    return child, rollout_reward

        # At random node, after selected action from a specific node
        elif current_node.node_type == 'BA':
            # Copy old belief
            #gp_new = copy.copy(current_node.belief) 
            #gp_new = current_node.belief

            # Sample a new set of observations and form a new belief
            #xobs = current_node.action
            obs = np.array(current_node.action)
            xobs = np.vstack([obs[:,0], obs[:,1]]).T

            if self.f_rew == 'mes' or self.f_rew == 'maxs-mes':
                r = self.aquisition_function(time = self.t, xvals = xobs, robot_model = belief, param = self.param)
            elif self.f_rew == 'exp_improve':
                r = self.aquisition_function(time = self.t, xvals = xobs, robot_model = belief, param = self.param)
            elif self.f_rew == 'naive':
                r = self.aquisition_function(time = self.t, xvals = xobs, robot_model = belief, param = self.param)
            elif self.f_rew == 'naive_value':
                r = self.aquisition_function(time = self.t, xvals = xobs, robot_model = belief, param = self.param)
            else:
                r = self.aquisition_function(time = self.t, xvals = xobs, robot_model = belief)

            if True:
                # ''Simulate'' the maximum likelihood observation
                if belief.model is None:
                    n_points, input_dim = xobs.shape
                    zobs = np.zeros((n_points, ))
                    zobs = np.reshape(zobs, (n_points, 1))
                else:
                    zobs, _= belief.predict_value(xobs)

                belief.add_data(xobs, zobs)
            else:
                zobs, _= belief.predict_value(xobs)

            belief.add_data(xobs, zobs)
            pose_new = current_node.dense_path[-1]
            child = Node(pose = pose_new, 
                         parent = current_node, 
                         name = current_node.name + '_belief' + str(current_node.depth + 1), 
                         action = None, 
                         dense_path = None, 
                         zvals = zobs)
            #print "Adding next belief child:", child.name
            current_node.add_children(child)

            # Recursive call
            return self.leaf_helper(child, reward + r, belief)

    ''' Returns the next most promising child of a belief node, and a FLAG indicating if belief node is fully explored '''
    def get_next_child(self, current_node):
        vals = {}
        for i, child in enumerate(current_node.children):
            #print "Considering child:", child.name, "with queries:", child.nqueries
            if child.nqueries == 0:
                return child, False
            vals[child] = child.reward/float(child.nqueries) + self.c * np.sqrt(2.0*np.log(float(current_node.nqueries))/float(child.nqueries)) 
        # Return the max node, or a random node if the value is equal
        return random.choice([key for key in vals.keys() if vals[key] == max(vals.values())]), True
        


class cMCTS(MCTS):
    '''Class that establishes a MCTS for nonmyopic planning'''
    def __init__(self, computation_budget, belief, initial_pose, rollout_length, path_generator, aquisition_function, f_rew, T, aq_param = None, use_cost = False, tree_type = 'dpw'):
        # Call the constructor of the super class
        super(cMCTS, self).__init__(computation_budget, belief, initial_pose, rollout_length, path_generator, aquisition_function, f_rew, T, aq_param, use_cost)
        self.tree_type = tree_type
        self.aq_param = aq_param
        # self.GP = belief

        # The differnt constatns use logarthmic vs polynomical exploriation
        if self.f_rew == 'mean':
            if self.tree_type == 'belief':
                self.c = 1000
            elif self.tree_type == 'dpw':
                self.c = 5000
        elif self.f_rew == 'exp_improve':
            self.c = 200
        elif self.f_rew == 'mes':
            if self.tree_type == 'belief':
                self.c = 1.0 / np.sqrt(2.0)
            elif self.tree_type == 'dpw':
                # self.c = 1.0 / np.sqrt(2.0)
                self.c = 1.0
                # self.c = 5.0
        else:
            self.c = 1.0
        print "Setting c to :", self.c

    def choose_trajectory(self, t):
        #Main function loop which makes the tree and selects the best child
        #Output: path to take, cost of that path

        # randomly sample the world for entropy search function
        if self.f_rew == 'mes':
            self.max_val, self.max_locs, self.target  = sample_max_vals(self.GP, t = t, visualize=True)
            param = (self.max_val, self.max_locs, self.target)
            print("Hello")
        elif self.f_rew == 'exp_improve':
            param = [self.current_max]
        elif self.f_rew == 'naive' or self.f_rew == 'naive_value':
            self.max_val, self.max_locs, self.target  = sample_max_vals(self.GP, t=t, nK=int(self.aq_param[0]), visualize=True, f_rew=self.f_rew)
            param = ((self.max_val, self.max_locs, self.target), self.aq_param[1])
        else:
            param = None

        # initialize tree
        if self.tree_type == 'dpw':
            self.tree = Tree(self.f_rew, self.aquisition_function, self.GP, self.cp, self.path_generator, t, depth = self.rl, param = param, c = self.c)
        elif self.tree_type == 'belief':
            self.tree = BeliefTree(self.f_rew, self.aquisition_function, self.GP, self.cp, self.path_generator, t, depth = self.rl, param = param, c = self.c)
        else:
            raise ValueError('Tree type must be one of either \'dpw\' or \'belief\'')
        #self.tree.get_next_leaf()
        #print self.tree.root.children[0].children

        time_start = time.time()            
        # while we still have time to compute, generate the tree
        i = 0
        while time.time() - time_start < self.comp_budget:#i < self.comp_budget:
            i += 1
            gp = copy.copy(self.GP)
            self.tree.get_next_leaf(gp)

            if True:
                gp = copy.copy(self.GP)
        time_end = time.time()
        print "Rollouts completed in", str(time_end - time_start) +  "s"
        print "Number of rollouts:", i
        self.tree.print_tree()

        print [(node.nqueries, node.reward/(node.nqueries+0.1)) for node in self.tree.root.children]

        # best_child = self.tree.root.children[np.argmax([node.nqueries for node in self.tree.root.children])]
        best_child = random.choice([node for node in self.tree.root.children if node.nqueries == max([n.nqueries for n in self.tree.root.children])])
        all_vals = {}
        for i, child in enumerate(self.tree.root.children):
            all_vals[i] = child.reward / (float(child.nqueries)+0.1)
            # print(str(i) + " is " + str(all_vals[i]))

        paths, dense_paths = self.path_generator.get_path_set(self.cp)
        return best_child.action, best_child.dense_path, best_child.reward/(float(best_child.nqueries)+1.0), paths, all_vals, self.max_locs, self.max_val, self.target

        # get the best action to take with most promising futures, base best on whether to
        # consider cost
        #best_sequence, best_val, all_vals = self.get_best_child()

        #Document the information
        #print "Number of rollouts:", i, "\t Size of tree:", len(self.tree)
        #logger.info("Number of rollouts: {} \t Size of tree: {}".format(i, len(self.tree)))
        #np.save('./figures/' + self.f_rew + '/tree_' + str(t) + '.npy', self.tree)
        #return self.tree[best_sequence][0], self.tree[best_sequence][1], best_val, paths, all_vals, self.max_locs, self.max_val

