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

class MCTS:
    '''Class that establishes a MCTS for nonmyopic planning'''

    def __init__(self, computation_budget, belief, initial_pose, rollout_length, path_generator, aquisition_function, f_rew, T, aq_param = None, use_cost = False):
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

        # constants for the UCT selection in the MCTS
        # determined through empirical observation
        if self.f_rew == 'mean':
            self.c = 300
        elif self.f_rew == 'exp_improve':
            self.c = 200
        elif self.f_rew == 'mes':
            self.c = 3
        else:
            self.c = 0.1

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
        while i < self.comp_budget:#time.time() - time_start < self.comp_budget:
            i += 1
            current_node = self.tree_policy()
            sequence = self.rollout_policy(current_node)
            reward, cost = self.get_reward(sequence)
            self.update_tree(reward, cost, sequence)

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
            self.tree[node + ' child ' + str(keys[a])] = (actions[keys[a]], dense_paths[keys[a]], 0, 0, 0) #add random path to the tree
            node = node + ' child ' + str(keys[a])
            sequence.append(node)

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
