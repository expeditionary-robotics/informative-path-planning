# !/usr/bin/python

'''
This library allows access to the Monte Carlo Tree Search class used in the PLUMES framework.
A MCTS allows for performing many forward simulation of multiple-chained actions in order to
select the single most promising action to take at some time t. We have presented a variation
of the MCTS by forward simulating within an incrementally updated GP belief world.

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''
import time
import random
import copy
from itertools import chain
import logging
from heuristic_rewards import *
import numpy as np
logger = logging.getLogger('robot')

class MCTS(object):
    '''Class that establishes a MCTS for nonmyopic planning'''

    def __init__(self, computation_budget, belief, initial_pose, rollout_length, path_generator, aquisition_function, f_rew, T, obs_world, aq_param=None, tree_type=None, use_sim_world=True):
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
        self.obs_world = obs_world
        self.use_sim_world = use_sim_world

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

        # constants for the UCT selection in the MCTS
        # determined through empirical observation
        if self.f_rew == 'mean':
            self.c = 300
        elif self.f_rew == 'exp_improve':
            self.c = 200
        elif self.f_rew == 'mes':
            self.c = 1.0 / np.sqrt(2.0)
        else:
            # self.c = 1.0
            self.c = 1.0 / np.sqrt(2.0)
            # self.c =  0.0

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
        paths = self.path_generator.generate_trajectories(self.cp, t, self.obs_world, self.use_sim_world)

        #Document the information
        print "Number of rollouts:", i, "\t Size of tree:", len(self.tree)
        logger.info("Number of rollouts: {} \t Size of tree: {}".format(i, len(self.tree)))
        np.save('./figures/' + self.f_rew + '/tree_' + str(t) + '.npy', self.tree)
        return self.tree[best_sequence][0], best_val, paths, all_vals, self.max_locs, self.max_val

    def initialize_tree(self):
        '''
        Creates a tree instance, which is a dictionary, that keeps track of the nodes in the world
        Output: tree (dictionary) an initial tree
        '''
        tree = {}
        # root of the tree is current location of the vehicle
        tree['root'] = (self.cp, 0) #(pose, number of queries)
        actions = self.path_generator.generate_trajectories(self.cp, 0, self.obs_world, self.use_sim_world)
        for i, action in enumerate(actions):
             #(samples robot observes, path, cost, reward, number of times queried)
            tree['child '+str(i)] = (action, action, 0, 0, 0)
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
        actions = self.path_generator.generate_trajectories(self.cp, self.t, self.obs_world, self.use_sim_world)
        for i, val in enumerate(actions):
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
            actions = self.path_generator.generate_trajectories(node.pose, self.t, self.obs_world, self.use_sim_world)
            #check that paths were generated; if not, roll back if possible
            if len(actions) > 0:
                #select a random action
                try: 
                    a = np.random.randint(0,len(actions)-1)
                except:
                    a = 0
            else:
                print 'No actions were viably generated; rolling back'
                sequence.remove(node)
                if len(sequence) == 0:
                    print "Empty sequence ", sequence, node
                    logger.warning("Bad Sequence")

            #create the sequence and add to the tree
            try:
                self.tree[node + ' child ' + str(a)] = (actions[a], actions[a], 0, 0, 0) #add random path to the tree
                node = node + ' child ' + str(a)
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
        
        obs = list(chain.from_iterable(samples))

        if self.f_rew == 'maxs-mes':
            reward = self.aquisition_function(time = self.t, xvals = obs, robot_model = self.GP, param = (self.max_val, self.max_locs, self.target))
            return reward, cost

        for s in samples:
            obs = np.array(s)
            xobs = np.vstack([obs[:,0], obs[:,1]]).T
            if self.f_rew == 'mes':
                reward += self.aquisition_function(time = self.t, xvals = xobs, robot_model = sim_world, param = (self.max_val, self.max_locs, self.target))
            elif self.f_rew == 'exp_improve':
                reward += self.aquisition_function(time=self.t, xvals = xobs, robot_model = sim_world, param = [self.current_max])
            else:
                reward += self.aquisition_function(time=self.t, xvals = xobs, robot_model = sim_world)

            if sim_world.model is None:
                n_points, input_dim = xobs.shape
                zmean, zvar = np.zeros((n_points, )), np.eye(n_points) * self.GP.variance
                zobs = np.random.multivariate_normal(mean=zmean, cov=zvar)
                zobs = np.reshape(zobs, (n_points, 1))
            else:
                zobs = sim_world.posterior_samples(xobs, full_cov=False, size=1)
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
        actions = self.path_generator.generate_trajectories(self.cp, t, self.obs_world, self.use_sim_world)
        for i, action in enumerate(actions):
            try:
                r = self.tree['child '+ str(i)][3]
                value[i] = r

                if r > best: 
                    best = r
                    best_child = 'child '+ str(i)
            except:
                pass
        return best_child, best, value


class Node(object):
    def __init__(self, pose, parent, name, action=None, zvals=None):
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

            # If the root node, depth is 0
            if parent is None:
                self.depth = 0
            else:
                self.depth = parent.depth + 1
        else:
            self.node_type = 'BA'
            self.action = action
            self.depth = parent.depth

    def add_children(self, child_node):
        if self.children is None:
            self.children = []
        self.children.append(child_node)

    def print_self(self):
        print self.name

class Tree(object):
    def __init__(self, f_rew, f_aqu, pose, path_generator, t, depth, param, c, obs_world, use_sim_world=True):
        self.path_generator = path_generator
        self.max_depth = depth
        self.param = param
        self.t = t
        self.f_rew = f_rew
        self.aquisition_function = f_aqu
        self.c = c
        self.obs_world = obs_world
        self.use_sim_world = use_sim_world

        self.root = Node(pose, parent=None, name='root', action=None, zvals=None)

    def get_best_child(self):
        return random.choice([node for node in self.root.children if node.nqueries == np.nanmax([n.nqueries for n in self.root.children])])
        # return self.root.children[np.argmax([node.nqueries for node in self.root.children])]

    def backprop(self, leaf_node, reward):
        if leaf_node.parent is None:
            leaf_node.nqueries += 1
            leaf_node.reward += reward
            return
        else:
            leaf_node.nqueries += 1
            leaf_node.reward += reward
            self.backprop(leaf_node.parent, reward)
            return

    def get_next_leaf(self, belief):
        next_leaf, reward = self.leaf_helper(self.root, reward=0.0, belief=copy.deepcopy(belief))
        self.backprop(next_leaf, reward)

    def leaf_helper(self, current_node, reward, belief):
        # belief = copy.deepcopy(belief)
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
            # Sample a new set of observations and form a new belief
            obs = np.array(current_node.action)
            if belief.dim == 2:
                xobs = np.vstack([obs[:, 0], obs[:, 1]]).T
            elif belief.dim == 3:
                xobs = np.vstack([obs[:, 0], obs[:, 1], (self.t+current_node.depth)*np.ones(obs.shape[0])]).T

            if self.f_rew == 'mes' or self.f_rew == 'exp_improve':
                r = self.aquisition_function(time=self.t+current_node.depth, xvals=xobs, robot_model=belief, param=self.param)
            elif self.f_rew == 'gumbel':
                param = sample_max_vals_gumbel(belief, t=self.t+current_node.depth, obstacles=self.obs_world)
                r = self.aquisition_function(time=self.t+current_node.depth, xvals=xobs, robot_model=belief, param=param)
            else:
                r = self.aquisition_function(time=self.t+current_node.depth, xvals=xobs, robot_model=belief)

            if current_node.children is not None:
                alpha = 3.0 / (10.0 * (self.max_depth - current_node.depth) - 3.0)
                nchild = len(current_node.children)

                if current_node.depth < self.max_depth - 1 and np.floor(nchild ** alpha) == np.floor((nchild - 1) ** alpha):
                    # print 'revisiting child'
                    nqueries = [node.nqueries for node in current_node.children]
                    child = random.choice([node for node in current_node.children if node.nqueries == np.nanmin(nqueries)])
                    belief.add_data(xobs, child.zvals)
                    return self.leaf_helper(child, reward + r, belief)

            if belief.xvals is None:
                print 'belief model is None'
                n_points, _ = xobs.shape
                zmean, zvar = np.zeros((n_points, )), np.eye(n_points) * belief.variance
                zobs = np.random.multivariate_normal(mean=zmean, cov=zvar)
                zobs = np.reshape(zobs, (n_points, 1))
            else:
                zobs = belief.posterior_samples(xobs, full_cov=False, size=1)
            belief.add_data(xobs, zobs)

            pose_new = current_node.action[-1]
            child = Node(pose=pose_new,
                         parent=current_node,
                         name=current_node.name + '_belief' + str(current_node.depth + 1),
                         action=None,
                         zvals=zobs)
            current_node.add_children(child)

            # Recursive call
            return self.leaf_helper(child, reward + r, belief)

    def get_next_child(self, current_node):
        vals = {}
        e_d = 0.5 * (1.0 - (3.0/(10.0*(self.max_depth - current_node.depth))))
        for child in current_node.children:
            if child.nqueries == 0:
                return child
            vals[child] = child.reward/float(child.nqueries) + self.c * np.sqrt((float(current_node.nqueries) ** e_d)/float(child.nqueries))
        # Return the max node, or a random node if the value is equal
        return random.choice([key for key in vals.keys() if vals[key] == np.nanmax(vals.values())])

    def build_action_children(self, parent):
        actions = self.path_generator.generate_trajectories(parent.pose, self.t, self.obs_world, self.use_sim_world)
        if len(actions) == 0:
            print "No actions!"
            return

        #print "Creating children for:", parent.name
        for i, action in enumerate(actions):
            #print "Action:", i
            parent.add_children(Node(pose=parent.pose,
                                     parent=parent,
                                     name=parent.name + '_action' + str(i),
                                     action=action,
                                     zvals=None))

    def print_tree(self):
        counter = self.print_helper(self.root)
        print "# nodes in tree:", counter

    def print_helper(self, cur_node):
        if cur_node.children is None:
            return 1
        else:
            counter = 0
            for child in cur_node.children:
                counter += self.print_helper(child)
            return counter

class BeliefTree(Tree):
    ''' Inherit class, that implements more standard MCTS, and assumes MLE observation to deal with continuous spaces '''
    def __init__(self, f_rew, f_aqu, pose, path_generator, t, depth, param, c, obs_world, use_sim_world=True):
        super(BeliefTree, self).__init__(f_rew, f_aqu, pose, path_generator, t, depth, param, c, obs_world, use_sim_world)

    def get_best_child(self):
        # Max Reward-based node selection
        # return random.choice([node for node in self.root.children if node.nqueries == np.nanmax([n.nqueries for n in self.tree.children])])
        return self.root.children[np.argmax([node.nqueries for node in self.root.children])]
        # vals = {}
        # for child in self.root.children:
        #     if child.nqueries == 0:
        #         return child, False
        #     vals[child] = child.reward/float(child.nqueries)
        # # Return the max node, or a random node if the value is equal
        # return random.choice([key for key in vals.keys() if vals[key] == max(vals.values())])

    def random_rollouts(self, current_node, reward, belief):
        cur_depth = current_node.depth
        pose = current_node.pose
        belief = copy.deepcopy(belief)
        while cur_depth <= self.max_depth:
            actions = self.path_generator.generate_trajectories(pose, self.t, self.obs_world, self.use_sim_world)
            # No viable trajectories from current location
            if len(actions) == 0:
                return reward

            #select a random action
            try:
                a = random.randint(0, len(actions) - 1)
            except:
                a = 0
            obs = np.array(actions[a])

            if belief.dim == 2:
                xobs = np.vstack([obs[:, 0], obs[:, 1]]).T
            elif belief.dim == 3:
                xobs = np.vstack([obs[:, 0], obs[:, 1], (self.t+cur_depth) * np.ones(obs.shape[0])]).T

            if self.f_rew == 'mes' or self.f_rew == 'exp_improve':
                r = self.aquisition_function(time=self.t+cur_depth, xvals=xobs, robot_model=belief, param=self.param)
            elif self.f_rew == 'gumbel':
                param = sample_max_vals_gumbel(belief, t=self.t+cur_depth, obstacles=self.obs_world)
                r = self.aquisition_function(time=self.t+cur_depth, xvals=xobs, robot_model=belief, param=param)

            else:
                r = self.aquisition_function(time=self.t+cur_depth, xvals=xobs, robot_model=belief)

            # ''Simulate'' the maximum likelihood observation
            if belief.xvals is None:
                n_points, _ = xobs.shape
                zobs = np.zeros((n_points, ))
                zobs = np.reshape(zobs, (n_points, 1))
            else:
                zobs, _ = belief.predict_value(xobs)

            belief.add_data(xobs, zobs)

            pose = actions[a][-1]
            reward += r
            cur_depth += 1

        return reward

    def leaf_helper(self, current_node, reward, belief):
        belief = copy.deepcopy(belief)
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

                child, full_action_set = self.get_next_child(current_node)

                if full_action_set:
                    # Recursive call
                    return self.leaf_helper(child, reward, belief)
                else:
                    # Do random rollouts
                    rollout_reward = self.random_rollouts(current_node, reward, belief)
                    return child, rollout_reward

        # At random node, after selected action from a specific node
        elif current_node.node_type == 'BA':
            # Sample a new set of observations and form a new belief
            obs = np.array(current_node.action)
            if belief.dim == 2:
                xobs = np.vstack([obs[:, 0], obs[:, 1]]).T
            elif belief.dim == 3:
                xobs = np.vstack([obs[:, 0], obs[:, 1], (self.t+current_node.depth) * np.ones(obs.shape[0])]).T

            if self.f_rew == 'mes' or self.f_rew == 'exp_improve':
                r = self.aquisition_function(time=self.t+current_node.depth, xvals=xobs, robot_model=belief, param=self.param)
            elif self.f_rew == 'gumbel':
                param = sample_max_vals_gumbel(belief, t=self.t+current_node.depth, obstacles=self.obs_world)
                r = self.aquisition_function(time=self.t+current_node.depth, xvals=xobs, robot_model=belief, param=param)

            else:
                r = self.aquisition_function(time=self.t+current_node.depth, xvals=xobs, robot_model=belief)

            # ''Simulate'' the maximum likelihood observation
            if belief.xvals is None:
                n_points, _ = xobs.shape
                zobs = np.zeros((n_points, ))
                zobs = np.reshape(zobs, (n_points, 1))
            else:
                zobs, _ = belief.predict_value(xobs)

            belief.add_data(xobs, zobs)

            pose_new = current_node.action[-1]
            child = Node(pose=pose_new,
                         parent=current_node,
                         name=current_node.name + '_belief' + str(current_node.depth + 1),
                         action=None,
                         zvals=zobs)
            #print "Adding next belief child:", child.name
            current_node.add_children(child)

            # Recursive call
            return self.leaf_helper(child, reward + r, belief)

    def get_next_child(self, current_node):
        ''' Returns the next most promising child of a belief node, and a FLAG indicating if belief node is fully explored '''
        vals = {}
        for child in current_node.children:
            if child.nqueries == 0:
                return child, False
            vals[child] = child.reward/float(child.nqueries) + self.c * np.sqrt(2.0*np.log(float(current_node.nqueries))/float(child.nqueries))
        # Return the max node, or a random node if the value is equal
        return random.choice([key for key in vals.keys() if vals[key] == max(vals.values())]), True

class cMCTS(MCTS):
    '''Class that establishes a MCTS for nonmyopic planning'''
    def __init__(self, computation_budget, belief, initial_pose, rollout_length, path_generator, aquisition_function, f_rew, T, obs_world, aq_param = None, tree_type = 'dpw', use_sim_world=True):
        # Call the constructor of the super class
        super(cMCTS, self).__init__(computation_budget, belief, initial_pose, rollout_length, path_generator, aquisition_function, f_rew, T, obs_world, aq_param)
        self.tree_type = tree_type
        self.obs_world = obs_world
        self.use_sim_world = use_sim_world
        self.param = aq_param
        self.GP = copy.deepcopy(belief)

        # The differnt constatns use logarthmic vs polynomical exploriation
        if self.f_rew == 'mean':
            if self.tree_type == 'belief':
                self.c = 1000.
            elif self.tree_type == 'dpw':
                self.c = 5000.
        elif self.f_rew == 'exp_improve':
            self.c = 200.
        elif self.f_rew == 'mes' or self.f_rew == 'gumbel':
            if self.tree_type == 'belief':
                self.c = 1.0 / np.sqrt(2.0)
            elif self.tree_type == 'dpw':
                self.c = 1.0 / np.sqrt(2.0)
        else:
            self.c = 1.0
        print "Setting c to :", self.c

    def choose_trajectory(self, t):
        #Main function loop which makes the tree and selects the best child
        #Output: path to take, cost of that path

        # randomly sample the world for entropy search function
        if self.f_rew == 'mes':
            self.max_val, self.max_locs, self.target = sample_max_vals(self.GP, t=t, obstacles=self.obs_world)
            param = (self.max_val, self.max_locs, self.target)
        elif self.f_rew == 'exp_improve':
            param = self.param
        # elif self.f_rew == 'gumbel':
        #     param = sample_max_vals_gumbel(self.GP, t=t, obstacles=self.obs_world)
        else:
            param = None

        # initialize tree
        if self.tree_type == 'dpw':
            self.tree = Tree(self.f_rew, self.aquisition_function, self.cp, self.path_generator, t, depth=self.rl, param=param, c=self.c, obs_world=self.obs_world, use_sim_world=self.use_sim_world)
        elif self.tree_type == 'belief':
            self.tree = BeliefTree(self.f_rew, self.aquisition_function, self.cp, self.path_generator, t, depth=self.rl, param=param, c=self.c, obs_world=self.obs_world, use_sim_world=self.use_sim_world)
        else:
            raise ValueError('Tree type must be one of either \'dpw\' or \'belief\'')

        time_start = time.time()
        # while we still have time to compute, generate the tree
        i = 0
        gp = copy.copy(self.GP)
        while i < self.comp_budget:
            i += 1
            self.tree.get_next_leaf(gp)
            gp = copy.copy(self.GP)
        time_end = time.time()
        print "Rollouts completed in", str(time_end - time_start) +  "s"
        print "Number of rollouts:", i
        self.tree.print_tree()

        # best_child = self.tree.root.children[np.argmax([node.nqueries for node in self.tree.root.children])]
        best_child = random.choice([node for node in self.tree.root.children if node.nqueries == np.nanmax([n.nqueries for n in self.tree.root.children])])
        all_vals = {}
        actions = []
        for i, child in enumerate(self.tree.root.children):
            all_vals[i] = child.reward / float(child.nqueries)
            actions.append(child.action)
        return np.array(best_child.action), best_child.reward/float(best_child.nqueries), actions, all_vals, self.max_locs, self.max_val
