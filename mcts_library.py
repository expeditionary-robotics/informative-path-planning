# !/usr/bin/python

'''
This library allows access to the Monte Carlo Tree Search class used in the PLUMES framework

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''

from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib import cm
from sklearn import mixture
from IPython.display import display
from scipy.stats import multivariate_normal
import numpy as np
import scipy as sp
import math
import os
import GPy as GPy
import dubins
import time
from itertools import chain
import pdb
import logging
logger = logging.getLogger('robot')
from aq_library import *
import copy

class MCTS:
    '''Class that establishes a MCTS for nonmyopic planning'''

    def __init__(self, computation_budget, belief, initial_pose, rollout_length, path_generator, aquisition_function, f_rew, time, aq_param = None):
        '''Initialize with constraints for the planning, including whether there is a budget or planning horizon
        Inputs:
            computation_budget (float) number of seconds to run the tree building procedure
            belief (GP model) current belief of the vehicle
            initial_pose (tuple of floats) location of the vehicle in world coordinates
            rollout_length (int) number of actions to rollout after selecting a child (tree depth)
            frontier_size (int) number of options for each action in the tree (tree breadth)
            path_generator (string) how action sets should be developed
            aquisition_function (function) the criteria to make decisions
            f_rew (string) the name of the function used to make decisions
            time (float) time in the global world used for aquisition weighting
        '''

        # Parameterization for the search
        self.comp_budget = computation_budget
        self.GP = belief
        self.cp = initial_pose
        self.rl = rollout_length
        self.path_generator = path_generator
        self.aquisition_function = aquisition_function
        self.f_rew = f_rew
        self.t = time

        # The tree
        self.tree = None
        
        # Elements which are relevant for some acquisition functions
        self.params = None
        self.max_val = None
        self.max_locs = None
        self.current_max = aq_param

    def choose_trajectory(self, t, loc=None):
        ''' Main function loop which makes the tree and selects the best child
        Output:
            path to take, cost of that path
        '''
        # initialize tree
        self.tree = self.initialize_tree() 
        i = 0 #iteration count

        # randonly sample the world for entropy search function
        if self.f_rew == 'mes':
            self.max_val, self.max_locs = sample_max_vals(self.GP, t = t)
            
        time_start = time.clock()            
            
        # while we still have time to compute, generate the tree
        while time.clock() - time_start < self.comp_budget:
            i += 1
            current_node = self.tree_policy()
            sequence = self.rollout_policy(current_node)
            reward, cost= self.get_reward(sequence, loc)
            self.update_tree(reward, cost, sequence)

        # get the best action to take with most promising futures
        if loc is None:
        	best_sequence, best_val, all_vals = self.get_best_child()
    	else:
    		best_sequence, best_val, all_vals = self.get_best_child(True)
        print "Number of rollouts:", i, "\t Size of tree:", len(self.tree)
        logger.info("Number of rollouts: {} \t Size of tree: {}".format(i, len(self.tree)))

        paths = self.path_generator.get_path_set(self.cp)                
        return self.tree[best_sequence][0], best_val, paths, all_vals, self.max_locs, self.max_val

    def initialize_tree(self):
        '''Creates a tree instance, which is a dictionary, that keeps track of the nodes in the world
        Output:
            tree (dictionary) an initial tree
        '''
        tree = {}
        # root of the tree is current location of the vehicle
        tree['root'] = (self.cp, 0) #(pose, number of queries)
        actions = self.path_generator.get_path_set(self.cp)
        for action, samples in actions.items():
            tree['child '+ str(action)] = (samples, 0, 0, 0) #(samples, cost, reward, number of times queried)
        return tree

    def tree_policy(self):
        '''Implements the UCB policy to select the child to expand and forward simulate. From Arora paper, the following is defined:
            avg_r - average reward of all rollouts that have passed through node n
            c_p - some arbitrary constant, they use 0.1
            N - number of times parent has been evaluated
            n - number of times that node has been evaluated
            the formula: avg_r + c_p * np.sqrt(2*np.log(N)/n)
        '''
        leaf_eval = {}
        # TODO: check initialization, when everything is zero. appears to be throwing error
        actions = self.path_generator.get_path_set(self.cp)
        for i, val in actions.items():
            try:
                node = 'child '+ str(i)
                leaf_eval[node] = self.tree[node][2] + 0.1*np.sqrt(2*(np.log(self.tree['root'][1]))/self.tree[node][3])
            except:
                pass
        return max(leaf_eval, key=leaf_eval.get)

    def rollout_policy(self, node):
        '''Select random actions to expand the child node
        Input:
            node (the name of the child node that is to be expanded)
        Output:
            sequence (list of names of nodes that make the sequence in the tree)
        '''

        sequence = [node] #include the child node
        for i in xrange(self.rl):
            actions = self.path_generator.get_path_set(self.tree[node][0][-1]) #plan from the last point in the sample
            if len(actions) == 0:
                print 'No actions were viably generated'
            try:
                
                try:
                    a = np.random.randint(0,len(actions)-1) #choose a random path
                except:
                    if len(actions) != 0:
                        a = 0

                keys = actions.keys()
                if len(keys) <= 1:
                    print 'few paths available!'
                self.tree[node + ' child ' + str(keys[a])] = (actions[keys[a]], 0, 0, 0) #add random path to the tree
                node = node + ' child ' + str(keys[a])
                sequence.append(node)
            except:
                print 'rolling back'
                sequence.remove(node)
                try:
                    node = sequence[-1]
                except:
                    print "Empty sequence", sequence, node
                    logger.warning('Bad sequence')
        return sequence

    def get_reward(self, sequence, loc=None):
        '''Evaluate the sequence to get the reward, defined by the percentage of entropy reduction.
        Input:
            sequence (list of strings) names of the nodes in the tree
        Outut:
            reward value from the aquisition function of choice
        '''
        sim_world = copy.copy(self.GP)
        samples = []
        obs = []
        cost = 0
        for seq in sequence:
        	# for i in self.tree[seq][0]:
        	# 	samples.append(i)
        	samples.append(self.tree[seq][0])
        obs = list(chain.from_iterable(samples))
        if loc is not None:
        	cost = self.path_generator.path_cost(self.tree[seq][0], loc)

        reward = 0
        for s in samples:
        	obs = np.array(s)
        	xobs = np.vstack([obs[:,0], obs[:,1]]).T
	        if self.f_rew == 'mes':
	            reward += self.aquisition_function(time = self.t, xvals = xobs, robot_model = sim_world, param = self.max_val)
	        elif self.f_rew == 'exp_improve':
	            reward += self.aquisition_function(time=self.t, xvals = xobs, robot_model = sim_world, param = [self.current_max])
	        else:
	            reward += self.aquisition_function(time=self.t, xvals = xobs, robot_model = sim_world)
	        # xobs = np.array(obs)
	        zmean, zvar = sim_world.predict_value(xobs)
	        zobs = []
	        for m,v in zip(zmean, zvar):
	        	zobs.append(np.random.normal(m, np.sqrt(v), 1))
	        # zobs = np.random.normal(zmean, np.sqrt(zvar[0][0]), 1)
	        sim_world.add_data(xobs, zobs)
        return reward, cost

    
    def update_tree(self, reward, cost, sequence):
        '''Propogate the reward for the sequence
        Input:
            reward (float) the reward or utility value of the sequence
            sequence (list of strings) the names of nodes that form the sequence
        '''
        self.tree['root'] = (self.tree['root'][0], self.tree['root'][1]+1)
        path_cost = 0
        for seq in sequence:
            samples, cos, rew, queries = self.tree[seq]
            queries += 1
            n = queries
            rew = ((n-1)*rew+reward)/n
            path_cost += self.path_generator.path_cost(self.tree[seq][0])
            cos = ((n-1)*cos+(cost+path_cost))/n
            self.tree[seq] = (samples, cos, rew, queries)

    def get_best_child(self, use_cost=False):
        '''Query the tree for the best child in the actions
        Output:
            (string, float) node name of the best child, the cost of that child
        '''
        best = -float('inf')
        best_child = None
        value = {}
        actions = self.path_generator.get_path_set(self.cp)
        keys = actions.keys()
        for i in keys:
            try:
                if use_cost == False:
                	r = self.tree['child '+ str(i)][2]
                	value[i] = r
            	else:
            		r = self.tree['child '+ str(i)][2]/self.tree['child ' + str(i)][3]
            		value[i] = r
                if r > best: 
                    best = r
                    best_child = 'child '+ str(i)
            except:
                pass
        return best_child, best, value