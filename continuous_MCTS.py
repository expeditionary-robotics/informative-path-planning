# !/usr/bin/python

'''
Based on the library of PLUMES, extend Monte Carlo Tree Search class to continuous-action case. 

This library allows access to the Monte Carlo Tree Search class used in the PLUMES framework.
A MCTS allows for performing many forward simulation of multiple-chained actions in order to 
select the single most promising action to take at some time t. We have presented a variation
of the MCTS by forward simulating within an incrementally updated GP belief world.

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
from mcts_library import *
import continuous_traj as traj

class conti_action_MCTS(MCTS):
    '''
    Class inherited from MCTS class. 
    '''
    def __init__(self, time, computation_budget, belief, initial_pose, rollout_length, path_generator, aquisition_function, f_rew, T, aq_param = None, use_cost = False, tree_type = 'dpw'):
        super(conti_action_MCTS, self).__init__(computation_budget, belief, initial_pose, rollout_length, path_generator, aquisition_function, f_rew, T, aq_param = None, use_cost = False, tree_type = 'dpw')
        self.t = time

    def simulate(self, t):

        self.initialize_tree()

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
            self.update_tree(reward, cost, sequence) #Backpropagation
            value_grad = self.get_value_gradient()
            self.update_action(value_grad)

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


    def update_action(self, value_grad):
        '''
        Based on the gradient value, update action and re-iterate generation of trajectory 
        '''
        grad = self.get_value_gradient
        # self.tree

    def get_value_gradient(self):
        val_gradient = 0.0

        return val_gradient


if __name__ == "__main__":
    print("Hello")
    sample_step = 0.5
    ranges = (0., 10., 0., 10.)
    start = (0.25, 0.25, 0.0)
    path_length = 1.5*175
    coverage_path = [start]

    across = 9.75
    rise = 0.38
    cp = start
    waypoints = [cp]
    l = 0

    for i in range(0,51):
        if i%2 == 0:
            if cp[0] > ranges[1]/2:
                cp = (cp[0]-across+0.25, cp[1], cp[2])
                l += across
            else:
                cp = (cp[0]+across-0.25, cp[1], cp[2])
                l += across
        else:
            cp = (cp[0], cp[1]+rise, cp[2])
            l += rise
        waypoints.append(cp)

    x = [w[0] for w in waypoints]
    y = [w[1] for w in waypoints]

    samples = [start]
    extra = 0
    addit = 0
    last = start
    for i,w in enumerate(waypoints):
        if i%4 == 0:
            last = w[0]
            while last+sample_step <= waypoints[i+1][0]:
                samples.append((last+sample_step, w[1], w[2])) 
                last = samples[-1][0]
            remainder = across-last
        elif (i+1)%4 == 0:
            last = waypoints[i-1][0]
            while last-sample_step+(remainder) >= waypoints[i][0]:
                samples.append((last-sample_step+(remainder), w[1], w[2])) 
                last = samples[-1][0]
            remainder = across-last

    xs = [s[0] for s in samples]
    ys = [s[1] for s in samples]



    reward_function = 'mes'
    ranges = (0., 10., 0., 10.)

    world = Environment(ranges = ranges, # x1min, x1max, x2min, x2max constraints
                        NUM_PTS = 20, 
                        variance = 100.0, 
                        lengthscale = 1.0, 
                        visualize = True,
                        seed = 3)

    evaluation = Evaluation(world = world, 
                            reward_function = reward_function)

    # Gather some prior observations to train the kernel (optional)
    x1observe = np.linspace(ranges[0]+0.5, ranges[1]-0,5, 8)
    x2observe = np.linspace(ranges[2]+0.5, ranges[3]-0.5, 8)
    x1observe, x2observe = np.meshgrid(x1observe, x2observe, sparse = False, indexing = 'xy')  
    data = np.vstack([x1observe.ravel(), x2observe.ravel()]).T
    observations = world.sample_value(data)

    # Create the point robot
    robot = Nonmyopic_Robot(sample_world = world.sample_value, 
                start_loc = (5.0, 5.0, 0.0), 
                extent = ranges,
                kernel_file = None,
                kernel_dataset = None,
                prior_dataset =  None, 
                #prior_dataset =  (data, observations), 
                init_lengthscale = 1.0, 
                init_variance = 100.0, 
                noise = 0.0001,
                path_generator = 'default',
                frontier_size = 20, 
                horizon_length = 1.5, 
                turning_radius = 0.05, 
                sample_step = 0.5,
                evaluation = evaluation, 
                f_rew = reward_function, 
                create_animation = True,
                computation_budget = 5,
                rollout_length = 3) 

    robot.planner(T = 20)
    robot.visualize_world_model(screen = True)
    robot.visualize_trajectory(screen = True)
    robot.plot_information()

    # const_MCTS = conti_action_MCTS(5.0)
    # const_MCTS.initialize_tree()
    