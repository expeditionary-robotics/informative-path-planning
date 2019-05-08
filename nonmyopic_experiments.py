# !/usr/bin/python

'''
Script for running myopic experiments using the run_sim bash script.
Generally a function of convenience in the event of parallelizing simulation runs.
Note: some of the parameters may need to be set prior to running the bash script.

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''
import os
import time
import sys
import logging
import numpy as np
import argparse

import aq_library as aqlib
import mcts_library as mctslib
import gpmodel_library as gplib 
import evaluation_library as evalib 
import paths_library as pathlib 
import envmodel_library as envlib 
import robot_library as roblib
import obstacles as obslib

# Initialize command line options
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", action="store", type=int, help="Random seed for environment generation.", default = 0)
parser.add_argument("-r", "--reward", action="store", help="Reward function. Should be mes, ei, or info_gain.", default = 'naive_value')
parser.add_argument("-p", "--pathset", action="store", help="Action set type. Should be dubins, ...", default = 'dubins')
parser.add_argument("-t", "--tree", action="store", help="If using nonmyopic planner, what kind of tree serach. Should be dpw or belief.", default = 'dpw')
parser.add_argument("-n", "--nonmyopic", action="store_true", help="Run planner in nonmyopic mode if flag set.", default = False)
parser.add_argument("-c", "--cost", action="store_true", help="Divide reward of action by cost if flag set.", default = False)
parser.add_argument("-g", "--goal", action="store_true", help="Consider the reward of final point only if flag set.", default = False)


# Parse command line options
parse = parser.parse_args()

# Read command line options
SEED =  parse.seed
REWARD_FUNCTION = parse.reward
PATHSET = parse.pathset
USE_COST = parse.cost
NONMYOPIC = parse.nonmyopic
GOAL_ONLY = parse.goal
TREE_TYPE = parse.tree
DIM = 2 #2
DURATION = 1
LENGTHSCALE = 1# (2.5, 2.5, 30) #1

# Parameters for plotting based on the seed world information
MIN_COLOR = -25.
MAX_COLOR = 25.

# Set up paths for logging the data from the simulation run
if not os.path.exists('./figures/' + str(REWARD_FUNCTION)): 
    os.makedirs('./figures/' + str(REWARD_FUNCTION))
logging.basicConfig(filename = './figures/'+ REWARD_FUNCTION + '/robot.log', level = logging.INFO)
logger = logging.getLogger('robot')

# Create a random enviroment sampled from a GP with an RBF kernel and specified hyperparameters, mean function 0 
# The enviorment will be constrained by a set of uniformly distributed  sample points of size NUM_PTS x NUM_PTS
ranges = (0., 10., 0., 10.)

# Create obstacle world
ow = obslib.FreeWorld()
# ow = obslib.ChannelWorld(ranges, (3.5, 7.), 3., 0.3)
# ow = obslib.BugTrap(ranges, (2.2, 3.0), 4.6, orientation = 'left', width = 5.0)
# ow = obslib.BlockWorld(ranges,12, dim_blocks=(1., 1.), centers=[(2.5, 2.5), (7.,4.), (5., 8.), (8.75, 6.), (3.5,6.), (6.,1.5), (1.75,5.), (6.2,6.), (8.,8.5), (4.2, 3.8), (8.75,2.5), (2.2,8.2)])

world = envlib.Environment(ranges = ranges,
                           NUM_PTS = 20, 
                           variance = 100.0, # TODO: why is only the lengthscale a vector for asymmetric kernels? 
                           lengthscale = LENGTHSCALE,
                           noise = 0.5,
                           dim = DIM,
                           visualize = True,
                           seed = SEED,
                           MIN_COLOR = MIN_COLOR, 
                           MAX_COLOR = MAX_COLOR, 
                           obstacle_world = ow, 
                           time_duration = DURATION)
                           # noise= 5.0)

# Create the evaluation class used to quantify the simulation metrics
evaluation = evalib.Evaluation(world = world, reward_function = REWARD_FUNCTION)

# Generate a prior dataset
x1observe = np.linspace(ranges[0], ranges[1], 20)
x2observe = np.linspace(ranges[2], ranges[3], 20)
x1observe, x2observe = np.meshgrid(x1observe, x2observe, sparse = False, indexing = 'xy')  
if DIM == 2:
    data = np.vstack([x1observe.ravel(), x2observe.ravel()]).T
elif DIM == 3:
    data = np.vstack([x1observe.ravel(), x2observe.ravel(), 0*np.ones(len(x1observe.ravel()))]).T
observations = world.sample_value(data)


# Define the algorithm parameters
kwargs = {  'sample_world': world.sample_value,
            'start_loc': (5.0, 5.0, 0.0), #where robot is instantiated
            'start_time' : 0.0, #time at which robot is instantiated
            'extent': ranges, #extent of the explorable environment
            'kernel_file': None,
            'kernel_dataset': None,
            #'prior_dataset':  (data, observations), 
            'prior_dataset': None,
            'init_lengthscale': LENGTHSCALE, 
            'init_variance': 100.0, 
            'noise': 0.5000,
            'path_generator': PATHSET, #options: default, dubins, equal_dubins, fully_reachable_goal, fully_reachable_step
            'goal_only': GOAL_ONLY, #select only if using fully reachable step and you want the reward of the step to only be the goal
            'frontier_size': 15,
            'horizon_length': 1.5, 
            'turning_radius': 0.05,
            'sample_step': 0.5,
            'evaluation': evaluation, 
            'f_rew': REWARD_FUNCTION, 
            'create_animation': True, #logs images to the file folder
            'learn_params': False, #if kernel params should be trained online
            'nonmyopic': NONMYOPIC,
            'discretization': (20, 20), #parameterizes the fully reachable sets
            'use_cost': USE_COST, #select if you want to use a cost heuristic
            'MIN_COLOR': MIN_COLOR,
            'MAX_COLOR': MAX_COLOR,
            'computation_budget': 250,
            'rollout_length': 5,
            'obstacle_world' : ow, 
            'tree_type': TREE_TYPE,
            'dimension': DIM}


# Create the point robot
robot = roblib.Robot(**kwargs) 

robot.planner(T = 150)
#robot.visualize_world_model(screen = True)
robot.visualize_trajectory(screen = False) #creates a summary trajectory image
robot.plot_information() #plots all of the metrics of interest
