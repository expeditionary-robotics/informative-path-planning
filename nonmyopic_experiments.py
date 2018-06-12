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

import aq_library as aqlib
import mcts_library as mctslib
import gpmodel_library as gplib 
import evaluation_library as evalib 
import paths_library as pathlib 
import envmodel_library as envlib 
import robot_library as roblib

# Allow selection of seed world to be consistent, and to run through reward functions
seed =  0#int(sys.argv[1])
reward_function = 'mes'#sys.argv[2]

# Parameters for plotting based on the seed world information
MIN_COLOR = -25.
MAX_COLOR = 25.

# Set up paths for logging the data from the simulation run
if not os.path.exists('./figures/' + str(reward_function)): 
    os.makedirs('./figures/' + str(reward_function))
logging.basicConfig(filename = './figures/'+ reward_function + '/robot.log', level = logging.INFO)
logger = logging.getLogger('robot')

# Create a random enviroment sampled from a GP with an RBF kernel and specified hyperparameters, mean function 0 
# The enviorment will be constrained by a set of uniformly distributed  sample points of size NUM_PTS x NUM_PTS
ranges = (0., 10., 0., 10.)

world = envlib.Environment(ranges = ranges,
                           NUM_PTS = 20, 
                           variance = 100.0, 
                           lengthscale = 1.0, 
                           visualize = True,
                           seed = seed,
                           MIN_COLOR=MIN_COLOR, 
                           MAX_COLOR=MAX_COLOR)

# Create the evaluation class used to quantify the simulation metrics
evaluation = evalib.Evaluation(world = world, reward_function = reward_function)


x1observe = np.linspace(0., 10., 20)
x2observe = np.linspace(0., 10., 20)
x1observe, x2observe = np.meshgrid(x1observe, x2observe, sparse = False, indexing = 'xy')  
data = np.vstack([x1observe.ravel(), x2observe.ravel()]).T
observations = world.sample_value(data)

# Create the point robot
robot = roblib.Robot(sample_world = world.sample_value, #function handle for collecting observations
                     start_loc = (5.0, 5.0, 0.0), #where robot is instantiated
                     extent = ranges, #extent of the explorable environment
                     kernel_file = None,
                     kernel_dataset = None,
                     prior_dataset =  None, #(data, observations), 
                     init_lengthscale = 1.0, 
                     init_variance = 100.0, 
                     noise = 0.0001,
                     path_generator = 'dubins', #options: default, dubins, equal_dubins, fully_reachable_goal, fully_reachable_step
                     goal_only = False, #select only if using fully reachable step and you want the reward of the step to only be the goal
                     frontier_size = 5,
                     horizon_length = 1.5, 
                     turning_radius = 0.05,
                     sample_step = 0.5,
                     evaluation = evaluation, 
                     f_rew = reward_function, 
                     create_animation = True, #logs images to the file folder
                     learn_params=False, #if kernel params should be trained online
                     nonmyopic=True, #select if you want to use MCTS
                     discretization=(20,20), #parameterizes the fully reachable sets
                     use_cost=False, #select if you want to use a cost heuristic
                     MIN_COLOR=MIN_COLOR,
                     MAX_COLOR=MAX_COLOR,
                     computation_budget=100.0) 

robot.planner(T = 175)
#robot.visualize_world_model(screen = True)
robot.visualize_trajectory(screen = False) #creates a summary trajectory image
robot.plot_information() #plots all of the metrics of interest

