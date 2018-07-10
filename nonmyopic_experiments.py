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
import obstacles as obslib

print "User specified options: SEED, REWARD_FUNCTION, PATHSET, USE_COST, NONMYOPIC, GOAL_ONLY"
# Allow selection of seed world to be consistent, and to run through reward functions
SEED =  int(sys.argv[1])
REWARD_FUNCTION = sys.argv[2]
PATHSET = sys.argv[3]
USE_COST = (sys.argv[4] == "True")
NONMYOPIC = (sys.argv[5] == "True")
GOAL_ONLY = (sys.argv[6] == "True")
TREE_TYPE = sys.argv[7] # one of dpw or belief

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
#ow = obslib.BlockWorld(ranges,13, dim_blocks=(1., 1.), centers=[(2.5, 2.5), (0.75, 8.25), (7.,4.), (5., 8.), (9., 7.), (3.5,6.), (6.,1.5), (1.75,5.), (6.2,6.), (8.,9.), (4.2, 3.8), (9.,2.5), (2.5,8.)])

world = envlib.Environment(ranges = ranges,
                           NUM_PTS = 20, 
                           variance = 100.0, 
                           lengthscale = 1.0, 
                           visualize = True,
                           seed = SEED,
                           MIN_COLOR=MIN_COLOR, 
                           MAX_COLOR=MAX_COLOR, 
                           obstacle_world = ow)

# Create the evaluation class used to quantify the simulation metrics
evaluation = evalib.Evaluation(world = world, reward_function = REWARD_FUNCTION)

# Create the point robot
robot = roblib.Robot(sample_world = world.sample_value, #function handle for collecting observations
                     start_loc = (5.0, 5.0, 0.0), #where robot is instantiated
                     extent = ranges, #extent of the explorable environment
                     kernel_file = None,
                     kernel_dataset = None,
                     #prior_dataset =  (data, observations), 
                     prior_dataset = None,
                     init_lengthscale = 1.0, 
                     init_variance = 100.0, 
                     noise = 1.0001,
                     #noise = 0.5000,
                     path_generator = PATHSET, #options: default, dubins, equal_dubins, fully_reachable_goal, fully_reachable_step
                     goal_only = GOAL_ONLY, #select only if using fully reachable step and you want the reward of the step to only be the goal
                     frontier_size = 15,
                     horizon_length = 1.5, 
                     turning_radius = 0.05,
                     sample_step = 0.5,
                     evaluation = evaluation, 
                     f_rew = REWARD_FUNCTION, 
                     create_animation = True, #logs images to the file folder
                     learn_params = False, #if kernel params should be trained online
                     nonmyopic = NONMYOPIC,
                     discretization = (20, 20), #parameterizes the fully reachable sets
                     use_cost = USE_COST, #select if you want to use a cost heuristic
                     MIN_COLOR = MIN_COLOR,
                     MAX_COLOR = MAX_COLOR,
                     computation_budget = 250.0,
                     rollout_length = 5,
                     obstacle_world = ow, 
                     tree_type = TREE_TYPE) 

robot.planner(T = 150)
#robot.visualize_world_model(screen = True)
robot.visualize_trajectory(screen = False) #creates a summary trajectory image
robot.plot_information() #plots all of the metrics of interest

