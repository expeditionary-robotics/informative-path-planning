# !/usr/bin/python

'''
Script for running myopic experiments using the run_mysim bash script.
Keep in mind that some of the parameters may need to be adjusted before running\
which are not necessarily set by the command line interface!

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''
import os
import sys
import logging
import numpy as np

import heuristic_rewards as aqlib
import gpmodel_library as gplib 
import mission_logger as evalib 
import generate_actions as pathlib 
import phenomenon_simulator as envlib 
import simulation_agent as roblib
import generate_metric_environment as obslib

# Allow selection of seed world to be consistent, and to run through reward functions
seed =  0#int(sys.argv[1])
reward_function = 'mean'#sys.argv[2]

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

world = envlib.Phenomenon(ranges = ranges,
                          NUM_PTS = 20, 
                          variance = 100.0, 
                          lengthscale = 1., 
                          kparams={'lengthscale':(1.5, 1.5), 'variance':(100.,100.)},
                          dim=2,
                          kernel='rbf',
                          seed=seed,
                          metric_world=obslib.World(ranges),
                          time_duration=1,
                          MIN_COLOR=MIN_COLOR, 
                          MAX_COLOR=MAX_COLOR)

# Create the evaluation class used to quantify the simulation metrics
evaluation = evalib.Evaluation(world = world, reward_function = reward_function)

# Populate a world with obstacles
# ow = obslib.FreeWorld()
# ow = obslib.BlockWorld(ranges, 1, dim_blocks= (2,2), centers=[(7,7)])
# ow = obslib.ChannelWorld(ranges, (2.5,7), 3, 0.2)

ow = obslib.World(ranges)
paths = pathlib.ActionSet(num_actions=15,
                          length=1.5,
                          turning_radius=0.05,
                          radius_angle=np.pi/4,
                          num_samples=3,
                          safe_threshold=50.,
                          unknown_threshold=-2.,
                          allow_reverse=True,
                          allow_stay=False)

# Create the point robot
robot = roblib.Robot(sample_world = world.sample_value, #function handle for collecting observations
                     start_loc = (5.0, 5.0, 0.0), #where robot is instantiated
                     start_time=0,
                     extent = ranges, #extent of the explorable environment
                     dimension=2,
                     kernel_file = None,
                     kernel_dataset = None,
                     prior_dataset =  None, #(data, observations),
                     kernel='rbf',
                     kparams={'lengthscale':(1.5, 1.5), 'variance':(100.,100.)}, 
                     init_lengthscale = 1.0, 
                     init_variance = 100.0, 
                     noise = 0.5,
                     path_generator = paths,
                     evaluation = evaluation, 
                     f_rew = reward_function, 
                     create_animation = True, #logs images to the file folder
                     nonmyopic=True, #select if you want to use MCTS
                     MIN_COLOR=MIN_COLOR,
                     MAX_COLOR=MAX_COLOR,
                     obstacle_world=ow,
                     running_simulation=True,
                     tree_type='dpw',
                     rollout_length=7,
                     computation_budget=200) 

robot.planner(T = 50)
#robot.visualize_world_model(screen = True)
robot.visualize_trajectory(screen = False) #creates a summary trajectory image
