# !/usr/bin/python

'''
Script for running myopic experiments using the run_mysim bash script.
Keep in mind that some of the parameters may need to be adjusted before running\
which are not necessarily set by the command line interface!

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''
import os
import logging
import argparse
import numpy as np

import mission_logger as evalib
import generate_actions as pathlib
import phenomenon_simulator as envlib
import simulation_agent as roblib
import generate_metric_environment as obslib

# Allow selection of seed world to be consistent, and to run through reward functions
# Initialize command line options
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", action="store", type=int, help="Random seed for environment generation.", default = 0)
parser.add_argument("-r", "--reward", action="store", help="Reward function. Should be mes, mean, gumbel, naive, or naive_value, ei, or info_gain.", default = 'gumbel')
parser.add_argument("-t", "--tree", action="store", help="If using nonmyopic planner, what kind of tree serach. Should be dpw or belief.", default = 'dpw')
parser.add_argument("-n", "--nonmyopic", action="store_true", help="Run planner in nonmyopic mode if flag set.", default = False)

# Parse command line options
parse = parser.parse_args()

# Read command line options
SEED = parse.seed
REWARD_FUNCTION = parse.reward
NONMYOPIC = parse.nonmyopic
TREE_TYPE = parse.tree

DIM = 3
DURATION = 150
MISSION_DURATION = 150
LENGTHSCALE = 1.0#(1.5, 1.5, 100.)
VARIANCE = 100.
NOISE = 0.1
KERNEL = 'swell'
KERNEL_PARAMS = {'lengthscale':(1.5, 1.5), 'variance':(100., 100.)}

# Parameters for plotting based on the seed world information
MIN_COLOR = -25.
MAX_COLOR = 25.

# Set up paths for logging the data from the simulation run
if not os.path.exists('./figures/' + str(REWARD_FUNCTION)):
    os.makedirs('./figures/' + str(REWARD_FUNCTION))
logging.basicConfig(filename='./figures/'+ REWARD_FUNCTION + '/robot.log', level=logging.INFO)
logger = logging.getLogger('robot')

# Create a random enviroment sampled from a GP with an RBF kernel and specified hyperparameters,
# mean function 0. The environment will be constrained by a set of uniformly distributed
# sample points of size NUM_PTS x NUM_PTS
ranges = (0., 10., 0., 10.)

world = envlib.Phenomenon(ranges=ranges,
                          NUM_PTS=20,
                          variance=VARIANCE,
                          lengthscale=LENGTHSCALE,
                          kparams=KERNEL_PARAMS,
                          dim=DIM,
                          kernel=KERNEL,
                          seed=SEED,
                          metric_world=obslib.World(ranges),
                          time_duration=DURATION,
                          MIN_COLOR=MIN_COLOR,
                          MAX_COLOR=MAX_COLOR)

# Create the evaluation class used to quantify the simulation metrics
evaluation = evalib.Evaluation(world=world, reward_function=REWARD_FUNCTION)

# Populate a world with obstacles
# ow = obslib.BlockWorld(ranges, 1, dim_blocks= (2,2), centers=[(7,7)])
# ow = obslib.ChannelWorld(ranges, (2.5,7), 3, 0.2)
ow = obslib.World(ranges)

# Create the path generator which allows for feasible actions sets based on robot mechanics
paths = pathlib.ActionSet(num_actions=15,
                          length=1.5,
                          turning_radius=0.05,
                          radius_angle=3*np.pi/4,
                          num_samples=3,
                          safe_threshold=50.,
                          unknown_threshold=-2.,
                          allow_reverse=False,
                          allow_stay=False)

# Create the point robot
robot = roblib.Robot(sample_world=world.sample_value, #function handle for collecting observations
                     start_loc=(5.0, 5.0, 0.0), #where robot is instantiated
                     start_time=0,
                     extent=ranges, #extent of the explorable environment
                     dimension=DIM,
                     kernel_file=None,
                     kernel_dataset=None,
                     prior_dataset=None, #(data, observations),
                     kernel=KERNEL,
                     kparams=KERNEL_PARAMS,
                     init_lengthscale=LENGTHSCALE,
                     init_variance=VARIANCE,
                     noise=NOISE,
                     path_generator=paths,
                     evaluation=evaluation,
                     f_rew=REWARD_FUNCTION,
                     create_animation=True, #logs images to the file folder
                     nonmyopic=NONMYOPIC, #select if you want to use MCTS
                     MIN_COLOR=MIN_COLOR,
                     MAX_COLOR=MAX_COLOR,
                     obstacle_world=ow,
                     running_simulation=True,
                     tree_type=TREE_TYPE,
                     rollout_length=7,
                     computation_budget=MISSION_DURATION)#200)

robot.planner(T=MISSION_DURATION)
robot.visualize_trajectory(screen=False) #creates a summary trajectory image
robot.visualize_reward(t=MISSION_DURATION)
robot.plot_information()
