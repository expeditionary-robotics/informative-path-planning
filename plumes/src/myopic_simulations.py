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
import gpmodel_library as gplib 

# Allow selection of seed world to be consistent, and to run through reward functions
# Initialize command line options
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", action="store", type=int, help="Random seed for environment generation.", default = 100)
parser.add_argument("-r", "--reward", action="store", help="Reward function. Should be mes, mean, gumbel, naive, or naive_value, ei, or info_gain.", default = 'mean')
parser.add_argument("-t", "--tree", action="store", help="If using nonmyopic planner, what kind of tree serach. Should be dpw or belief.", default = 'belief')
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
LENGTHSCALE = (1.5, 1.5, 100.)                                                         
VARIANCE = 100.
NOISE = 0.1
KERNEL = 'rbf'
KERNEL_PARAMS = {'lengthscale':(1.5, 50.), 'variance':(100., 10.), 'period':50.}

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
                          # model='figures/environment_model.pickle')

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

# Create a prior dataset
# belief = gplib.OnlineGPModel(ranges=ranges,
#                              lengthscale=(10., 10., 100.),
#                              variance=100.,
#                              noise=NOISE,
#                              dim=DIM,
#                              kernel='learned',
#                              kparams={'lengthscale':(1.5, (10.,10.,100.)), 'variance':(100.,100.), 'period':(80., 80., 80.)})
# x1observe = np.linspace(ranges[0], ranges[1], 20)
# x2observe = np.linspace(ranges[2], ranges[3], 20)
# x1observe, x2observe = np.meshgrid(x1observe, x2observe, sparse = False, indexing = 'xy')  
# if DIM == 2:
#    data = np.vstack([x1observe.ravel(), x2observe.ravel()]).T
# elif DIM == 3:
#    data = np.vstack([x1observe.ravel(), x2observe.ravel(), 0*np.ones(len(x1observe.ravel()))]).T
#    observations = world.sample_value(data, time=0)
#    for t in [i+1 for i in range(80)]:
#       temp_data = np.vstack([x1observe.ravel(), x2observe.ravel(), t*np.ones(len(x1observe.ravel()))]).T
#       temp_observations = world.sample_value(temp_data, time=t)
#       data = np.append(data, temp_data, axis=0)
#       observations = np.append(observations, temp_observations, axis=0)
#    belief.train_kernel(xvals=data, zvals=observations)

# Create the point robot
robot = roblib.Robot(sample_world=world.sample_value, #function handle for collecting observations
                     phenom_dim=DIM,
                     start_loc=(5.0, 5.0, 0.0), #where robot is instantiated
                     start_time=0,
                     extent=ranges, #extent of the explorable environment
                     dimension=DIM,
                     kernel_file=None,
                     kernel_dataset=None,
                     prior_dataset=None, #(data, observations),
                     # kernel='rbf',#'polar',#'seperable',#KERNEL,
                     kernel=KERNEL,
                     # kparams={'lengthscale':(1.5, 100.), 'variance':(5., 100.), 'period':150.},#KERNEL_PARAMS,
                     kparams={'lengthscale':(1.5, 100.), 'variance':(100., np.pi), 'period':78.},#KERNEL_PARAMS,
                     init_lengthscale=LENGTHSCALE,#1.5, #LENGTHSCALE,
                     init_variance=VARIANCE,#500, #VARIANCE,
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
