# Imports
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
import sys
import logging

import aq_library as aqlib
import mcts_library as mctslib
import gpmodel_library as gplib 
import evaluation_library as evalib 
import paths_library as pathlib 
import envmodel_library as envlib 
import robot_library as roblib

seed = 0#int(sys.argv[1])
reward_function = 'exp_improve'#sys.argv[2]

MIN_COLOR = -25.
MAX_COLOR = 25.

if not os.path.exists('./figures/' + str(reward_function)): 
    os.makedirs('./figures/' + str(reward_function))
logging.basicConfig(filename = './figures/'+ reward_function + '/robot.log', level = logging.INFO)
logger = logging.getLogger('robot')
from ipp_library import *

# Create a random enviroment sampled from a GP with an RBF kernel and specified hyperparameters, mean function 0 
# The enviorment will be constrained by a set of uniformly distributed  sample points of size NUM_PTS x NUM_PTS
''' Options include mean, info_gain, hotspot_info, and mes'''
ranges = (0., 10., 0., 10.)

world = envlib.Environment(ranges = ranges, # x1min, x1max, x2min, x2max constraints
                    NUM_PTS = 20, 
                    variance = 100.0, 
                    lengthscale = 1.0, 
                    visualize = True,
                    seed = seed,
                    MIN_COLOR=MIN_COLOR, 
                    MAX_COLOR=MAX_COLOR)

evaluation = evalib.Evaluation(world = world, reward_function = reward_function)

# Create the point robot
robot = roblib.Robot(sample_world = world.sample_value, 
              start_loc = (5.0, 5.0, 0.0), 
              extent = ranges,
              kernel_file = None,
              kernel_dataset = None,
              prior_dataset =  None, 
              #prior_dataset =  (data, observations), 
              init_lengthscale = 1.0, 
              init_variance = 100.0, 
              noise = 0.0001,
              path_generator = 'dubins',
              goal_only = False, #select only if using fully reachable step and you want the step to only be in the direction of the goal
              frontier_size = 20, 
              horizon_length = 1.5, 
              turning_radius = 0.05,
              sample_step = 0.5,
              evaluation = evaluation, 
              f_rew = reward_function, 
              create_animation = True,
              learn_params=False,
              nonmyopic=True, #select if you want to use MCTS
              discretization=(20,20),
              use_cost=False, #select if you want to use a cost heuristic
              MIN_COLOR=MIN_COLOR,
              MAX_COLOR=MAX_COLOR) 

robot.planner(T = 50)
#robot.visualize_world_model(screen = True)
robot.visualize_trajectory(screen = False)
robot.plot_information()

