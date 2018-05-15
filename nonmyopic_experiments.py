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

seed = int(sys.argv[1])
reward_function = sys.argv[2]

if not os.path.exists('./figures/' + str(reward_function)): 
    os.makedirs('./figures/' + str(reward_function))
logging.basicConfig(filename = './figures/'+ reward_function + '/robot.log', level = logging.INFO)
logger = logging.getLogger('robot')
from ipp_library import *

# Create a random enviroment sampled from a GP with an RBF kernel and specified hyperparameters, mean function 0 
# The enviorment will be constrained by a set of uniformly distributed  sample points of size NUM_PTS x NUM_PTS
''' Options include mean, info_gain, hotspot_info, and mes'''
ranges = (0., 10., 0., 10.)

world = Environment(ranges = ranges, # x1min, x1max, x2min, x2max constraints
                    NUM_PTS = 20, 
                    variance = 100.0, 
                    lengthscale = 1.0, 
                    visualize = True,
                    seed = seed)

evaluation = Evaluation(world = world, reward_function = reward_function)

# Gather some prior observations to train the kernel (optional)
#x1observe = np.linspace(ranges[0]+0.5, ranges[1]-0,5, 15)
#x2observe = np.linspace(ranges[2]+0.5, ranges[3]-0.5, 15)
x1observe = np.linspace(0., 10., 10)
x2observe = np.linspace(0., 10., 10)
x1observe, x2observe = np.meshgrid(x1observe, x2observe, sparse = False, indexing = 'xy')  
data = np.vstack([x1observe.ravel(), x2observe.ravel()]).T

# TODO: make the edges less appealing!

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
              path_generator = 'dubins',
              frontier_size = 20, 
              horizon_length = 1.5, 
              turning_radius = 0.05,
              sample_step = 0.5,
              evaluation = evaluation, 
              f_rew = reward_function, 
              create_animation = True,
              computation_budget = 10.0,
              rollout_length = 5) 

robot.planner(T = 175)
#robot.visualize_world_model(screen = True)
#robot.visualize_trajectory(screen = True)
robot.plot_information()

