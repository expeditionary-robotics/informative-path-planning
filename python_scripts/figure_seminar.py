# Necessary imports
#matplotlib inline

from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib import cm
from sklearn import mixture
from IPython.display import display
from scipy.stats import multivariate_normal
import numpy as np
import math
import os
import GPy as GPy
import dubins
import time
from itertools import chain

from ipp_library import *

# from robot_library import *
# from mcts_library import *
# from gpmodel_library import *
# from evaluation_library import *
# from aq_library import *
# from obstacles import *
# from paths_library import * 
# from envmodel_library import *

if __name__ == "__main__":
    
    # Create a random enviroment sampled from a GP with an RBF kernel and specified hyperparameters, mean function 0 
    # The enviorment will be constrained by a set of uniformly distributed  sample points of size NUM_PTS x NUM_PTS

    ''' Options include mean, info_gain, and hotspot_info'''
    reward_function = 'hotspot_info'

    world = Environment(ranges = (-10., 10., -10., 10.), # x1min, x1max, x2min, x2max constraints
                        NUM_PTS = 20, 
                        variance = 100.0, 
                        lengthscale = 3.0, 
                        visualize = True,
                        seed = 1)

    evaluation = Evaluation(world = world, 
                            reward_function = reward_function)

    # Gather some prior observations to train the kernel (optional)
    ranges = (-10, 10, -10, 10)
    x1observe = np.linspace(ranges[0], ranges[1], 5)
    x2observe = np.linspace(ranges[2], ranges[3], 5)
    x1observe, x2observe = np.meshgrid(x1observe, x2observe, sparse = False, indexing = 'xy')  
    data = np.vstack([x1observe.ravel(), x2observe.ravel()]).T
    observations = world.sample_value(data)

    # Create the point robot
    robot = Nonmyopic_Robot(sample_world = world.sample_value, 
                start_loc = (0.0, 0.0, 0.0), 
                extent = (-10., 10., -10., 10.),
                kernel_file = None,
                kernel_dataset = None,
                prior_dataset =  None, 
                init_lengthscale = 3.0, 
                init_variance = 100.0, 
                noise = 0.05,
                path_generator = 'equal_dubins',
                frontier_size = 20, 
                horizon_length = 5.0, 
                turning_radius = 0.5, 
                sample_step = 2.0,
                evaluation = evaluation, 
                f_rew = reward_function,
                computation_budget = 2.0,
                rollout_length = 3)

    robot.planner(T = 150)
    robot.visualize_world_model()
    robot.visualize_trajectory()
    robot.plot_information()
