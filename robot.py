from model import *
from belief import *
from path_generator import *
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
from sklearn import mixture
from IPython.display import display
from scipy.stats import multivariate_normal
import numpy as np
import math
import os
import GPy as GPy


class Robot:
    '''The Robot class, which includes the vechiles current model of the world, path set represetnation, and
        infromative path planning algorithm'''  
    def __init__(self, start_loc):
        self.start_loc = start_loc # Initial location of the robot
        self.delta = 0.30 # Sampling rate of the robot
        self.num_paths = 4 # Number of paths in the path set
    
    # Generate data from a Gaussian mixture model            
    def initializeGP(self, ranges, training_points, visualize = True):
        # Sample inputs and outputs 2D data
        if visualize:
            x = np.linspace(ranges[0], ranges[1], 100)
            y = np.linspace(ranges[2], ranges[3], 100)
            xvals, yvals = np.meshgrid(x, y, sparse = False, indexing = 'xy')
            zvals = rv(xvals, yvals)

        xtrain = np.linspace(ranges[0], ranges[1], training_points)
        ytrain = np.linspace(ranges[2], ranges[3], training_points)
        xtrain, ytrain= np.meshgrid(xtrain, ytrain, sparse = False, indexing = 'xy')
        data = np.vstack([xtrain.ravel(), ytrain.ravel()]).T
        ztrain = rv_sample(xtrain, ytrain)

        # Create and train parmeters of GP model
        self.GP = GPModel(data, np.reshape(ztrain, (1, data.shape[0])), lengthscale = 10.0, variance = 0.5)        