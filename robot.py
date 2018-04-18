import model
reload(model)
from model import *
from belief import *
from path_generator import *
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
from IPython.display import display
from scipy.stats import multivariate_normal
import numpy as np
import math
import os
import GPy as GPy

# Simulating the enviormental phenonmena as a scaled mixture of Gaussians            
def rv(xvals, yvals):
    # Create mixture of Gaussian models
    C1 = [[10, 0], [0, 10]]
    C2 = [[24., 3], [0.8, 2.1]]
    C3 = [[3, 0], [0, 3]]
    m1 = [3, 8]
    m2 = [-5, -5]
    m3 = [5, -7]
    
    pos = np.empty(xvals.shape + (2,))
    pos[:, :, 0] = xvals
    pos[:, :, 1] = yvals

    val = 100. * ((1./3.) * 10.* multivariate_normal.pdf(pos, mean = m1, cov = C1) + \
            5. * (1./3.) * multivariate_normal.pdf(pos, mean = m2, cov = C2) + \
            5. * (1./3.) * multivariate_normal.pdf(pos, mean = m3, cov = C3))
    #return np.reshape(val, (val.shape[0] * val.shape[0], 1))
    return val

# Sensors have access to a noisy version of the true environmental distirbution          
def rv_sample(xvals, yvals):
    data = rv(xvals, yvals)
    return rv(xvals, yvals) + np.random.randn(xvals.shape[0], xvals.shape[1]) * 0.35

class Robot:
    '''The Robot class, which includes the vechiles current model of the world, path set represetnation, and
        infromative path planning algorithm'''  
    def __init__(self, start_loc):
        self.start_loc = start_loc # Initial location of the robot
        self.delta = 0.30 # Sampling rate of the robot
        self.num_paths = 4 # Number of paths in the path set
    
    # Generate data from a Gaussian mixture model            
    def initializeGP(self, ranges, training_points):
        # Sample inputs and outputs 2D data
        np.random.seed(0)
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
        self.GP = model.GPModel(data, np.reshape(ztrain, (data.shape[0], 1)), learn_kernel=False, lengthscale = 10.0, variance = 0.5)
    def initalizePath(generator='general', frontier_size, horizon_length, turning_radius, sample_size):
        options = {'general':Path_Generator(frontier_size, horizon_length, turning_radius, sample_size),
                   'dubins':Dubins_Path_Generator(frontier_size, horizon_length, turning_radius, sample_size),
                   'dubins_equal': Dubins_EqualPath_Generator(frontier_size, horizon_length, turning_radius, sample_size)}
        
        self.pg = options[generator] 