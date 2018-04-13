from model import *
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


class Environment:
    '''The Environment class, which represents a retangular Gaussian world, initially parameterized with:
    
    ranges (tuple of floats) a tuple representing the max/min extenet of the 2D rectangular domain i.e. (-10, 10, -50, 50)
    NUM_PTS (int) the number of points in each dimension to sample for initialization, resulting in a sample grid of size NUM_PTS x NUM_PTS
    variance (float) the variance parameter of the squared exponential kernel
    lengthscale (float) the lengthscale parameter of the squared exponential kernel
    noise (float) the sensor noise parameter of the squared exponential kernel
    visualize (boolean) a boolean flag to plot the surface of the resulting environment ''' 

    def __init__(self, ranges, NUM_PTS, variance = 0.5, lengthscale = 1.0, noise = 0.05, visualize = True, dim = 2):
        ''' Initialize a random Gaussian environment using the input kernel, assuming zero mean'''
        # Save the parmeters of GP model
        self.variance = variance
        self.lengthscale = lengthscale
        self.dim = dim
        
        # Expect ranges to be a 4-tuple consisting of x1min, x1max, x2min, and x2max
        self.x1min = float(ranges[0])
        self.x1max = float(ranges[1])
        self.x2min = float(ranges[2])
        self.x2max = float(ranges[3]) 
        
        # Generate a set of discrete grid points, uniformly spread across the environment
        x1 = np.linspace(self.x1min, self.x1max, NUM_PTS)
        x2 = np.linspace(self.x2min, self.x2max, NUM_PTS)
        x1vals, x2vals = np.meshgrid(x1, x2, sparse = False, indexing = 'xy') # dimension: NUM_PTS x NUM_PTS
        data = np.vstack([x1vals.ravel(), x2vals.ravel()]).T # dimension: NUM_PTS*NUM_PTS x 2

        # Take an initial sample in the GP prior, conditioned on no other data
        xsamples = np.reshape(np.array(data[0, :]), (1, dim)) # dimension: 1 x 2
        zsamples = np.reshape(np.random.normal(loc = 0, scale = variance), (1,1)) # dimension: 1 x 1 
                
        # Initialze a GP model with a first sampled point    
        self.GP = GPModel(xsamples, zsamples, learn_kernel = False, lengthscale = lengthscale, variance = variance)   
    
        # Iterate through the rest of the grid sequentially and sample a z values, condidtioned on previous samples
        for index, point in enumerate(data[1:, :]):
            # Get a new sample point
            xs = np.reshape(np.array(point), (1, dim))
    
            # Compute the predicted mean and variance
            mean, var = self.GP.m.predict(xs, full_cov = False, include_likelihood = True)
            
            # Sample a new observation, given the mean and variance
            zs = np.random.normal(loc = mean, scale = var)
            
            # Add new sample point to the GP model
            zsamples = np.vstack([zsamples, np.reshape(zs, (1, 1))])
            xsamples = np.vstack([xsamples, np.reshape(xs, (1, dim))])
            self.GP.m.set_XY(X = xsamples, Y = zsamples)
      
        # Plot the surface mesh and scatter plot representation of the samples points
        if visualize == True:
            fig = plt.figure()
            ax = fig.add_subplot(211, projection = '3d')
            surf = ax.plot_surface(x1vals, x2vals, zsamples.reshape(x1vals.shape), cmap = cm.coolwarm, linewidth = 1)

            ax2 = fig.add_subplot(212, projection = '3d')
            scatter = ax2.scatter(data[:, 0], data[:, 1], zsamples, c = zsamples, cmap = cm.coolwarm)
            plt.show()           
        
        print "Environment initialized with bounds X1: (", self.x1min, ",", self.x1max, ")  X2:(", self.x2min, ",", self.x2max, ")" 
 
    ''' Return a noisy sample of the true value of environment at a set of point. 
    point_set should be an N x dim array of floats '''        
    def sample_value(self, point_set):
        assert(point_set.shape[0] >= 1)            
        assert(point_set.shape[1] == self.dim)        

        mean, var = self.GP.m.predict(point_set, full_cov = False, include_likelihood = True)
        return mean
    