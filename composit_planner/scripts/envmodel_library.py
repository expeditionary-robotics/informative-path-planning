# !/usr/bin/python

'''
This library is for accessing the Environmental Model in the IP PLUMES framework developed. It is within this model that the observable, true world, can be accessed.

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''

from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib import cm
from IPython.display import display
import numpy as np
import math
import os
import GPy as GPy
import logging
logger = logging.getLogger('robot')
from gpmodel_library import GPModel
from gpmodel_library import OnlineGPModel
import obstacles as obslib



class Environment:
    '''The Environment class, which represents a retangular Gaussian world.
    ''' 
    def __init__(self, ranges, NUM_PTS, variance, lengthscale, noise = 0.0001, 
            visualize = True, seed = None, dim = 2, model = None, MIN_COLOR=-25.0, MAX_COLOR=25.0, 
            obstacle_world = obslib.FreeWorld()):
        ''' Initialize a random Gaussian environment using the input kernel, 
            assuming zero mean function.
        Input:
        ranges (tuple of floats): a tuple representing the max/min of 2D 
            rectangular domain i.e. (-10, 10, -50, 50)
        NUM_PTS (int): the number of points in each dimension to sample for 
            initialization, resulting in a sample grid of size NUM_PTS x NUM_PTS
        variance (float): the variance parameter of the kernel
        lengthscale (float): the lengthscale parameter of the kernel
        noise (float): the sensor noise parameter of the kernel
        visualize (boolean): flag to plot the surface of the environment 
        seed (int): an integer seed for the random draws. If set to \'None\', 
            no seed is used 
        MIN_COLOR (float) used for plottig the range for the visualization
        MAX_COLOR (float) used for plotting the range for the visualization
        '''

        # Save the parmeters of GP model
        self.variance = variance
        self.lengthscale = lengthscale
        self.dim = dim
        self.noise = noise
        self.obstacle_world = obstacle_world
        logger.info('Environment seed: {}'.format(seed))
        
        # Expect ranges to be a 4-tuple consisting of x1min, x1max, x2min, and x2max
        self.x1min = float(ranges[0])
        self.x1max = float(ranges[1])
        self.x2min = float(ranges[2])
        self.x2max = float(ranges[3])

        if model is not None:
            self.GP = model
            # Plot the surface mesh and scatter plot representation of the samples points
            if visualize == True:   
                # Generate a set of observations from robot model with which to make contour plots
                x1vals = np.linspace(ranges[0], ranges[1], 40)
                x2vals = np.linspace(ranges[2], ranges[3], 40)
                x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy') # dimension: NUM_PTS x NUM_PTS       
                data = np.vstack([x1.ravel(), x2.ravel()]).T
                observations, var = self.GP.predict_value(data, include_noise = False)        
                
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                ax2.set_xlim(ranges[0:2])
                ax2.set_ylim(ranges[2:])        
                ax2.set_title('Countour Plot of the True World Model')     
                plot = ax2.contourf(x1, x2, observations.reshape(x1.shape), cmap = 'viridis', vmin = MIN_COLOR, vmax = MAX_COLOR, levels=np.linspace(MIN_COLOR, MAX_COLOR, 15))

                scatter = ax2.scatter(self.GP.xvals[:, 0], self.GP.xvals[:, 1], c = self.GP.zvals.ravel(), s = 4.0, cmap = 'viridis')
                maxind = np.argmax(self.GP.zvals)
                ax2.scatter(self.GP.xvals[maxind, 0], self.GP.xvals[maxind,1], color = 'k', marker = '*', s = 500)
                fig2.colorbar(plot, ax=ax2)

                fig2.savefig('./figures/world_model_countour.png')
                #plt.show()           
                plt.close()
        else:
            # Generate a set of discrete grid points, uniformly spread across the environment
            x1 = np.linspace(self.x1min, self.x1max, NUM_PTS)
            x2 = np.linspace(self.x2min, self.x2max, NUM_PTS)
            # dimension: NUM_PTS x NUM_PTS
            x1vals, x2vals = np.meshgrid(x1, x2, sparse = False, indexing = 'xy') 
            # dimension: NUM_PTS*NUM_PTS x 2
            data = np.vstack([x1vals.ravel(), x2vals.ravel()]).T 

            bb = ((ranges[1] - ranges[0])*0.05, (ranges[3] - ranges[2]) * 0.05)
            ranges = (ranges[0] + bb[0], ranges[1] - bb[0], ranges[2] + bb[1], ranges[3] - bb[1])
            # Initialize maxima arbitrarily to violate boundary constraints
            maxima = [self.x1min, self.x2min]

            # Continue to generate random environments until the global maximia 
            # lives within the boundary constraints
            while maxima[0] < ranges[0] or maxima[0] > ranges[1] or \
                  maxima[1] < ranges[2] or maxima[1] > ranges[3] or \
                  self.obstacle_world.in_obstacle(maxima, buff = 0.0):
                print "Current environment in violation of boundary constraint. Regenerating!"
                logger.warning("Current environment in violation of boundary constraint. Regenerating!")

                # Intialize a GP model of the environment
                self.GP = OnlineGPModel(ranges = ranges, lengthscale = lengthscale, variance = variance)         
                data = np.vstack([x1vals.ravel(), x2vals.ravel()]).T 

                # Take an initial sample in the GP prior, conditioned on no other data
                xsamples = np.reshape(np.array(data[0, :]), (1, dim)) # dimension: 1 x 2        
                mean, var = self.GP.predict_value(xsamples, include_noise = False)   
                if seed is not None:
                    np.random.seed(seed)
                    seed += 1
                zsamples = np.random.normal(loc = 0, scale = np.sqrt(var))
                zsamples = np.reshape(zsamples, (1,1)) # dimension: 1 x 1 
                                    
                # Add initial sample data point to the GP model
                self.GP.add_data(xsamples, zsamples)                            
                np.random.seed(seed)
                observations = self.GP.posterior_samples(data[1:, :], full_cov = True, size=1)
                self.GP.add_data(data[1:, :], observations)                            
                        
                '''
                # Iterate through the rest of the grid sequentially and sample a z values, 
                # conditioned on previous samples
                for index, point in enumerate(data[1:, :]):
                    # Get a new sample point
                    xs = np.reshape(np.array(point), (1, dim))
            
                    # Compute the predicted mean and variance
                    mean, var = self.GP.predict_value(xs)
                    
                    # Sample a new observation, given the mean and variance
                    if seed is not None:
                        np.random.seed(seed)
                        seed += 1            
                    zs = np.random.normal(loc = mean, scale = np.sqrt(var))
                    
                    # Add new sample point to the GP model
                    zsamples = np.vstack([zsamples, np.reshape(zs, (1, 1))])
                    xsamples = np.vstack([xsamples, np.reshape(xs, (1, dim))])
                    self.GP.add_data(np.reshape(xs, (1, dim)), np.reshape(zs, (1, 1)))
                '''
            
                maxima = self.GP.xvals[np.argmax(self.GP.zvals), :]

                # Plot the surface mesh and scatter plot representation of the samples points
                if visualize == True:   
                    # the 3D surface
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection = '3d')
                    ax.set_title('Surface of the Simulated Environment')
                    surf = ax.plot_surface(x1vals, x2vals, self.GP.zvals.reshape(x1vals.shape), cmap = cm.coolwarm, linewidth = 1)
                    if not os.path.exists('./figures'):
                        os.makedirs('./figures')
                    fig.savefig('./figures/world_model_surface.png')
                    
                    # the contour map            
                    fig2 = plt.figure(figsize=(8, 6))
                    ax2 = fig2.add_subplot(111)
                    ax2.set_title('Countour Plot of the Simulated Environment')     
                    plot = ax2.contourf(x1vals, x2vals, self.GP.zvals.reshape(x1vals.shape), cmap = 'viridis', vmin = MIN_COLOR, vmax = MAX_COLOR, levels=np.linspace(MIN_COLOR, MAX_COLOR, 15))
                    scatter = ax2.scatter(data[:, 0], data[:, 1], c = self.GP.zvals.ravel(), s = 4.0, cmap = 'viridis')
                    maxind = np.argmax(self.GP.zvals)
                    ax2.scatter(self.GP.xvals[maxind, 0], self.GP.xvals[maxind,1], color = 'k', marker = '*', s = 500)
                    fig2.colorbar(plot, ax=ax2)

                    # If available, plot the obstacles in the world
                    if len(self.obstacle_world.get_obstacles()) != 0:
                        for o in self.obstacle_world.get_obstacles():
                            x,y = o.exterior.xy
                            ax2.plot(x,y,'r',linewidth=3)

                    fig2.savefig('./figures/world_model_countour.png')
                    #plt.show()           
                    plt.close()
        
            print "Environment initialized with bounds X1: (", self.x1min, ",", self.x1max, ")  X2:(", self.x2min, ",", self.x2max, ")"
            logger.info("Environment initialized with bounds X1: ({}, {})  X2: ({}, {})".format(self.x1min, self.x1max, self.x2min, self.x2max)) 

    def sample_value(self, xvals):
        ''' The public interface to the Environment class. Returns a noisy sample of the true value of environment at a set of point. 
        Input:
            xvals (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2 
        
        Returns:
            mean (float array): an nparray of floats representing predictive mean, with dimension NUM_PTS x 1 
        '''
        assert(xvals.shape[0] >= 1)            
        assert(xvals.shape[1] == self.dim)        

        mean, var = self.GP.predict_value(xvals, include_noise = False)
        return mean + np.random.normal(loc = 0, scale = np.sqrt(self.noise))
