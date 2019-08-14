# !/usr/bin/python

'''
This library is for accessing the Environmental Model in the IP PLUMES framework developed. It is within this model that the observable, true world, can be accessed.

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import numpy as np
import copy
import GPy as GPy
import pickle
import logging
logger = logging.getLogger('robot')
from gpmodel_library import GPModel
import generate_metric_environment as gme

class Phenomenon:
    '''The Phenomenon class, which simulates the true, natural phenomenon as
    a retangular Gaussian world.
    ''' 
    def __init__(self, ranges, NUM_PTS, variance, lengthscale, kparams, noise=0.0001, 
            seed=0, dim=2, model=None, metric_world=gme.World(), time_duration=1, kernel='rbf',
            MIN_COLOR=-25., MAX_COLOR=25.):
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
        seed (int): an integer seed for the random draws. If set to \'None\', 
            no seed is used 
        dim (int) dimensions of the phenomenon, (2D or 3D)
        model (GP model) if there is a distribution we would like to use directly
        metric_world (simulated gme World) allows for obstacles to be detected
        time_duration (int) duration of a transience
        '''

        # Save the parmeters of GP model
        self.variance = variance
        self.lengthscale = lengthscale
        self.dim = dim
        self.noise = noise
        self.obstacle_world = metric_world
        self.time_duration = time_duration
        self.kernel=kernel
        logger.info('Environment seed: {}'.format(seed))

        self.MIN_COLOR = MIN_COLOR
        self.MAX_COLOR = MAX_COLOR
        
        # Expect ranges to be a 4-tuple consisting of x1min, x1max, x2min, and x2max
        self.x1min = float(ranges[0])
        self.x1max = float(ranges[1])
        self.x2min = float(ranges[2])
        self.x2max = float(ranges[3])

        if model is not None:
            self.GP = model
            # Plot the surface mesh and scatter plot representation of the samples points
            plot_world(ranges,
                       self.obstacle_world,
                       copy.deepcopy(self.GP),
                       './figures/world_model_countour.png',
                       time=None,
                       MIN_COLOR=self.MIN_COLOR,
                       MAX_COLOR=self.MAX_COLOR)
        else:
            # Generate a set of discrete grid points, uniformly spread across the environment
            x1vals = np.linspace(self.x1min, self.x1max, NUM_PTS)
            x2vals = np.linspace(self.x2min, self.x2max, NUM_PTS)
            x1, x2 = np.meshgrid(x1vals, x2vals, sparse=False, indexing='xy') 
          
            # A dictionary to hold the GP model for each time stamp
            self.models = {}

            for T in xrange(self.time_duration):
                print "Generating environment for time ", T
                logger.warning("Generating environment for time %d", T)

                # Initialize maxima arbitrarily to violate boundary constraints
                maxima = [self.x1min, self.x2min]
                in_violation = True

                # Continue to generate random environments until the global maxima 
                # lives within the boundary constraints
                while in_violation:

                    # Initialize points at time T
                    if self.dim == 2:
                        data = np.vstack([x1.ravel(), x2.ravel()]).T 
                    elif self.dim == 3:
                        data = np.vstack([x1.ravel(), x2.ravel(), T*np.ones(len(x1.ravel()))]).T 
                    
                    # Intialize a GP model of the environment
                    self.GP = GPModel(ranges=ranges,
                                      lengthscale=lengthscale,
                                      variance=variance,
                                      noise=noise,
                                      dim=dim,
                                      kparams=kparams,
                                      kernel=kernel) 

                    print "Current environment in violation of boundary constraint. Regenerating!"
                    logger.warning("Current environment in violation of boundary constraint. Regenerating!")
                    if T == 0:        
                        # Take an initial sample in the GP prior, conditioned on no other data
                        xsamples = np.reshape(np.array(data[0, :]), (1, dim)) # dimension: 1 x dim
                        mean, var = self.GP.predict_value(xsamples, include_noise=False)   
                        
                        # Set the random seed for numpy, and increment for drawing data
                        np.random.seed(seed)
                        seed += 1

                        # Generate data to populate the GP with
                        zsamples = np.random.normal(loc=mean, scale=np.sqrt(var))
                        zsamples = np.reshape(zsamples, (1, 1)) # dimension: 1 x 1
                        self.GP.add_data(xsamples, zsamples)                            
                    
                        # Draw observations from the posterior (with new random seed), then update the GP with those samples
                        np.random.seed(seed)
                        observations = self.GP.posterior_samples(data[1:, :], full_cov=True, size=1)
                        self.GP.add_data(data[1:, :], observations)                            
                    else:
                        np.random.seed(seed)
                        observations = self.models[T-1].posterior_samples(data, full_cov=True, size=1)
                        self.GP.add_data(data, observations)
                         
                
                    # Extract the maxima and increment random seed
                    maxima = self.GP.xvals[np.argmax(self.GP.zvals), :]
                    seed += 1
                    in_violation = not self.obstacle_world.contains_point(maxima)

                # Satisfactory world has been found. Save the model and plot.
                print maxima, np.max(self.GP.zvals)
                self.models[T] = copy.deepcopy(self.GP)
                plot_world(ranges,
                           self.obstacle_world,
                           copy.deepcopy(self.GP),
                           './figures/world_model_countour.'+ str(T) + '.png', 
                           time=T,
                           MIN_COLOR=self.MIN_COLOR,
                           MAX_COLOR=self.MAX_COLOR)
                plt.close('all')
            
            # Dump the GP models, for later evaluation
            with open('./figures/environment_model.pickle', 'wb') as handle:
                print 'Dumping Generated Worlds to File'
                pickle.dump(self.models, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
            print "Environment initialized with bounds X1: (", self.x1min, ",", self.x1max, ")  X2:(", self.x2min, ",", self.x2max, ")"
            logger.info("Environment initialized with bounds X1: ({}, {})  X2: ({}, {})".format(self.x1min, self.x1max, self.x2min, self.x2max)) 

    def sample_value(self, xvals, time=0):
        ''' The public interface to the Environment class. Returns a noisy sample of the true value of environment at a set of point. 
        Input:
            xvals (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2 
            time (int): the true world model time
        Returns:
            mean (float array): an nparray of floats representing predictive mean, with dimension NUM_PTS x 1 
        '''
        assert(xvals.shape[0] >= 1)            
        assert(xvals.shape[1] == self.dim)        

        if self.dim == 2:
            mean, var = self.GP.predict_value(xvals, include_noise=False)
        elif self.dim == 3:
            T = time #get the time of the model phenomenon
            mean, var = self.models[T].predict_value(xvals, include_noise=False)
        else:
            raise ValueError('Model dimension must be 2 or 3!');

        # return the world value with additive noise (to simulate sensor)
        return mean + np.random.normal(loc=0, scale=np.sqrt(self.noise))

def plot_world(ranges, metric_world, GP, filename, time, MIN_COLOR=-25., MAX_COLOR=25.):
    ''' Helper function to draw contour plots of the phenomenon '''
    x1vals = np.linspace(ranges[0], ranges[1], 40)
    x2vals = np.linspace(ranges[2], ranges[3], 40)
    x1, x2 = np.meshgrid(x1vals, x2vals, sparse=False, indexing='xy') # dimension: NUM_PTS x NUM_PTS       
    
    dim = GP.dim
    if dim == 2:
        data = np.vstack([x1.ravel(), x2.ravel()]).T 
    elif dim == 3:
        data = np.vstack([x1.ravel(), x2.ravel(), time*np.ones(len(x1.ravel()))]).T
    observations, var = GP.predict_value(data, include_noise=False)
    # observations = dynamic_func(x1.ravel(), x2.ravel(), time*np.ones(len(x1.ravel()))) #+ observations
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(ranges[0:2])
    ax.set_ylim(ranges[2:])        
    ax.set_title('Countour Plot of the True World Model')     
    plot = ax.contourf(x1, x2, observations.reshape(x1.shape), 25, cmap='viridis', zorder=8, vmin=MIN_COLOR, vmax=MAX_COLOR, levels=np.linspace(MIN_COLOR, MAX_COLOR, 15))
    scatter = ax.scatter(GP.xvals[:, 0], GP.xvals[:, 1], c=GP.zvals.ravel(), s=4.0, cmap='viridis', zorder=9, vmin=MIN_COLOR, vmax=MAX_COLOR)
    maxind = np.argmax(GP.zvals)
    ax.scatter(GP.xvals[maxind, 0], GP.xvals[maxind,1], color='k', marker='*', s=1000, zorder=10)
    fig.colorbar(plot, ax=ax)

    if len(metric_world.obstacles) != 0:
        for o in metric_world.obstacles:
            plt.gca().add_patch(PolygonPatch(o.geom, alpha=0.9, color='r', zorder=11))

    fig.savefig(filename)
    plt.close()

def dynamic_func(x1, x2, t):
    ang1 = 1.5*np.sin(t/0.1)
    ang2 = 1.5*np.cos(t/0.1)

    temp1 = np.exp(-(np.power(((x1-5-ang1)/0.7),2)))
    temp2 = np.exp(-(np.power(((x2-5-ang2)/0.7),2)))

    f = np.multiply(temp1,temp2)
    f = np.array(list(f))[:,None]

    return f

if __name__ == '__main__':
    world = gme.World([0., 10., 0., 10.])
    # world.add_blocks(3, (1.5, 1.5), ((3, 3), (5, 5), (7, 7)))
    rbf_variance = 100
    rbf_lengthscale = (1.5, 1.5, 100.)

    ode_variance = (10, 10)
    ode_lengthscale = (1.5, 1.0)

    swell_variance = (100., 100.)
    swell_lengthscale = (1.5, 1.5)

    phenom = Phenomenon(ranges=[0., 10., 0., 10.],
                        NUM_PTS=20,
                        variance=rbf_variance,
                        lengthscale=rbf_lengthscale,
                        noise=0.5,
                        seed=200,
                        dim=3,
                        model=None,
                        metric_world=world,
                        time_duration=5,
                        kparams={'lengthscale':swell_lengthscale, 'variance':swell_variance},
                        kernel='transport',
                        MIN_COLOR=0,
                        MAX_COLOR=3)
