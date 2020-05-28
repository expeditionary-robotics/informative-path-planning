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
import pdb
import logging
logger = logging.getLogger('robot')


''' 
This library file aggregates a number of classes that are useful for performing the informative path planning (IPP) problem. Detailed documentation is provided for each of the classes inline.

Maintainers: Genevieve Flaspohler and Victoria Preston
License: MIT
'''
# MIN_COLOR = 3.0
# MAX_COLOR = 7.5
MIN_COLOR = -25.
MAX_COLOR = 25.

class GPModel:
    '''The GPModel class, which is a wrapper on top of GPy.'''     
    
    def __init__(self, ranges, lengthscale, variance, noise = 0.0001, dimension = 2, kernel = 'rbf'):
        '''Initialize a GP regression model with given kernel parameters. 
        Inputs:
            ranges (list of floats) the bounds of the world
            lengthscale (float) the lengthscale parameter of kernel
            variance (float) the variance parameter of kernel
            noise (float) the sensor noise parameter of kernel
            dimension (float) the dimension of the environment; only 2D supported
            kernel (string) the type of kernel; only 'rbf' supported now
        '''
        
        # Model parameterization (noise, lengthscale, variance)
        self.noise = noise
        self.lengthscale = lengthscale
        self.variance = variance
        
        self.ranges = ranges
        
        # The Gaussian dataset; start with null set
        self.xvals = None
        self.zvals = None

        
        # The dimension of the evironment
        if dimension == 2:
            self.dim = dimension
        else:
            raise ValueError('Environment must have dimension 2 \'rbf\'')

        if kernel == 'rbf':
            self.kern = GPy.kern.RBF(input_dim = self.dim, lengthscale = lengthscale, variance = variance) 
        else:
            raise ValueError('Kernel type must by \'rbf\'')
            
        # Intitally, before any data is created, 
        self.model = None
        self.temp_model = None
         
    def predict_value(self, xvals, TEMP = False):
        ''' Public method returns the mean and variance predictions at a set of input locations.
        Inputs:
            xvals (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2
        
        Returns: 
            mean (float array): an nparray of floats representing predictive mean, with dimension NUM_PTS x 1         
            var (float array): an nparray of floats representing predictive variance, with dimension NUM_PTS x 1 
        '''        

        assert(xvals.shape[0] >= 1)            
        assert(xvals.shape[1] == self.dim)    
        
        n_points, input_dim = xvals.shape
       
        if TEMP: 
            # With no observations, predict 0 mean everywhere and prior variance
            if self.temp_model == None:
                return np.zeros((n_points, 1)), np.ones((n_points, 1)) * self.variance
            
            # Else, return the predicted values
            mean, var = self.temp_model.predict(xvals, full_cov = False, include_likelihood = True)
            return mean, var        

        # With no observations, predict 0 mean everywhere and prior variance
        if self.model == None:
            return np.zeros((n_points, 1)), np.ones((n_points, 1)) * self.variance
        
        # Else, return the predicted values
        mean, var = self.model.predict(xvals, full_cov = False, include_likelihood = True)
        return mean, var        

    def add_data_and_temp_model(self, xvals, zvals):
        ''' Public method that adds data to a temporay GP model and returns that model
        Inputs:
            xvals (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2
            zvals (float array): an nparray of floats representing sensor observations, with dimension NUM_PTS x 1 
        ''' 
        
        if self.xvals is None:
            xvals = xvals
        else:
            xvals = np.vstack([self.xvals, xvals])
            
        if self.zvals is None:
            zvals = zvals
        else:
            zvals = np.vstack([self.zvals, zvals])

        # Create a temporary model
        self.temp_model = GPy.models.GPRegression(np.array(xvals), np.array(zvals), self.kern)


    def add_data(self, xvals, zvals):
        ''' Public method that adds data to an the GP model.
        Inputs:
            xvals (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2
            zvals (float array): an nparray of floats representing sensor observations, with dimension NUM_PTS x 1 
        ''' 
        
        if self.xvals is None:
            self.xvals = xvals
        else:
            self.xvals = np.vstack([self.xvals, xvals])
            
        if self.zvals is None:
            self.zvals = zvals
        else:
            self.zvals = np.vstack([self.zvals, zvals])

        # If the model hasn't been created yet (can't be created until we have data), create GPy model
        if self.model == None:
            self.model = GPy.models.GPRegression(np.array(self.xvals), np.array(self.zvals), self.kern)
        # Else add to the exisiting model
        else:
            self.model.set_XY(X = np.array(self.xvals), Y = np.array(self.zvals))

    def load_kernel(self, kernel_file = 'kernel_model.npy'):
        ''' Public method that loads kernel parameters from file.
        Inputs:
            kernel_file (string): a filename string with the location of the kernel parameters 
        '''    
        
        # Read pre-trained kernel parameters from file, if avaliable and no training data is provided
        if os.path.isfile(kernel_file):
            print "Loading kernel parameters from file"
            logger.info("Loading kernel parameters from file")
            self.kern[:] = np.load(kernel_file)
        else:
            raise ValueError("Failed to load kernel. Kernel parameter file not found.")
        return

    def train_kernel(self, xvals = None, zvals = None, kernel_file = 'kernel_model.npy'):
        ''' Public method that optmizes kernel parameters based on input data and saves to files.
        Inputs:
            xvals (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2
            zvals (float array): an nparray of floats representing sensor observations, with dimension NUM_PTS x 1        
            kernel_file (string): a filename string with the location to save the kernel parameters 
        Outputs:
            nothing is returned, but a kernel file is created.
        '''      
        
        # Read pre-trained kernel parameters from file, if available and no 
        # training data is provided
        # if xvals is not None and zvals is not None:
        #     if self.xvals is None:
        #         self.xvals = xvals
        #     else:
        #         self.xvals = np.vstack([self.xvals, xvals])
                
        #     if self.zvals is None:
        #         self.zvals = zvals
        #     else:
        #         self.zvals = np.vstack([self.zvals, zvals])

        if self.xvals is not None and self.zvals is not None:
            xvals = self.xvals[::5]
            zvals = self.zvals[::5]

            print "Optimizing kernel parameters given data"
            logger.info("Optimizing kernel parameters given data")
            # Initilaize a GP model (used only for optmizing kernel hyperparamters)
            self.m = GPy.models.GPRegression(np.array(xvals), np.array(zvals), self.kern)
            self.m.initialize_parameter()

            # Constrain the hyperparameters during optmization
            self.m.constrain_positive('')
            self.m['Gaussian_noise.variance'].constrain_fixed(self.noise)

            # Train the kernel hyperparameters
            self.m.optimize_restarts(num_restarts = 2, messages = True)

            # Save the hyperparemters to file
            np.save(kernel_file, self.kern[:])
            self.lengthscale = self.kern.lengthscale
            self.variance = self.kern.variance

        else:
            raise ValueError("Failed to train kernel. No training data provided.")


class Environment:
    '''The Environment class, which represents a retangular Gaussian world.
    ''' 
    def __init__(self, ranges, NUM_PTS, variance, lengthscale, noise = 0.0001, 
            visualize = True, seed = None, dim = 2, model = None):
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
        '''

        # Save the parmeters of GP model
        self.variance = variance
        self.lengthscale = lengthscale
        self.dim = dim
        self.noise = noise
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
                observations, var = self.GP.predict_value(data)        
                
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
                  maxima[1] < ranges[2] or maxima[1] > ranges[3]:
                print "Current environment in violation of boundary constraint. Regenerating!"
                logger.warning("Current environment in violation of boundary constraint. Regenerating!")

                # Intialize a GP model of the environment
                self.GP = GPModel(ranges = ranges, lengthscale = lengthscale, variance = variance)         

                # Take an initial sample in the GP prior, conditioned on no other data
                # This is done to 
                xsamples = np.reshape(np.array(data[0, :]), (1, dim)) # dimension: 1 x 2        
                mean, var = self.GP.predict_value(xsamples)   
                if seed is not None:
                    np.random.seed(seed)
                    seed += 1
                zsamples = np.random.normal(loc = 0, scale = np.sqrt(var))
                zsamples = np.reshape(zsamples, (1,1)) # dimension: 1 x 1 
                                    
                # Add initial sample data point to the GP model
                self.GP.add_data(xsamples, zsamples)                            
                        
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
            
                maxima = self.GP.xvals[np.argmax(self.GP.zvals), :]

                # Plot the surface mesh and scatter plot representation of the samples points
                if visualize == True:   
                    # the 3D surface
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection = '3d')
                    ax.set_title('Surface of the Simulated Environment')
                    surf = ax.plot_surface(x1vals, x2vals, zsamples.reshape(x1vals.shape), cmap = cm.coolwarm, linewidth = 1)
                    if not os.path.exists('./figures'):
                        os.makedirs('./figures')
                    fig.savefig('./figures/world_model_surface.png')
                    
                    # the contour map            
                    fig2 = plt.figure(figsize=(8, 6))
                    ax2 = fig2.add_subplot(111)
                    ax2.set_title('Countour Plot of the Simulated Environment')     
                    plot = ax2.contourf(x1vals, x2vals, zsamples.reshape(x1vals.shape), cmap = 'viridis', vmin = MIN_COLOR, vmax = MAX_COLOR, levels=np.linspace(MIN_COLOR, MAX_COLOR, 15))
                    scatter = ax2.scatter(data[:, 0], data[:, 1], c = zsamples.ravel(), s = 4.0, cmap = 'viridis')
                    maxind = np.argmax(zsamples)
                    ax2.scatter(xsamples[maxind, 0], xsamples[maxind,1], color = 'k', marker = '*', s = 500)
                    fig2.colorbar(plot, ax=ax2)

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

        mean, var = self.GP.predict_value(xvals)
        return mean + np.random.normal(loc = 0, scale = np.sqrt(self.noise))


'''The Path_Generator class which creates naive point-to-point straightline paths'''
class Path_Generator:    
    def __init__(self, frontier_size, horizon_length, turning_radius, sample_step, extent):
        ''' Initialize a path generator
        Input:
            frontier_size (int) the number of points on the frontier we should consider for navigation
            horizon_length (float) distance between the vehicle and the horizon to consider
            turning_radius (float) the feasible turning radius for the vehicle
            sample_step (float) the unit length along the path from which to draw a sample
            extent (list of floats) the world boundaries
        '''

        # the parameters for the dubin trajectory
        self.fs = frontier_size
        self.hl = horizon_length
        self.tr = turning_radius
        self.ss = sample_step
        self.extent = extent

        # Global variables
        self.goals = [] #The frontier coordinates
        self.samples = {} #The sample points which form the paths
        self.cp = (0,0,0) #The current pose of the vehicle

    def generate_frontier_points(self):
        '''From the frontier_size and horizon_length, generate the frontier points to goal'''
        angle = np.linspace(-2.35,2.35,self.fs) #fix the possibilities to 75% of the unit circle, ignoring points directly behind the vehicle
        goals = []
        for a in angle:
            # x = self.hl*np.cos(self.cp[2]+a)+self.cp[0]
            # y = self.hl*np.sin(self.cp[2]+a)+self.cp[1]


            x = self.hl*np.cos(self.cp[2]+a)+self.cp[0]
            if x >= self.extent[1]-3*self.tr:
                x = self.extent[1]-3*self.tr
                y = (x-self.cp[0])*np.sin(self.cp[2]+a)+self.cp[1]
            elif x <= self.extent[0]+3*self.tr:
                x = self.extent[0]+3*self.tr
                y = (x-self.cp[0])*np.sin(self.cp[2]+a)+self.cp[1]
            else:
                y = self.hl*np.sin(self.cp[2]+a)+self.cp[1]
                if y >= self.extent[3]-3*self.tr:
                    y = self.extent[3]-3*self.tr
                    x = (y-self.cp[1])*-np.cos(self.cp[2]+a)+self.cp[0]
                elif y <= self.extent[2]+3*self.tr:
                    y = self.extent[2]+3*self.tr
                    x = (y-self.cp[1])*-np.cos(self.cp[2]+a)+self.cp[0]
            p = self.cp[2]+a
            if np.linalg.norm([self.cp[0]-x, self.cp[1]-y]) <= self.tr:
                pass
            elif x > self.extent[1]-3*self.tr or x < self.extent[0]+3*self.tr:
                pass
            elif y > self.extent[3]-3*self.tr or y < self.extent[2]+3*self.tr:
                pass
            else:
                goals.append((x,y,p))

        self.goals = goals
        return self.goals

    def make_sample_paths(self):
        '''Connect the current_pose to the goal places'''
        cp = np.array(self.cp)
        coords = {}
        for i,goal in enumerate(self.goals):
            g = np.array(goal)
            distance = np.sqrt((cp[0]-g[0])**2 + (cp[1]-g[1])**2)
            samples = int(round(distance/self.ss))

            # Don't include the start location but do include the end point
            for j in range(0,samples):
                x = cp[0]+((j+1)*self.ss)*np.cos(g[2])
                y = cp[1]+((j+1)*self.ss)*np.sin(g[2])
                a = g[2]
                try: 
                    coords[i].append((x,y,a))
                except:
                    coords[i] = []
                    coords[i].append((x,y,a))
        self.samples = coords
        return self.samples

    def get_path_set(self, current_pose):
        '''Primary interface for getting list of path sample points for evaluation
        Input:
            current_pose (tuple of x, y, z, a which are floats) current location of the robot in world coordinates
        Output:
            paths (dictionary of frontier keys and sample points)
        '''
        self.cp = current_pose
        self.generate_frontier_points()
        paths = self.make_sample_paths()
        return paths

    def get_frontier_points(self):
        ''' Method to access the goal points'''
        return self.goals

    def get_sample_points(self):
        return self.samples


class Dubins_Path_Generator(Path_Generator):
    '''
    The Dubins_Path_Generator class, which inherits from the Path_Generator class. Replaces the make_sample_paths
    method with paths generated using the dubins library
    '''
    
    def buffered_paths(self):
        coords = {}
        for i,goal in enumerate(self.goals):            
            path = dubins.shortest_path(self.cp, goal, self.tr)
            configurations, _ = path.sample_many(self.ss)
            configurations.append(goal)

            temp = []
            for config in configurations:
                if config[0] > self.extent[0] and config[0] < self.extent[1] and config[1] > self.extent[2] and config[1] < self.extent[3]:
                    temp.append(config)
                else:
                    temp = []
                    break

            if len(temp) < 2:
                pass
            else:
                coords[i] = temp

        if len(coords) == 0:
            pdb.set_trace()
        return coords    
        
    def make_sample_paths(self):
        '''Connect the current_pose to the goal places'''
        coords = self.buffered_paths()
        
        if len(coords) == 0:
            print 'no viable path'
            
        self.samples = coords
        return coords


class Dubins_EqualPath_Generator(Path_Generator):
    '''
    The Dubins_EqualPath_Generator class which inherits from Path_Generator. Modifies Dubin Curve paths so that all
    options have an equal number of sampling points
    '''
        
    def make_sample_paths(self):
        '''Connect the current_pose to the goal places'''
        coords = {}
        for i,goal in enumerate(self.goals):
            g = (goal[0],goal[1],self.cp[2])
            path = dubins.shortest_path(self.cp, goal, self.tr)
            configurations, _ = path.sample_many(self.ss)
            coords[i] = [config for config in configurations if config[0] > self.extent[0] and config[0] < self.extent[1] and config[1] > self.extent[2] and config[1] < self.extent[3]]
        
        # find the "shortest" path in sample space
        current_min = 1000
        for key,path in coords.items():
            if len(path) < current_min and len(path) > 1:
                current_min = len(path)
        
        # limit all paths to the shortest path in sample space
        # NOTE! for edge cases nar borders, this limits the paths significantly
        for key,path in coords.items():
            if len(path) > current_min:
                path = path[0:current_min]
                coords[key]=path
        

class MCTS:
    '''Class that establishes a MCTS for nonmyopic planning'''

    def __init__(self, computation_budget, belief, initial_pose, rollout_length, frontier_size, path_generator, aquisition_function, f_rew, time, aq_param = None):
        '''Initialize with constraints for the planning, including whether there is a budget or planning horizon
        Inputs:
            computation_budget (float) number of seconds to run the tree building procedure
            belief (GP model) current belief of the vehicle
            initial_pose (tuple of floats) location of the vehicle in world coordinates
            rollout_length (int) number of actions to rollout after selecting a child (tree depth)
            frontier_size (int) number of options for each action in the tree (tree breadth)
            path_generator (string) how action sets should be developed
            aquisition_function (function) the criteria to make decisions
            f_rew (string) the name of the function used to make decisions
            time (float) time in the global world used for aquisition weighting
        '''

        # Parameterization for the search
        self.comp_budget = computation_budget
        self.GP = belief
        self.cp = initial_pose
        self.rl = rollout_length
        self.fs = frontier_size
        self.path_generator = path_generator
        self.aquisition_function = aquisition_function
        self.f_rew = f_rew
        self.t = time

        # The tree
        self.tree = None
        
        # Elements which are relevant for some acquisition functions
        self.params = None
        self.max_val = None
        self.max_locs = None
        self.current_max = aq_param

    def choose_trajectory(self, t):
        ''' Main function loop which makes the tree and selects the best child
        Output:
            path to take, cost of that path
        '''
        # initialize tree
        self.tree = self.initialize_tree() 
        i = 0 #iteration count

        # randonly sample the world for entropy search function
        if self.f_rew == 'mes' or self.f_rew == 'maxs-mes':
            self.max_val, self.max_locs, self.target  = sample_max_vals(self.GP, t = t)
            
        time_start = time.time()            
            
        # while we still have time to compute, generate the tree
        while time.time() - time_start < self.comp_budget:
            i += 1
            current_node = self.tree_policy()
            sequence = self.rollout_policy(current_node)
            reward = self.get_reward(sequence)
            self.update_tree(reward, sequence)

        # get the best action to take with most promising futures
        best_sequence, best_val, all_vals = self.get_best_child()
        print "Number of rollouts:", i, "\t Size of tree:", len(self.tree)
        logger.info("Number of rollouts: {} \t Size of tree: {}".format(i, len(self.tree)))

        paths = self.path_generator.get_path_set(self.cp)                
        return self.tree[best_sequence][0], best_val, paths, all_vals, self.max_locs, self.max_val

    def initialize_tree(self):
        '''Creates a tree instance, which is a dictionary, that keeps track of the nodes in the world
        Output:
            tree (dictionary) an initial tree
        '''
        tree = {}
        # root of the tree is current location of the vehicle
        tree['root'] = (self.cp, 0) #(pose, number of queries)
        actions = self.path_generator.get_path_set(self.cp)
        for action, samples in actions.items():
            cost = np.sqrt((self.cp[0]-samples[-1][0])**2 + (self.cp[1]-samples[-1][1])**2)
            tree['child '+ str(action)] = (samples, cost, 0, 0) #(samples, cost, reward, number of times queried)
        return tree

    def tree_policy(self):
        '''Implements the UCB policy to select the child to expand and forward simulate. From Arora paper, the following is defined:
            avg_r - average reward of all rollouts that have passed through node n
            c_p - some arbitrary constant, they use 0.1
            N - number of times parent has been evaluated
            n - number of times that node has been evaluated
            the formula: avg_r + c_p * np.sqrt(2*np.log(N)/n)
        '''
        leaf_eval = {}
        # TODO: check initialization, when everything is zero. appears to be throwing error
        actions = self.path_generator.get_path_set(self.cp)
        for i, val in actions.items():
            try:
                node = 'child '+ str(i)
                leaf_eval[node] = self.tree[node][2] + 0.1*np.sqrt(2*(np.log(self.tree['root'][1]))/self.tree[node][3])
            except:
                pass
        return max(leaf_eval, key=leaf_eval.get)

    def rollout_policy(self, node):
        '''Select random actions to expand the child node
        Input:
            node (the name of the child node that is to be expanded)
        Output:
            sequence (list of names of nodes that make the sequence in the tree)
        '''

        sequence = [node] #include the child node
        #TODO use the cost metric to signal action termination, for now using horizon
        for i in xrange(self.rl):
            actions = self.path_generator.get_path_set(self.tree[node][0][-1]) #plan from the last point in the sample
            if len(actions) == 0:
                print 'No actions were viably generated'
            try:
                
                try:
                    a = np.random.randint(0,len(actions)-1) #choose a random path
                except:
                    if len(actions) != 0:
                        a = 0

                keys = actions.keys()
                # print keys
                if len(keys) <= 1:
                    #print 'few paths available!'
                    pass
                #TODO add cost metrics
                self.tree[node + ' child ' + str(keys[a])] = (actions[keys[a]], 0, 0, 0) #add random path to the tree
                node = node + ' child ' + str(keys[a])
                sequence.append(node)
            except:
                print 'rolling back'
                sequence.remove(node)
                try:
                    node = sequence[-1]
                except:
                    print "Empty sequence", sequence, node
                    logger.warning('Bad sequence')
        return sequence

    def get_reward(self, sequence):
        '''Evaluate the sequence to get the reward, defined by the percentage of entropy reduction.
        Input:
            sequence (list of strings) names of the nodes in the tree
        Outut:
            reward value from the aquisition function of choice
        '''
        sim_world = self.GP
        samples = []
        obs = []
        for seq in sequence:
            samples.append(self.tree[seq][0])
        obs = list(chain.from_iterable(samples))

        if self.f_rew == 'mes' or self.f_rew == 'maxs-mes':
            return self.aquisition_function(time = self.t, xvals = obs, robot_model = sim_world, param = (self.max_val, self.max_locs, self.target))
        elif self.f_rew == 'exp_improve':
            return self.aquisition_function(time=self.t, xvals = obs, robot_model = sim_world, param = [self.current_max])
        else:
            return self.aquisition_function(time=self.t, xvals = obs, robot_model = sim_world)

    
    def update_tree(self, reward, sequence):
        '''Propogate the reward for the sequence
        Input:
            reward (float) the reward or utility value of the sequence
            sequence (list of strings) the names of nodes that form the sequence
        '''
        #TODO update costs as well
        self.tree['root'] = (self.tree['root'][0], self.tree['root'][1]+1)
        for seq in sequence:
            samples, cost, rew, queries = self.tree[seq]
            queries += 1
            n = queries
            rew = ((n-1)*rew+reward)/n
            self.tree[seq] = (samples, cost, rew, queries)

    def get_best_child(self):
        '''Query the tree for the best child in the actions
        Output:
            (string, float) node name of the best child, the cost of that child
        '''
        best = -float('inf')
        best_child = None
        value = {}
        for i in xrange(self.fs):
            try:
                r = self.tree['child '+ str(i)][2]
                value[i] = r
                #if r > best and len(self.tree['child '+ str(i)][0]) > 1: 
                if r > best: 
                    best = r
                    best_child = 'child '+ str(i)
            except:
                pass
        return best_child, best, value


class Robot(object):
    ''' The Robot class, which includes the vehicles current model of the world and IPP algorithms.'''

    def __init__(self, sample_world, start_loc = (0.0, 0.0, 0.0), extent = (-10., 10., -10., 10.), 
            kernel_file = None, kernel_dataset = None, prior_dataset = None, init_lengthscale = 10.0, 
            init_variance = 100.0, noise = 0.05, path_generator = 'default', frontier_size = 6, 
            horizon_length = 5, turning_radius = 1, sample_step = 0.5, evaluation = None, 
            f_rew = 'mean', create_animation = False, learn_params = False):
        ''' Initialize the robot class with a GP model, initial location, path sets, and prior dataset
        Inputs:
            sample_world (method) a function handle that takes a set of locations as input and returns a set of observations
            start_loc (tuple of floats) the location of the robot initially in 2-D space e.g. (0.0, 0.0, 0.0)
            extent (tuple of floats): a tuple representing the max/min of 2D rectangular domain i.e. (-10, 10, -50, 50)
            kernel_file (string) a filename specifying the location of the stored kernel values
            kernel_dataset (tuple of nparrays) a tuple (xvals, zvals), where xvals is a Npoint x 2 nparray of type float and zvals is a Npoint x 1 nparray of type float 
            prior_dataset (tuple of nparrays) a tuple (xvals, zvals), where xvals is a Npoint x 2 nparray of type float and zvals is a Npoint x 1 nparray of type float
            init_lengthscale (float) lengthscale param of kernel
            init_variance (float) variance param of kernel
            noise (float) the sensor noise parameter of kernel 
            path_generator (string): one of default, dubins, or equal_dubins. Robot path parameterization. 
            frontier_size (int): the number of paths in the generated path set
            horizon_length (float): the length of the paths generated by the robot 
            turning_radius (float): the turning radius (in units of distance) of the robot
            sample_set (float): the step size (in units of distance) between sequential samples on a trajectory
            evaluation (Evaluation object): an evaluation object for performance metric compuation
            f_rew (string): the reward function. One of {hotspot_info, mean, info_gain, exp_info, mes}
                    create_animation (boolean): save the generate world model and trajectory to file at each timestep 
        '''

        # Parameterization for the robot
        self.ranges = extent
        self.create_animation = create_animation
        self.eval = evaluation
        self.loc = start_loc     
        self.sample_world = sample_world
        self.f_rew = f_rew
        self.fs = frontier_size
        self.maxes = []
        self.current_max = -1000
        self.current_max_loc = [0,0]
        self.max_locs = None
        self.max_val = None
        self.learn_params = learn_params
        self.target = None
        
        if f_rew == 'hotspot_info':
            self.aquisition_function = hotspot_info_UCB
        elif f_rew == 'mean':
            self.aquisition_function = mean_UCB  
        elif f_rew == 'info_gain':
            self.aquisition_function = info_gain
        elif f_rew == 'mes':
            self.aquisition_function = mves
        elif f_rew == 'maxs-mes':
            self.aquisition_function = mves_maximal_set
        elif f_rew == 'exp_improve':
            self.aquisition_function = exp_improvement
        else:
            raise ValueError('Only \'hotspot_info\' and \'mean\' and \'info_gain\' and \'mes\' and \'exp_improve\' reward fucntions supported.')

        # Initialize the robot's GP model with the initial kernel parameters
        self.GP = GPModel(ranges = extent, lengthscale = init_lengthscale, variance = init_variance)
                
        # If both a kernel training dataset and a prior dataset are provided, train the kernel using both
        if  kernel_dataset is not None and prior_dataset is not None:
            data = np.vstack([prior_dataset[0], kernel_dataset[0]])
            observations = np.vstack([prior_dataset[1], kernel_dataset[1]])
            self.GP.train_kernel(data, observations, kernel_file) 
        # Train the kernel using the provided kernel dataset
        elif kernel_dataset is not None:
            self.GP.train_kernel(kernel_dataset[0], kernel_dataset[1], kernel_file)
        # If a kernel file is provided, load the kernel parameters
        elif kernel_file is not None:
            self.GP.load_kernel()
        # No kernel information was provided, so the kernel will be initialized with provided values
        else:
            pass
        
        # Incorporate the prior dataset into the model
        if prior_dataset is not None:
            self.GP.add_data(prior_dataset[0], prior_dataset[1]) 
        
        # The path generation class for the robot
        path_options = {'default':Path_Generator(frontier_size, horizon_length, turning_radius, sample_step, self.ranges),
                        'dubins': Dubins_Path_Generator(frontier_size, horizon_length, turning_radius, sample_step, self.ranges),
                        'equal_dubins': Dubins_EqualPath_Generator(frontier_size, horizon_length, turning_radius, sample_step, self.ranges)}
                        # 'continuous_traj' : continuous_traj(frontier_size, horizon_length, )}
                        
        self.path_generator = path_options[path_generator]

    def choose_trajectory(self, t):
        ''' Select the best trajectory avaliable to the robot at the current pose, according to the aquisition function.
        Input: 
            t (int > 0): the current planning iteration (value of a point can change with algortihm progress)
        Output:
            either None or the (best path, best path value, all paths, all values, the max_locs for some functions)
        '''
        value = {}
        param = None    
        
        max_locs = max_vals = None      
        if self.f_rew == 'mes' or self.f_rew == 'maxs-mes':
            self.max_val, self.max_locs, self.target = sample_max_vals(self.GP, t = t)
            
        paths = self.path_generator.get_path_set(self.loc)

        for path, points in paths.items():
            if self.f_rew == 'mes' or self.f_rew == 'maxs-mes':
                param = (self.max_val, self.max_locs, self.target)
            elif self.f_rew == 'exp_improve':
                if len(self.maxes) == 0:
                    param = [self.current_max]
                else:
                    param = self.maxes
            value[path] = self.aquisition_function(time = t, xvals = points, robot_model = self.GP, param = param)            
        try:
            return paths[max(value, key = value.get)], value[max(value, key = value.get)], paths, value, self.max_locs
        except:
            return None
    
    def collect_observations(self, xobs):
        ''' Gather noisy samples of the environment and updates the robot's GP model.
        Input: 
            xobs (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2 '''
        zobs = self.sample_world(xobs)       
        self.GP.add_data(xobs, zobs)

        for z, x in zip (zobs, xobs):
            if z[0] > self.current_max:
                self.current_max = z[0]
                self.current_max_loc = [x[0],x[1]]

    def predict_max(self):
        # If no observations have been collected, return default value
        if self.GP.xvals is None:
            return np.array([0., 0.]).reshape(1,2), 0.

        ''' First option, return the max value observed so far '''
        #return self.GP.xvals[np.argmax(self.GP.zvals), :], np.max(self.GP.zvals)

        ''' Second option: generate a set of predictions from model and return max '''
        # Generate a set of observations from robot model with which to predict mean
        x1vals = np.linspace(self.ranges[0], self.ranges[1], 30)
        x2vals = np.linspace(self.ranges[2], self.ranges[3], 30)
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy') 
        data = np.vstack([x1.ravel(), x2.ravel()]).T
        observations, var = self.GP.predict_value(data)        

        return data[np.argmax(observations), :], np.max(observations)
        
    def planner(self, T):
        ''' Gather noisy samples of the environment and updates the robot's GP model  
        Input: 
            T (int > 0): the length of the planning horization (number of planning iterations)'''
        self.trajectory = []
        
        for t in xrange(T):
            # Select the best trajectory according to the robot's aquisition function
            print "[", t, "] Current Location:  ", self.loc
            logger.info("[{}] Current Location: {}".format(t, self.loc))
            best_path, best_val, all_paths, all_values, max_locs = self.choose_trajectory(t = t)
            
            # Given this choice, update the evaluation metrics 
            # TODO: fix this
            pred_loc, pred_val = self.predict_max()
            print "Current predicted max and value: \t", pred_loc, "\t", pred_val
            logger.info("Current predicted max and value: {} \t {}".format(pred_loc, pred_val))

            try:
                self.eval.update_metrics(len(self.trajectory), self.GP, all_paths, best_path, \
                value = best_val, max_loc = pred_loc, max_val = pred_val, params = [self.current_max, self.current_max_loc, self.max_val, self.max_locs]) 
            except:
                max_locs = [[-1, -1], [-1, -1]]
                max_val = [-1,-1]
                self.eval.update_metrics(len(self.trajectory), self.GP, all_paths, best_path, \
                        value = best_val, max_loc = pred_loc, max_val = pred_val, params = [self.current_max, self.current_max_loc, max_val, max_locs]) 

            
            if best_path == None:
                break
            data = np.array(best_path)
            x1 = data[:,0]
            x2 = data[:,1]
            xlocs = np.vstack([x1, x2]).T           
            
            if len(best_path) != 1:
                self.collect_observations(xlocs)
            if t < T/3 and self.learn_params == True:
                self.GP.train_kernel()
            self.trajectory.append(best_path)
            
            if self.create_animation:
                self.visualize_trajectory(screen = False, filename = t, best_path = best_path, 
                        maxes = max_locs, all_paths = all_paths, all_vals = all_values)            

            # if len(best_path) == 1:
            #     self.loc = (best_path[-1][0], best_path[-1][1], best_path[-1][2]-0.45)
            # else:
            self.loc = best_path[-1]
        np.savetxt('./figures/' + self.f_rew+ '/robot_model.csv', (self.GP.xvals[:, 0], self.GP.xvals[:, 1], self.GP.zvals[:, 0]))

    
    def visualize_trajectory(self, screen = True, filename = 'SUMMARY', best_path = None, 
        maxes = None, all_paths = None, all_vals = None):      
        ''' Visualize the set of paths chosen by the robot 
        Inputs:
            screen (boolean): determines whether the figure is plotted to the screen or saved to file
            filename (string): substring for the last part of the filename i.e. '0', '1', ...
            best_path (path object)
            maxes (list of locations)
            all_paths (list of path objects)
            all_vals (list of all path rewards) 
            T (string or int): string append to the figure filename
        '''
        
        # Generate a set of observations from robot model with which to make contour plots
        x1vals = np.linspace(self.ranges[0], self.ranges[1], 100)
        x2vals = np.linspace(self.ranges[2], self.ranges[3], 100)
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy') 
        data = np.vstack([x1.ravel(), x2.ravel()]).T
        observations, var = self.GP.predict_value(data)        
        
       
        # Plot the current robot model of the world
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(self.ranges[0:2])
        ax.set_ylim(self.ranges[2:])
        plot = ax.contourf(x1, x2, observations.reshape(x1.shape), cmap = 'viridis', vmin = MIN_COLOR, vmax = MAX_COLOR, levels=np.linspace(MIN_COLOR, MAX_COLOR, 15))
        if self.GP.xvals is not None:
            scatter = ax.scatter(self.GP.xvals[:, 0], self.GP.xvals[:, 1], c='k', s = 20.0, cmap = 'viridis')                
        color = iter(plt.cm.cool(np.linspace(0,1,len(self.trajectory))))
       
        # Plot the current trajectory
        for i, path in enumerate(self.trajectory):
            c = next(color)
            f = np.array(path)
            plt.plot(f[:,0], f[:,1], c=c, marker='*')

        # If available, plot the current set of options available to robot, colored
        # by their value (red: low, yellow: high)
        if all_paths is not None:
            all_vals = [x for x in all_vals.values()]   
            path_color = iter(plt.cm.autumn(np.linspace(0, max(all_vals),len(all_vals))/ max(all_vals)))        
            path_order = np.argsort(all_vals)
            
            for index in path_order:
                c = next(path_color)                
                points = all_paths[all_paths.keys()[index]]
                f = np.array(points)
                plt.plot(f[:,0], f[:,1], c = c, marker='.')
               
        # If available, plot the selected path in green
        if best_path is not None:
            f = np.array(best_path)
            plt.plot(f[:,0], f[:,1], c = 'g', marker='*')
           
        # If available, plot the current location of the maxes for mes
        if maxes is not None:
            for coord in maxes:
                plt.scatter(coord[0], coord[1], color = 'r', marker = '*', s = 500.0)
            # plt.scatter(maxes[:, 0], maxes[:, 1], color = 'r', marker = '*', s = 500.0)
           
        # Either plot to screen or save to file
        if screen:
            plt.show()           
        else:
            if not os.path.exists('./figures/' + str(self.f_rew)):
                os.makedirs('./figures/' + str(self.f_rew))
            fig.savefig('./figures/' + str(self.f_rew)+ '/trajectory-N.' + str(filename) + '.png')
            #plt.show()
            plt.close()
            
    def visualize_world_model(self, screen = True, filename = 'SUMMARY'):
        ''' Visaulize the robots current world model by sampling points uniformly in space and 
        plotting the predicted function value at those locations.
        Inputs:
            screen (boolean): determines whether the figure is plotted to the screen or saved to file 
            filename (String): name of the file to be made
            maxes (locations of largest points in the world)
        '''
        # Generate a set of observations from robot model with which to make contour plots
        x1vals = np.linspace(self.ranges[0], self.ranges[1], 100)
        x2vals = np.linspace(self.ranges[2], self.ranges[3], 100)
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy') # dimension: NUM_PTS x NUM_PTS       
        data = np.vstack([x1.ravel(), x2.ravel()]).T
        observations, var = self.GP.predict_value(data)        
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.set_xlim(self.ranges[0:2])
        ax2.set_ylim(self.ranges[2:])        
        ax2.set_title('Countour Plot of the Robot\'s World Model')     
        plot = ax2.contourf(x1, x2, observations.reshape(x1.shape), cmap = 'viridis', vmin = MIN_COLOR, vmax = MAX_COLOR, levels=np.linspace(MIN_COLOR, MAX_COLOR, 15))

        # Plot the samples taken by the robot
        if self.GP.xvals is not None:
            scatter = ax2.scatter(self.GP.xvals[:, 0], self.GP.xvals[:, 1], c=self.GP.zvals.ravel(), s = 10.0, cmap = 'viridis')        
        if screen:
            plt.show()           
        else:
            if not os.path.exists('./figures/' + str(self.f_rew)):
                os.makedirs('./figures/' + str(self.f_rew))
            fig.savefig('./figures/' + str(self.f_rew)+ '/world_model.' + str(filename) + '.png')
            plt.close()
    
    def plot_information(self):
        ''' Visualizes the accumulation of reward and aquisition functions ''' 
        self.eval.plot_metrics()


class Nonmyopic_Robot(Robot):
    '''This robot inherits from the Robot class, but uses a MCTS in order to perform global horizon planning'''
    
    def __init__(self, sample_world, start_loc = (0.0, 0.0, 0.0), extent = (-10., 10., -10., 10.), 
            kernel_file = None, kernel_dataset = None, prior_dataset = None, init_lengthscale = 10.0, 
            init_variance = 100.0, noise = 0.05, path_generator = 'default', frontier_size = 6, 
            horizon_length = 5, turning_radius = 1, sample_step = 0.5, evaluation = None, 
            f_rew = 'mean', create_animation = False, computation_budget = 60, rollout_length = 6):
        ''' Initialize the robot class with a GP model, initial location, path sets, and prior dataset'''
       
        self.__class__ = Nonmyopic_Robot
        super(Nonmyopic_Robot, self).__init__(sample_world, start_loc, extent, kernel_file, kernel_dataset, 
            prior_dataset, init_lengthscale, init_variance, noise, path_generator, frontier_size, 
            horizon_length, turning_radius, sample_step, evaluation, f_rew, create_animation)        
    
        #Robot.__init__(self, sample_world, start_loc, extent, kernel_file, kernel_dataset, 
        #    prior_dataset, init_lengthscale, init_variance, noise, path_generator, frontier_size, 
        #    horizon_length, turning_radius, sample_step, evaluation, f_rew, create_animation)        
        
        # Computation limits
        self.comp_budget = computation_budget
        self.roll_length = rollout_length

    def planner(self, T = 3):
        ''' Use a monte carlo tree search in order to perform long-horizon planning
        Input:
            T (int) time, or number of steps to take in the real world
        '''
        self.trajectory = []
                 
        for t in xrange(T):
            print "[", t, "] Current Location:  ", self.loc            
            logger.info("[{}] Current Location: {}".format(t, self.loc))

            if self.f_rew == "exp_improve":
                param = self.current_max
            else:
                param = None

            #computation_budget, belief, initial_pose, planning_limit, frontier_size, path_generator, aquisition_function, reward, time
            mcts = MCTS(self.comp_budget, self.GP, self.loc, self.roll_length, self.fs, \
                    self.path_generator, self.aquisition_function, self.f_rew, t, aq_param = param)
            best_path, best_val, all_paths, all_vals, max_locs, max_val = mcts.choose_trajectory(t = t)


            if self.create_animation:
                self.visualize_trajectory(screen = False, filename = str(t),  best_path = best_path,\
                        maxes = max_locs, all_paths = all_paths, all_vals = all_vals)

            # Update relevent metrics with selected path
            data = np.array(best_path)
            x1 = data[:,0]
            x2 = data[:,1]
            xlocs = np.vstack([x1, x2]).T
            all_paths = self.path_generator.get_path_set(self.loc)
            
            pred_loc, pred_val = self.predict_max()
            print "Current predicted max and value: \t", pred_loc, "\t", pred_val
            logger.info("Current predicted max and value: {} \t {}".format(pred_loc, pred_val))

            try:
                self.eval.update_metrics(len(self.trajectory), self.GP, all_paths, best_path, \
                        value = best_val, max_loc = pred_loc, max_val = pred_val, params = [self.current_max, self.current_max_loc, max_val, max_locs]) 
            except:
                max_locs = [[-1, -1], [-1, -1]]
                max_val = [-1,-1]
                self.eval.update_metrics(len(self.trajectory), self.GP, all_paths, best_path, \
                        value = best_val, max_loc = pred_loc, max_val = pred_val, params = [self.current_max, self.current_max_loc, max_val, max_locs]) 


            self.collect_observations(xlocs)
            if t < T/3 and self.learn_params == True:
                self.GP.train_kernel()
            self.trajectory.append(best_path)        

            self.loc = best_path[-1]
       
        np.savetxt('./figures/' + self.f_rew+ '/robot_model.csv', (self.GP.xvals[:, 0], self.GP.xvals[:, 1], self.GP.zvals[:, 0]))
        
class Evaluation:
    ''' The Evaluation class, which includes the ground truth world model and a selection of reward criteria.
    
    Inputs:
        world (Environment object): an environment object that represents the ground truth environment
        f_rew (string): the reward function. One of {hotspot_info, mean, info_gain, mes, exp_improve} 
    '''
    def __init__(self, world, reward_function = 'mean'):
        ''' Initialize the evaluation module and select reward function'''
        self.world = world
        self.max_val = np.max(world.GP.zvals)
        self.max_loc = world.GP.xvals[np.argmax(world.GP.zvals), :]
        self.reward_function = reward_function

        print "World max value", self.max_val, "at location", self.max_loc
        logger.info("World max value {} at location {}".format(self.max_val, self.max_loc))
        
        self.metrics = {'aquisition_function': {},
                        'mean_reward': {}, 
                        'info_gain_reward': {},                         
                        'hotspot_info_reward': {}, 
                        'MSE': {},                         
                        'hotspot_error': {},                         
                        'instant_regret': {},
                        'max_val_regret': {},
                        'regret_bound': {},
                        'simple_regret': {},
                        'sample_regret_loc': {},
                        'sample_regret_val': {},
                        'max_loc_error': {},
                        'max_val_error': {},
                        'current_highest_obs': {},
                        'current_highest_obs_loc_x': {},
                        'current_highest_obs_loc_y': {},
                        'star_obs_0': {},
                        'star_obs_1': {},
                        'star_obs_loc_x_0': {},
                        'star_obs_loc_x_1': {},
                        'star_obs_loc_y_0': {},
                        'star_obs_loc_y_1': {},
                        'robot_location_x': {},
                        'robot_location_y': {},
                        'robot_location_a': {},
                       }
        self.reward_function = reward_function
        
        if reward_function == 'hotspot_info':
            self.f_rew = self.hotspot_info_reward
            self.f_aqu = hotspot_info_UCB
        elif reward_function == 'mean':
            self.f_rew = self.mean_reward
            self.f_aqu = mean_UCB      
        elif reward_function == 'info_gain':
            self.f_rew = self.info_gain_reward
            self.f_aqu = info_gain   
        elif reward_function == 'mes':
            self.f_aqu = mves
            self.f_rew = self.mean_reward 
        elif reward_function == 'maxs-mes':
            self.f_aqu = mves_maximal_set
            self.f_rew = self.mean_reward 
        elif reward_function == 'exp_improve':
            self.f_aqu = exp_improvement
            self.f_rew = self.mean_reward
        else:
            raise ValueError('Only \'mean\' and \'hotspot_info\' and \'info_gain\' and \' mes\' and \'exp_improve\' reward functions currently supported.')    
    
    '''Reward Functions - should have the form (def reward(time, xvals, robot_model)), where:
        time (int): the current timestep of planning
        xvals (list of float tuples): representing a path i.e. [(3.0, 4.0), (5.6, 7.2), ... ])
        robot_model (GPModel)
    '''
    def mean_reward(self, time, xvals, robot_model):
        ''' Predcited mean (true) reward function'''
        data = np.array(xvals)
        x1 = data[:,0]
        x2 = data[:,1]
        queries = np.vstack([x1, x2]).T   
        
        mu, var = self.world.GP.predict_value(queries)
        return np.sum(mu)   

    def hotspot_info_reward(self, time, xvals, robot_model):
        ''' The reward information gathered plus the exploitation value gathered'''    
        LAMBDA = 1.0# TOOD: should depend on time
        data = np.array(xvals)
        x1 = data[:,0]
        x2 = data[:,1]
        queries = np.vstack([x1, x2]).T   
        
        mu, var = self.world.GP.predict_value(queries)    
        return self.info_gain_reward(time, xvals, robot_model) + LAMBDA * np.sum(mu)
    
    def info_gain_reward(self, time, xvals, robot_model):
        ''' The information reward gathered '''
        return info_gain(time, xvals, robot_model)
    
    def inst_regret(self, t, all_paths, selected_path, robot_model, param = None):
        ''' The instantaneous Kapoor regret of a selected path, according to the specified reward function
        Input:
            all_paths: the set of all avalaible paths to the robot at time t
            selected path: the path selected by the robot at time t 
            robot_model (GP Model)
        '''

        value_omni = {}        
        for path, points in all_paths.items():           
            if param is None:
                value_omni[path] =  self.f_rew(time = t, xvals = points, robot_model = robot_model)  
            else:
                value_omni[path] =  mves(time = t, xvals = points, robot_model = robot_model, param = (self.max_val).reshape(1,1))  

        value_max = value_omni[max(value_omni, key = value_omni.get)]
        if param is None:
            value_selected = self.f_rew(time = t, xvals = selected_path, robot_model = robot_model)
        else:
            value_selected =  mves(time = t, xvals = selected_path, robot_model = robot_model, param = (self.max_val).reshape(1,1))  
        return value_max - value_selected
    
    def simple_regret(self, xvals):
        ''' The simple regret of a selected trajecotry
        Input:
            max_loc (nparray 1 x 2)
        '''
        error = 0.0
        for point in xvals:
            error += np.linalg.norm(np.array(point[0:-1]) -  self.max_loc)
        error /= float(len(xvals))

        return error

    def sample_regret(self, robot_model):
        if robot_model.xvals is None:
            return 0., 0.

        global_max_val = np.reshape(np.array(self.max_val), (1,1))
        global_max_loc = np.reshape(np.array(self.max_loc), (1,2))
        avg_loc_dist = sp.spatial.distance.cdist(global_max_loc, robot_model.xvals)
        avg_val_dist = sp.spatial.distance.cdist(global_max_val, robot_model.zvals)
        return np.mean(avg_loc_dist), np.mean(avg_val_dist)
    
    def max_error(self, max_loc, max_val):
        ''' The error of the current best guess for the global maximizer
        Input:
            max_loc (nparray 1 x 2)
            max_val (float)
        '''
        return np.linalg.norm(max_loc - self.max_loc), np.linalg.norm(max_val - self.max_val)

    def hotspot_error(self, robot_model, NTEST = 100, NHS = 100):
        ''' Compute the hotspot error on a set of test points, randomly distributed throughout the environment'''
        x1 = np.random.random_sample((NTEST, 1)) * (self.world.x1max - self.world.x1min) + self.world.x1min
        x2 = np.random.random_sample((NTEST, 1)) * (self.world.x2max - self.world.x2min) + self.world.x2min
        data = np.hstack((x1, x2))
        
        pred_world, var_world = self.world.GP.predict_value(data)
        pred_robot, var_robot = robot_model.predict_value(data)      

        # Get the NHOTSPOT most "valuable" points
        #print pred_world
        order = np.argsort(pred_world, axis = None)
        pred_world = pred_world[order[0:NHS]]
        pred_robot = pred_robot[order[0:NHS]]

        #print pred_world
        #print pred_robot
        #print order
        
        return ((pred_world - pred_robot) ** 2).mean()
    
    def regret_bound(self, t, T):
        pass
        
    def MSE(self, robot_model, NTEST = 100):
        ''' Compute the MSE on a set of test points, randomly distributed throughout the environment'''
        x1 = np.random.random_sample((NTEST, 1)) * (self.world.x1max - self.world.x1min) + self.world.x1min
        x2 = np.random.random_sample((NTEST, 1)) * (self.world.x2max - self.world.x2min) + self.world.x2min
        data = np.hstack((x1, x2))
        
        pred_world, var_world = self.world.GP.predict_value(data)
        pred_robot, var_robot = robot_model.predict_value(data)      
        
        return ((pred_world - pred_robot) ** 2).mean()
    
    ''' Helper functions '''

    def update_metrics(self, t, robot_model, all_paths, selected_path, value = None, max_loc = None, max_val = None, params = None):
        ''' Function to update avaliable metrics'''    
        #self.metrics['hotspot_info_reward'][t] = self.hotspot_info_reward(t, selected_path, robot_model, max_val)
        #self.metrics['mean_reward'][t] = self.mean_reward(t, selected_path, robot_model)
        self.metrics['aquisition_function'][t] = value

        self.metrics['simple_regret'][t] = self.simple_regret(selected_path)
        self.metrics['sample_regret_loc'][t], self.metrics['sample_regret_val'][t] = self.sample_regret(robot_model)
        self.metrics['max_loc_error'][t], self.metrics['max_val_error'][t] = self.max_error(max_loc, max_val)
        
        self.metrics['instant_regret'][t] = self.inst_regret(t, all_paths, selected_path, robot_model)
        self.metrics['max_val_regret'][t] = self.inst_regret(t, all_paths, selected_path, robot_model, param = 'info_regret')

        self.metrics['star_obs_0'][t] = params[2][0]
        self.metrics['star_obs_1'][t] = params[2][1]
        self.metrics['star_obs_loc_x_0'][t] = params[3][0][0]
        self.metrics['star_obs_loc_x_1'][t] = params[3][1][0]
        self.metrics['star_obs_loc_y_0'][t] = params[3][0][1]
        self.metrics['star_obs_loc_y_1'][t] = params[3][1][1]

        self.metrics['info_gain_reward'][t] = self.info_gain_reward(t, selected_path, robot_model)
        self.metrics['MSE'][t] = self.MSE(robot_model, NTEST = 200)
        self.metrics['hotspot_error'][t] = self.hotspot_error(robot_model, NTEST = 200, NHS = 100)

        self.metrics['current_highest_obs'][t] = params[0]
        self.metrics['current_highest_obs_loc_x'][t] = params[1][0]
        self.metrics['current_highest_obs_loc_y'][t] = params[1][1]
        self.metrics['robot_location_x'][t] = selected_path[0][0]
        self.metrics['robot_location_y'][t] = selected_path[0][1]
        self.metrics['robot_location_a'][t] = selected_path[0][2]
    
    def plot_metrics(self):
        ''' Plots the performance metrics computed over the course of a info'''
        # Asumme that all metrics have the same time as MSE; not necessary
        time = np.array(self.metrics['MSE'].keys())
        
        ''' Metrics that require a ground truth global model to compute'''        
        info_gain = np.cumsum(np.array(self.metrics['info_gain_reward'].values()))        
        aqu_fun = np.cumsum(np.array(self.metrics['aquisition_function'].values()))
        MSE = np.array(self.metrics['MSE'].values())
        hotspot_error = np.array(self.metrics['hotspot_error'].values())
        
        regret = np.cumsum(np.array(self.metrics['instant_regret'].values()))
        info_regret = np.cumsum(np.array(self.metrics['max_val_regret'].values()))

        max_loc_error = np.array(self.metrics['max_loc_error'].values())
        max_val_error = np.array(self.metrics['max_val_error'].values())
        simple_regret = np.array(self.metrics['simple_regret'].values())

        sample_regret_loc = np.array(self.metrics['sample_regret_loc'].values())
        sample_regret_val = np.array(self.metrics['sample_regret_val'].values())

        current_highest_obs = np.array(self.metrics['current_highest_obs'].values())
        current_highest_obs_loc_x = np.array(self.metrics['current_highest_obs_loc_x'].values())
        current_highest_obs_loc_y = np.array(self.metrics['current_highest_obs_loc_y'].values())
        robot_location_x = np.array(self.metrics['robot_location_x'].values())
        robot_location_y = np.array(self.metrics['robot_location_y'].values())
        robot_location_a = np.array(self.metrics['robot_location_a'].values())
        star_obs_0 = np.array(self.metrics['star_obs_0'].values())
        star_obs_1 = np.array(self.metrics['star_obs_1'].values())
        star_obs_loc_x_0 = np.array(self.metrics['star_obs_loc_x_0'].values())
        star_obs_loc_x_1 = np.array(self.metrics['star_obs_loc_x_1'].values())
        star_obs_loc_y_0 = np.array(self.metrics['star_obs_loc_y_0'].values())
        star_obs_loc_y_1 = np.array(self.metrics['star_obs_loc_y_1'].values())
        # star_obs_loc = np.array(self.metrics['star_obs_loc'].values())

        #mean = np.cumsum(np.array(self.metrics['mean_reward'].values()))
        #hotspot_info = np.cumsum(np.array(self.metrics['hotspot_info_reward'].values()))


        if not os.path.exists('./figures/' + str(self.reward_function)):
            os.makedirs('./figures/' + str(self.reward_function))
        ''' Save the relevent metrics as csv files '''
        np.savetxt('./figures/' + self.reward_function + '/metrics.csv', \
            (time.T, info_gain.T, aqu_fun.T, MSE.T, hotspot_error.T, max_loc_error.T, \
            max_val_error.T, simple_regret.T,  sample_regret_loc.T, sample_regret_val.T, \
            regret.T, info_regret.T, current_highest_obs.T, current_highest_obs_loc_x.T,current_highest_obs_loc_y.T, \
            robot_location_x.T, robot_location_y.T, robot_location_a.T, \
            star_obs_0.T, star_obs_loc_x_0.T, star_obs_loc_y_0.T, \
            star_obs_1.T, star_obs_loc_x_1.T, star_obs_loc_y_1.T))
        #np.savetxt('./figures/' + self.reward_function + '/aqu_fun.csv', aqu_fun)
        #np.savetxt('./figures/' + self.reward_function + '/MSE.csv', MSE)
        #np.savetxt('./figures/' + self.reward_function + '/hotspot_MSE.csv', hotspot_error)
        #np.savetxt('./figures/' + self.reward_function + '/max_loc_error.csv', max_loc_error)
        #np.savetxt('./figures/' + self.reward_function + '/max_val_error.csv', max_val_error)
        #np.savetxt('./figures/' + self.reward_function + '/simple_regret.csv', simple_regret)
        
        
        #fig, ax = plt.subplots(figsize=(8, 6))
        #ax.set_title('Accumulated Mean Reward')                     
        #plt.plot(time, mean, 'b')      
        
        #fig, ax = plt.subplots(figsize=(8, 6))
        #ax.set_title('Accumulated Hotspot Information Gain Reward')                             
        #plt.plot(time, hotspot_info, 'r')          
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Average Regret w.r.t. ' + self.reward_function + ' Reward')                     
        plt.plot(time, regret/time, 'b')
        fig.savefig('./figures/' + self.reward_function + '/snapping_regret.png')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Average Info Regret w.r.t. ' + self.reward_function + ' Reward')                     
        plt.plot(time, info_regret/time, 'b')
        fig.savefig('./figures/' + self.reward_function + '/snapping_info_regret.png')

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Accumulated Information Gain')                             
        plt.plot(time, info_gain, 'k')        
        fig.savefig('./figures/' + self.reward_function + '/information_gain.png')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Accumulated Aquisition Function')             
        plt.plot(time, aqu_fun, 'g')
        fig.savefig('./figures/' + self.reward_function + '/aqu_fun.png')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Max Location Error')                             
        plt.plot(time, max_loc_error, 'k')        
        fig.savefig('./figures/' + self.reward_function + '/error_location.png')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Max Value Error')                             
        plt.plot(time, max_val_error, 'k')        
        fig.savefig('./figures/' + self.reward_function + '/error_value.png')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Simple Regret w.r.t. Global Maximizer')                     
        plt.plot(time, simple_regret, 'b')        
        fig.savefig('./figures/' + self.reward_function + '/simple_regret.png')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Map MSE at 100 Random Test Points')                             
        plt.plot(time, MSE, 'r')  
        fig.savefig('./figures/' + self.reward_function + '/mse.png')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Map Hotspot Error at 100 Random Test Points')                             
        plt.plot(time, hotspot_error, 'r')  
        fig.savefig('./figures/' + self.reward_function + '/hotspot_mse.png')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Average sample loc distance to Maximizer')                             
        plt.plot(time, sample_regret_loc, 'r')
        fig.savefig('./figures/' + self.reward_function + '/sample_regret_loc.png')
  
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Average sample val distance to Maximizer')
        plt.plot(time, sample_regret_val, 'r')  
        fig.savefig('./figures/' + self.reward_function + '/sample_regret_val.png')
        
        #plt.show() 
        plt.close()



'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                    Aquisition Functions - should have the form:
    def alpha(time, xvals, robot_model, param), where:
        time (int): the current timestep of planning
        xvals (list of float tuples): representing a path i.e. [(3.0, 4.0), (5.6, 7.2), ... ])
        robot_model (GPModel object): the robot's current model of the environment
        param (mixed): some functions require specialized parameters, which is there this can be used
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def info_gain(time, xvals, robot_model, param = None):
    ''' Compute the information gain of a set of potential sample locations with respect to the underlying function conditioned or previous samples xobs'''        
    data = np.array(xvals)
    x1 = data[:,0]
    x2 = data[:,1]
    queries = np.vstack([x1, x2]).T   
    xobs = robot_model.xvals

    # If the robot hasn't taken any observations yet, simply return the entropy of the potential set
    if xobs is None:
        Sigma_after = robot_model.kern.K(queries)
        entropy_after, sign_after = np.linalg.slogdet(np.eye(Sigma_after.shape[0], Sigma_after.shape[1]) \
                                    + robot_model.variance * Sigma_after)
        #print "Entropy with no obs: ", entropy_after
        return 0.5 * sign_after * entropy_after

    all_data = np.vstack([xobs, queries])
    
    # The covariance matrices of the previous observations and combined observations respectively
    Sigma_before = robot_model.kern.K(xobs) 
    Sigma_total = robot_model.kern.K(all_data)       

    # The term H(y_a, y_obs)
    entropy_before, sign_before =  np.linalg.slogdet(np.eye(Sigma_before.shape[0], Sigma_before.shape[1]) \
                                    + robot_model.variance * Sigma_before)
    
    # The term H(y_a, y_obs)
    entropy_after, sign_after = np.linalg.slogdet(np.eye(Sigma_total.shape[0], Sigma_total.shape[1]) \
                                    + robot_model.variance * Sigma_total)

    # The term H(y_a | f)
    entropy_total = 2 * np.pi * np.e * sign_after * entropy_after - 2 * np.pi * np.e * sign_before * entropy_before
    #print "Entropy: ", entropy_total


    ''' TODO: this term seems like it should still be in the equation, but it makes the IG negative'''
    #entropy_const = 0.5 * np.log(2 * np.pi * np.e * robot_model.variance)
    entropy_const = 0.0

    # This assert should be true, but it's not :(
    #assert(entropy_after - entropy_before - entropy_const > 0)
    return entropy_total - entropy_const

    
def mean_UCB(time, xvals, robot_model, param = None):
    ''' Computes the UCB for a set of points along a trajectory '''
    data = np.array(xvals)
    x1 = data[:,0]
    x2 = data[:,1]
    queries = np.vstack([x1, x2]).T   
                              
    # The GPy interface can predict mean and variance at an array of points; this will be an overestimate
    mu, var = robot_model.predict_value(queries)
    
    delta = 0.9
    d = 20
    pit = np.pi**2 * (time + 1)**2 / 6.
    beta_t = 2 * np.log(d * pit / delta)

    return np.sum(mu) + np.sqrt(beta_t) * np.sum(np.fabs(var))


def hotspot_info_UCB(time, xvals, robot_model, param = None):
    ''' The reward information gathered plus the estimated exploitation value gathered'''
    data = np.array(xvals)
    x1 = data[:,0]
    x2 = data[:,1]
    queries = np.vstack([x1, x2]).T   
                              
    LAMBDA = 1.0 # TOOD: should depend on time
    mu, var = robot_model.predict_value(queries)
    
    delta = 0.9
    d = 20
    pit = np.pi**2 * (time + 1)**2 / 6.
    beta_t = 2 * np.log(d * pit / delta)

    return info_gain(time, xvals, robot_model) + LAMBDA * np.sum(mu) + np.sqrt(beta_t) * np.sum(np.fabs(var))


def sample_max_vals(robot_model, t, nK = 3, nFeatures = 300, visualize = True):
    ''' The mutual information between a potential set of samples and the local maxima'''
    # If the robot has not samples yet, return a constant value
    if robot_model.xvals is None:
        return None, None, None

    d = robot_model.xvals.shape[1] # The dimension of the points (should be 2D)     

    ''' Sample Maximum values i.e. return sampled max values for the posterior GP, conditioned on 
    current observations. Construct random freatures and optimize functions drawn from posterior GP.'''
    samples = np.zeros((nK, 1))
    locs = np.zeros((nK, 2))
    funcs = []
    delete_locs = []

    for i in xrange(nK):
        print "Starting global optimization", i, "of", nK
        logger.info("Starting global optimization {} of {}".format(i, nK))
        # Draw the weights for the random features
        # TODO: make sure this formula is correct
        W = np.random.normal(loc = 0.0, scale = np.sqrt(1./(robot_model.lengthscale ** 2.)), size = (nFeatures, d))
        b = 2 * np.pi * np.random.uniform(low = 0.0, high = 1.0, size = (nFeatures, 1))
        
        # Compute the features for xx
        Z = np.sqrt(2 * robot_model.variance / nFeatures) * np.cos(np.dot(W, robot_model.xvals.T) + b)
        
        # Draw the coefficient theta
        noise = np.random.normal(loc = 0.0, scale = 1.0, size = (nFeatures, 1))

        # TODO: Figure this code out
        if robot_model.xvals.shape[0] < nFeatures:
            #We adopt the formula $theta \sim \N(Z(Z'Z + \sigma^2 I)^{-1} y, I-Z(Z'Z + \sigma^2 I)Z')$.            
            try:
                Sigma = np.dot(Z.T, Z) + robot_model.noise * np.eye(robot_model.xvals.shape[0])
                mu = np.dot(np.dot(Z, np.linalg.inv(Sigma)), robot_model.zvals)
                [D, U] = np.linalg.eig(Sigma)
                U = np.real(U)
                D = np.real(np.reshape(D, (D.shape[0], 1)))

                R = np.reciprocal((np.sqrt(D) * (np.sqrt(D) + np.sqrt(robot_model.noise))))
                theta = noise - np.dot(Z, np.dot(U, R*(np.dot(U.T, np.dot(Z.T, noise))))) + mu
            except:
                # If Sigma is not positive definite, ignore this simulation
                print "[ERROR]: Sigma is not positive definite, ignoring simulation", i
                logger.warning("[ERROR]: Sigma is not positive definite, ignoring simulation {}".format(i))
                delete_locs.append(i)
                continue
        else:
            # $theta \sim \N((ZZ'/\sigma^2 + I)^{-1} Z y / \sigma^2, (ZZ'/\sigma^2 + I)^{-1})$.            
            try:
                Sigma = np.dot(Z, Z.T) / robot_model.noise + np.eye(nFeatures)
                Sigma = np.linalg.inv(Sigma)
                mu = np.dot(np.dot(Sigma, Z), robot_model.zvals) / robot_model.noise
                theta = mu + np.dot(np.linalg.cholesky(Sigma), noise)            
            except:
                # If Sigma is not positive definite, ignore this simulation
                print "[ERROR]: Sigma is not positive definite, ignoring simulation", i
                logger.warning("[ERROR]: Sigma is not positive definite, ignoring simulation {}".format(i))
                delete_locs.append(i)
                continue

            #theta = np.random.multivariate_normal(mean = np.reshape(mu, (nFeatures,)), cov = Sigma, size = (nFeatures, 1))
            
        # Obtain a function samples from posterior GP
        #def target(x): 
        #    pdb.set_trace()
        #    return np.dot(theta.T * np.sqrt(2.0 * robot_model.variance / nFeatures), np.cos(np.dot(W, x.T) + b)).T
        target = lambda x: np.dot(theta.T * np.sqrt(2.0 * robot_model.variance / nFeatures), np.cos(np.dot(W, x.T) + b)).T
        target_vector_n = lambda x: -target(x.reshape(1,2))
        
        # Can only take a 1D input
        #def target_gradient(x): 
        #    return np.dot(theta.T * -np.sqrt(2.0 * robot_model.variance / nFeatures), np.sin(np.dot(W, x.reshape((2,1))) + b) * W)
        target_gradient = lambda x: np.dot(theta.T * -np.sqrt(2.0 * robot_model.variance / nFeatures), np.sin(np.dot(W, x.reshape((2,1))) + b) * W)
        target_vector_gradient_n = lambda x: -np.asarray(target_gradient(x).reshape(2,))
                                                                    
        # Optimize the function
        status = False
        count = 0
        # Retry optimization up to 5 times; if hasn't converged, give up on this simulated world
        while status == False and count < 5:
            maxima, max_val, max_inv_hess, status = global_maximization(target, target_vector_n, target_gradient, 
                target_vector_gradient_n, robot_model.ranges, robot_model.xvals, visualize, 't' + str(t) + '.nK' + str(i))
            count += 1
        if status == False:
            delete_locs.append(i)
            continue
        
        samples[i] = max_val
        funcs.append(target)
        print "Max Value in Optimization \t \t", samples[i]
        logger.info("Max Value in Optimization \t {}".format(samples[i]))
        locs[i, :] = maxima
        
        #if max_val < np.max(robot_model.zvals) + 5.0 * np.sqrt(robot_model.noise) or \
        #    maxima[0] == robot_model.ranges[0] or maxima[0] == robot_model.ranges[1] or \
        #    maxima[1] == robot_model.ranges[2] or maxima[1] == robot_model.ranges[3]:
        if max_val < np.max(robot_model.zvals) + 5.0 * np.sqrt(robot_model.noise):
            samples[i] = np.max(robot_model.zvals) + 5.0 * np.sqrt(robot_model.noise)
            print "Max observed is bigger than max in opt:", samples[i]
            logger.info("Max observed is bigger than max in opt: {}".format(samples[i]))
            locs[i, :] = robot_model.xvals[np.argmax(robot_model.zvals)]

    samples = np.delete(samples, delete_locs, axis = 0)
    locs = np.delete(locs, delete_locs, axis = 0)

    # If all global optimizations fail, just return the max value seen so far
    if len(delete_locs) == nK:
        samples[0] = np.max(robot_model.zvals) + 5.0 * np.sqrt(robot_model.noise)
        locs[0, :] = robot_model.xvals[np.argmax(robot_model.zvals)]
   
    return samples, locs, funcs
      
def mves_maximal_set(time, xvals, robot_model, param):
    ''' Define the Acquisition Function for maximal-set information gain
   param is tuple (maxima, target) '''
    max_vals = param[0]
    max_locs = param[1]
    target = param[2]

    if max_vals is None:
        return 1.0

    data = np.array(xvals)
    x1 = data[:,0]
    x2 = data[:,1]
    queries = np.vstack([x1, x2]).T        
    d = queries.shape[1] # The dimension of the points (should be 2D)     

    # Initialize f, g
    f = 0
    for i in xrange(max_vals.shape[0]):
        # Compute the posterior mean/variance predictions and gradients.
        #mean, var = robot_model.predict_value(queries)
     
        #mean_before, var_before = robot_model.predict_value(np.reshape(max_locs[i], (1,2)))

        mean_before, var_before = robot_model.predict_value(queries)
        #print "Before mean var:", mean_before, var_before
     
        radius = 2.0
        radius_steps = 10
        angle_steps = 10
        ball_data = np.zeros(((radius_steps) * (angle_steps) + 1, queries .shape[1]))
        for ii, dist in enumerate(np.linspace(0., radius, radius_steps)):
            for jj, angle in enumerate(np.linspace(0., 2. * np.pi, angle_steps)):
                ball_data[ii*angle_steps + jj, :] = np.reshape(max_locs[i] + np.array([dist * np.cos(angle), dist * np.sin(angle)]), (1,2))
                #ball_data[ii*angle_steps + jj, :] = np.reshape(np.array([3., 3.]) + np.array([dist * np.cos(angle), dist * np.sin(angle)]), (1,2))
        ball_data[-1, :] = np.reshape(max_locs[i], (1,2))

        #observations = target[i](np.reshape(max_locs[i], (1,2)))
        observations = target[i](ball_data)
        #print "observations:", observations 
        temp_model = robot_model.add_data_and_temp_model(ball_data, observations)
        
        mean_after, var_after = robot_model.predict_value(queries, TEMP = True)
        #print "After mean var:", mean_after, var_after
       
        #print "before entroyp:", entropy_of_n(var_before)
        #print "after entroyp:", entropy_of_tn(a = None, b = max_vals[i], mu = mean_after, var = var_after)

        #f += sum(entropy_of_tn(None, np.max(robot_model.zvals), mean, var) - entropy_of_tn(None, np.max([robot_model.xvals, np.max(observations)]), mean_after, var_after))

        #utility = entropy_of_n(var_before) - entropy_of_tn(a = None, b = max_vals[i], mu = mean_after, var = var_after)
        utility = entropy_of_n(var_before) - entropy_of_n(var = var_after)
        #print "utility:", utility
        f += sum(utility)

    # Average f
    f = f / max_vals.shape[0]
    # f is an np array; return scalar value
    return f[0] 

def mves_maximal_set2(time, xvals, robot_model, param):
    ''' Define the Acquisition Function for maximal-set information gain
   param is tuple (maxima, target) '''

    max_vals = param[0]
    max_locs = param[1]
    target = param[2]

    #print "Max vals:", max_vals
    #print "Max locs:", max_locs
        
    if max_vals is None:
        return 1.0

    data = np.array(xvals)
    x1 = data[:,0]
    x2 = data[:,1]
    queries = np.vstack([x1, x2]).T        
    d = queries.shape[1] # The dimension of the points (should be 2D)     

    # Initialize f, g
    f = 0
    for i in xrange(max_vals.shape[0]):
        # Compute the posterior mean/variance predictions and gradients.
        #mean, var = robot_model.predict_value(queries)
        mean, var = robot_model.predict_value(queries)
     
        #mean_before, var_before = robot_model.predict_value(np.reshape(max_locs[i], (1,2)))
        #print "Before mean var:", mean, var

        observations = target[i](queries)
        temp_model = robot_model.add_data_and_temp_model(queries, observations)
        #print "observations:", observations 
        mean_after, var_after = robot_model.predict_value(np.reshape(max_locs[i], (1,2)), TEMP = True)
        #print "After mean var:", mean_after, var_after
       
        max_before = np.max(robot_model.zvals)
        #print "before entroyp:", entropy_of_tn(a = max_before, b = None, mu = mean, var = var)
        max_after = np.amax(np.vstack([observations, robot_model.zvals]), axis = None)
        #print "after entroyp:", entropy_of_tn(a = max_after, b = None, mu = mean_after, var = var_after)
        #f += sum(entropy_of_tn(None, np.max(robot_model.zvals), mean, var) - entropy_of_tn(None, np.max([robot_model.xvals, np.max(observations)]), mean_after, var_after))
        f += entropy_of_tn(a = max_before, b = None, mu = mean_before, var = var_before) - entropy_of_tn(a = max_after, b = None, mu = mean_after, var = var_after) + entropy_of_n(var)
        #print "f:", f
    # Average f
    f = f / max_vals.shape[0]
    # f is an np array; return scalar value
    return f[0][0]

def mves(time, xvals, robot_model, param):
    ''' Define the Acquisition Function and the Gradient of MES'''
    # Compute the aquisition function value f and garident g at the queried point x using MES, given samples
    # function maxes and a previous set of functino maxes
    maxes = param[0]
    # If no max values are provided, return default value
    if maxes is None:
        return 1.0

    data = np.array(xvals)
    x1 = data[:,0]
    x2 = data[:,1]
    queries = np.vstack([x1, x2]).T        
    
    d = queries.shape[1] # The dimension of the points (should be 2D)     

    # Initialize f, g
    f = 0
    for i in xrange(maxes.shape[0]):
        # Compute the posterior mean/variance predictions and gradients.
        #[meanVector, varVector, meangrad, vargrad] = mean_var(x, xx, ...
        #    yy, KernelMatrixInv{i}, l(i,:), sigma(i), sigma0(i));
        mean, var = robot_model.predict_value(queries)
        std_dev = np.sqrt(var)
        
        # Compute the acquisition function of MES.        
        gamma = (maxes[i] - mean) / var
        pdfgamma = sp.stats.norm.pdf(gamma)
        cdfgamma = sp.stats.norm.cdf(gamma)
        #f += sum(gamma * pdfgamma / (2.0 * cdfgamma) - np.log(cdfgamma))        
        #utility = gamma * pdfgamma / (2.0 * cdfgamma) - np.log(cdfgamma)
        '''
        print "---------------------"
        print "locs:", queries
        print "entropy of z:", entropy_of_n(var)
        print "entropy of tn:"
        print entropy_of_tn(a = maxes[i], b = None, mu = mean, var = var)
        print "means:", mean
        print "vars:", var
        print "maxes:", maxes[i]
        '''
        utility = entropy_of_n(var) - entropy_of_tn(a = None, b = maxes[i], mu = mean, var = var)
        #utility /= entropy_of_n(var) 
        #print "before:",  gamma * pdfgamma / (2.0 * cdfgamma) - np.log(cdfgamma)
        #print "utility:", utility
        f += sum(utility)

    # Average f
    f = f / maxes.shape[0]
    # f is an np array; return scalar value
    return f[0] 
    
def entropy_of_n(var):    
    return np.log(np.sqrt(2.0 * np.pi * var))

def entropy_of_tn(a, b, mu, var):
    ''' a (float) is the lower bound
        b (float) is the upper bound '''
    if a is None:
        Phi_alpha = 0
        phi_alpha = 0
        alpha = 0
    else:
        alpha = (a - mu) / var        
        Phi_alpha = sp.stats.norm.cdf(alpha)
        phi_alpha = sp.stats.norm.pdf(alpha)
    if b is None:
        Phi_beta = 1
        phi_beta = 0
        beta = 0
    else:
        beta = (b - mu) / var        
        Phi_beta = sp.stats.norm.cdf(beta)
        phi_beta = sp.stats.norm.pdf(beta)

    #print "phi_alpha", phi_alpha
    #print "Phi_alpha", Phi_alpha
    #print "phi_beta", phi_beta
    #print "Phi_beta", Phi_beta

    Z = Phi_beta - Phi_alpha
    #print (alpha * phi_alpha - beta * phi_beta) 
    
    return np.log(Z * np.sqrt(2.0 * np.pi * var)) + (alpha * phi_alpha - beta * phi_beta) / (2.0 * Z)

def global_maximization(target, target_vector_n, target_grad, target_vector_gradient_n, ranges, guesses, visualize, filename):
    ''' Perform efficient global maximization'''
    gridSize = 300
    # Create a buffer around the boundary so the optmization doesn't always concentrate there
    hold_ranges = ranges
    bb = ((ranges[1] - ranges[0])*0.05, (ranges[3] - ranges[2]) * 0.05)
    ranges = (ranges[0] + bb[0], ranges[1] - bb[0], ranges[2] + bb[1], ranges[3] - bb[1])
    
    # Uniformly sample gridSize number of points in interval xmin to xmax
    x1 = np.random.uniform(ranges[0], ranges[1], size = gridSize)
    x2 = np.random.uniform(ranges[2], ranges[3], size = gridSize)
    x1, x2 = np.meshgrid(x1, x2, sparse = False, indexing = 'xy')  
    
    Xgrid_sample = np.vstack([x1.ravel(), x2.ravel()]).T    
    Xgrid = np.vstack([Xgrid_sample, guesses])   
    
    # Get the function value at Xgrid locations
    y = target(Xgrid)
    max_index = np.argmax(y)   
    start = np.asarray(Xgrid[max_index, :])

    # If the highest sample point seen is ouside of the boundary, find the highest inside the boundary
    if start[0] < ranges[0] or start[0] > ranges[1] or start[1] < ranges[2] or start[1] > ranges[3]:
        y = target(Xgrid_sample)
        max_index = np.argmax(y)
        start = np.asarray(Xgrid_sample[max_index, :])
    
    if visualize:
        # Generate a set of observations from robot model with which to make contour plots
        x1vals = np.linspace(ranges[0], ranges[1], 100)
        x2vals = np.linspace(ranges[2], ranges[3], 100)
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy') # dimension: NUM_PTS x NUM_PTS       
        data = np.vstack([x1.ravel(), x2.ravel()]).T
        observations = target(data)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.set_xlim(hold_ranges[0:2])
        ax2.set_ylim(hold_ranges[2:])        
        ax2.set_title('Countour Plot of the Approximated World Model')     
        plot = ax2.contourf(x1, x2, observations.reshape(x1.shape), cmap = 'viridis', vmin = MIN_COLOR, vmax = MAX_COLOR, levels=np.linspace(MIN_COLOR, MAX_COLOR, 15))

    res = sp.optimize.minimize(fun = target_vector_n, x0 = start, method = 'SLSQP', \
            jac = target_vector_gradient_n, bounds = ((ranges[0], ranges[1]), (ranges[2], ranges[3])))

    if res['success'] == False:
        print "Failed to converge!"
        #print res

        logger.warning("Failed to converge! \n")
        return 0, 0, 0, False
    
    if visualize:
        # Generate a set of observations from robot model with which to make contour plots
        scatter = ax2.scatter(guesses[:, 0], guesses[:, 1], color = 'k', s = 20.0)
        scatter = ax2.scatter(res['x'][0], res['x'][1], color = 'r', s = 100.0)      

        if not os.path.exists('./figures/mes/opt'):
            os.makedirs('./figures/mes/opt')
        fig2.savefig('./figures/mes/opt/globalopt.' + str(filename) + '.png')
        #plt.show()
        plt.close()

    # print res
    return res['x'], -res['fun'], res['jac'], True


def exp_improvement(time, xvals, robot_model, param = None):
    ''' The aquisition function using expected information, as defined in Hennig and Schuler Entropy Search'''
    data = np.array(xvals)
    x1 = data[:,0]
    x2 = data[:,1]
    queries = np.vstack([x1,x2]).T

    mu, var = robot_model.predict_value(queries)
    avg_reward = 0

    if param == None:
        eta = 0.5
    else:
        eta = sum(param)/len(param)

    # z = (np.sum(mu)-eta)/np.sum(np.fabs(var))
    x = [m-eta for m in mu]
    x = np.sum(x)
    z = x/np.sum(np.fabs(var))
    big_phi = 0.5 * (1 + sp.special.erf(z/np.sqrt(2)))
    small_phi = 1/np.sqrt(2*np.pi) * np.exp(-z**2 / 2) 
    avg_reward = x*big_phi + np.sum(np.fabs(var))*small_phi#(np.sum(mu)-eta)*big_phi + np.sum(np.fabs(var))*small_phi
        
    return avg_reward
