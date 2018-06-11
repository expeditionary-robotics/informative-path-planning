# ~/usr/bin/python

'''
This library is for accessing the GPModel class, used in the IPP framework PLUMES

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''

from IPython.display import display
import numpy as np
import math
import os
import GPy as GPy
import logging
logger = logging.getLogger('robot')


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

    def add_data_to_temp_model(self, xvals, zvals):
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
            # x1vals = np.linspace(self.ranges[0], self.ranges[1], 10)
            # x2vals = np.linspace(self.ranges[2], self.ranges[3], 10)
            # x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy')
            # Z = np.vstack([x1.ravel(), x2.ravel()]).T
            # # self.model = GPy.models.SparseGPRegression(X=np.array(self.xvals), Y=np.array(self.zvals),kernel=self.kern, Z=Z)
            self.model = GPy.models.GPRegression(np.array(self.xvals), np.array(self.zvals), self.kern)
        # Else add to the exisiting model
        else:
            # self.model = GPy.models.SparseGPRegression(X=np.array(self.xvals), Y=np.array(self.zvals),kernel=self.kern, Z=Z)
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
        if self.xvals is not None and self.zvals is not None:
            xvals = self.xvals
            zvals = self.zvals

            print "Optimizing kernel parameters given data"
            logger.info("Optimizing kernel parameters given data")
            # Initilaize a GP model (used only for optmizing kernel hyperparamters)
            self.m = GPy.models.GPRegression(np.array(xvals), np.array(zvals), self.kern)
            # self.m = GPy.models.models.SparseGPRegression(X=np.array(self.xvals), Y=np.array(self.zvals),kernel= self.kern, num_inducing=1000)
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
