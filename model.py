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

class GPModel:
    '''The GPModel class, which is a wrapper on top of GP that allows automatic saving and 
    loading of trained kernel paramters, initially parameterized with:
    
    xvals (float array) an nparray of floats representing observation locations, with dimension NUM_PTS x 2
    zvals (float array) an nparray of floats representing sensor observations, with dimension NUM_PTS x 1
    learn_kernel (boolean) a boolean flag determining whether the kernel function is optimized based on the input
        data (xvals, yvals)/loaded from a saved file, or left fixed
    variance (float) the variance parameter of the squared exponential kernel
    lengthscale (float) the lengthscale parameter of the squared exponential kernel
    noise (float) the sensor noise parameter of the squared exponential kernel
    dimension (float) the dimension of the environment (currently, only 2D environments are supported)
    kernel (string) the type of kernel (currently, only 'rbf' kernels are supported) '''     
    
    def __init__(self, xvals, zvals, learn_kernel, lengthscale, variance,  noise = 0.05, dimension = 2, kernel = 'rbf'):
        '''Initialize a GP regression model based on xvals and zvals with given kernel parameters. If the learn_kernel
            flag is set, either load pre-learned kernel parameters from file or optimize kernal based on data'''
        
        # The noise parameter of the sensor
        self.noise = noise
        
        # The dimension of the evironment
        if dimension == 2:
            self.dim = dimension
        else:
            raise ValueError('Environment must have dimension 2 \'rbf\'')

        if kernel == 'rbf':
            self.kern = GPy.kern.RBF(input_dim = self.dim, lengthscale = lengthscale, variance = variance) 
        else:
            raise ValueError('Kernel type must by \'rbf\'')
    
        if learn_kernel:
            # Read pre-trained kernel parameters from file, if avaliable
            if os.path.isfile('kernel_model.npy'):
                print "Loading kernel parameters from file"
                # Initialize GP model from file with prior data
                self.m = GPy.models.GPRegression(np.array(xvals), np.array(zvals), self.kern, initialize = False)
                self.m.update_model(False)
                self.m.initialize_parameter()
                self.m[:] = np.load('kernel_model.npy')
                self.m.update_model(True)
            else:
                print "Optimizing kernel parameters given data"
                # Initilaize GP model
                self.m = GPy.models.GPRegression(np.array(xvals), np.array(zvals), self.kern)
                # self.m.initialize_parameter() ### this method does not exist for regression model

                # Constrain the hyperparameters during optmization
                self.m.constrain_positive('')
                self.m['rbf.variance'].constrain_bounded(0.01, 10)
                self.m['rbf.lengthscale'].constrain_bounded(0.01, 10)
                self.m['Gaussian_noise.variance'].constrain_fixed(noise)

                # Train the kernel hyperparameters
                self.m.optimize_restarts(num_restarts = 2, messages = True)

                # Save the hyperparemters to file
                np.save('kernel_model.npy', self.m.param_array)
        else:
            # Directly initilaize GP model
            self.m = GPy.models.GPRegression(np.array(xvals), np.array(zvals), self.kern)
            # self.m.initialize_parameter() #### this method does not exist for refression model

    # Visualize the learned GP kernel
    def kernel_plot(self):
        _ = self.kern.plot()
        plt.ylim([-10, 10])
        plt.xlim([-10, 10])
        plt.show()