from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
from sklearn import mixture
from IPython.display import display
from scipy.stats import multivariate_normal
import numpy as np
import math
import os

import GPy


class Environment:
    def __init__(self, ranges, gp):
        # Expect ranges to be a 4-tuple consisting of xmin, xmax, ymin, and ymax
        self.xmin = float(ranges[0])
        self.xmax = float(ranges[1])
        self.ymin = float(ranges[2])
        self.ymax = float(ranges[3]) 
        
        print "Environment onitialized with bounds X: (", self.xmin, ",", self.xmax, ")  Y:(", self.ymin, ",", self.ymax, ")" 
    
class GPModel:
    def __init__(self, xvals, yvals, lengthscale, variance, dimension = 2, noise = 0.05, kernel = 'rbf'):
        # The dimension of the evironment
        self.dim = dimension
        # The noise parameter of the sensor
        self.nosie = nosie
        
        if kernel == 'rbf':
            self.kern = GPy.kern.RBF(input_dim = self.dim, lengthscale = lengthscale, variance = variance)
        else:
            raise ValueError('Kernel type must by \'rbf\'')
        
        if os.path.isfile('kernel_model.npy'):
            print "Loading kernel parameters from file"
            
            # Initialize GP model from file
            self.m = GPy.models.GPRegression(np.array(xavls), np.array(yvals), self.kern, initialize = False)
            self.m.update_model(False)
            self.m.initialize_parameter()
            self.m[:] = np.load('kernel_model.npy')
            self.m.update_model(True)
            
        else:
            print "Optimizing kernel parameters"
            # Initilaize GP model
            self.m = GPy.models.GPRegression(np.array(xvals), np.array(yvals), self.kern)
            
            # Constrain the hyperparameters during optmization
            self.m.constrain_positive('')
            self.m['rbf.variance'].constrain_bounded(0.01, 10)
            self.m['rbf.lengthscale'].constrain_bounded(0.01, 10)
            self.m['Gaussian_noise.variance'].constrain_fixed(1e-2)
            
            # Train the kernel hyperparameters
            self.m.optimize_restarts(num_restarts = 2, messages = True)
            
            # Save the hyperparemters to file
            np.save('kernel_model.npy', self.m.param_array)
            
        # Visualize the learned GP kernel
        def kernel_plot(self):
            _ = self.kern.plot()
            plt.ylim([-1, 1])
            plt.xlim([-1, 1])
            plt.show()

            
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

def rv_sample(xvals, yvals):
    data = rv(xvals, yvals)
    return rv(xvals, yvals) + np.random.randn(xvals.shape[0], xvals.shape[1]) * 0.35

# Generate data from a Gaussian mixture model            
def generateGPData(ranges, training_points):
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

    # Define kernel
    kern = GPy.kern.RBF(input_dim = 2, lengthscale = 5.0, variance = 1.0)
    
    # Create and train parmeters of GP model
    m = GPy.models.GPRegression(data, np.reshape(ztrain, (data.shape[0], 1), kern))
    #m.optimize(messages = True, max_f_eval = 1000)
    
    fig = plt.figure()
    ax = fig.add_subplot(211, projection = '3d')
    surf = ax.plot_surface(xvals, yvals, zvals, cmap = cm.coolwarm, linewidth = 0)
    
    ax2 = fig.add_subplot(212, projection = '3d')
    print xtrain.shape
    print ytrain.shape
    print ztrain.shape
    scatter = ax2.scatter(xtrain, ytrain, ztrain, c = ztrain, cmap = cm.coolwarm)
    plt.show()
    # Sample inputs and outputs 2D data

ranges = (-10, 10, -10, 10)
training_points = 10

generateGPData(ranges, training_points)
#gp = GPModel(xvals, yvals, lengthscale = 10.0, variance = 0.5);            
#environemnt = Environment(ranges, gp)

'''
k = GPy.kern.RBF(input_dim=1,lengthscale=0.2)
X = np.linspace(0.,1.,500)
# 500 points evenly spaced over [0,1]
X = X[:,None]
# reshape X to make it n*D
mu = np.zeros((500))
# vector of the means
C = k.K(X,X)
# covariance matrix
# Generate 20 sample path with mean mu and covariance C
Z = np.random.multivariate_normal(mu,C,20)
print Z.shape
plt.figure()
# open new plotting window
for i in range(20):
    plt.plot(X[:],Z[i,:])
    
plt.show()
'''
