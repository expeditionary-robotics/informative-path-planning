# ~/usr/bin/python

'''
This library is for accessing the GPModel class, used in the IPP framework PLUMES

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''

# from IPython.displa/y import display
import numpy as np
import math
import os
import copy
from sets import Set
import GPy as GPy
from GPy.inference.latent_function_inference import exact_gaussian_inference
from GPy.util.linalg import pdinv, dpotrs, dpotri, symmetrify, jitchol, dtrtrs, tdot
from GPy.util import diag
import logging
import scipy as sp
logger = logging.getLogger('robot')
import pdb

import aq_library as aq_lib

class GPModel(object):
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
        self._maxima = None
        self._max_val = None
        
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
            # self.kern = GPy.kern.RBF(input_dim = self.dim, lengthscale = lengthscale, variance = variance, useGPU = True) 
        else:
            raise ValueError('Kernel type must by \'rbf\'')
            
        # Intitally, before any data is created, 
        self.model = None

    def predict_value(self, xvals, include_noise = False):
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

        # With no observations, predict 0 mean everywhere and prior variance
        if self.model == None:
            return np.zeros((n_points, 1)), np.ones((n_points, 1)) * self.variance
        
        # Else, return the predicted values
        mean, var = self.model.predict(xvals, full_cov = False, include_likelihood = include_noise)
        return mean, var        


    def add_data(self, xvals, zvals):
        ''' Public method that adds data to an the GP model.
        Inputs:
            xvals (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2
            zvals (float array): an nparray of floats representing sensor observations, with dimension NUM_PTS x 1 
        ''' 
        # Reset the maximum values
        print "Resetting the sampled maximum values"
        self._maxima = None
        self._max_val = None
       
        if self.xvals is None:
            self.xvals = xvals
        else:
            self.xvals = np.vstack([self.xvals, xvals])
            
        if self.zvals is None:
            self.zvals = zvals
        else:
            self.zvals = np.vstack([self.zvals, zvals])

        # If the model hasn't been created yet (can't be created until we have data), create GPy model
        if self.model == None or True:
            self.model = GPy.models.GPRegression(np.array(self.xvals), np.array(self.zvals), self.kern)
        # Else add to the exisiting model
        else:
            self.model.set_XY(X = np.array(self.xvals), Y = np.array(self.zvals))

    def posterior_samples(self, xvals, size=10, full_cov = True):
        fsim = self.model.posterior_samples_f(xvals, size, full_cov=full_cov).reshape((xvals.shape[0], -1))
        return fsim

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

    @property
    def maxima(self):
        ''' Property that returns the maxima for value calculations if already 
            set, or computes if new maxima not yet computed. ''' 
        if self._maxima is None:
            max_vals, max_locs, func = aq_lib.sample_max_vals(self) 
            self._maxima = (max_vals, max_locs, func)
        return self._maxima

class OnlineGPModel(GPModel):
    ''' This class inherits from the GP model class
        Implements online, recursive updates for a Gaussian Process using the 
        Woodbury-Morrison formula by modifying the Posteior class from the GPy Library 
    '''

    def __init__(self, ranges, lengthscale, variance, noise = 0.0001, dimension = 2, kernel = 'rbf'):
        super(OnlineGPModel, self).__init__(ranges, lengthscale, variance, noise, dimension, kernel)
        
        self._K_chol = None
        self._K = None
        #option 1:
        self._woodbury_chol = None
        self._woodbury_vector =  None
        self._woodbury_inv =  None

        #option 2:
        self._mean =  None
        self._covariance = None
        self._prior_mean = 0.
    
    def init_model(self, xvals, zvals):
        # Update internal data
        self.xvals = xvals
        self.zvals = zvals
    
        self._K = self.kern.K(self.xvals)

        Ky = self._K.copy()

        # Adds some additional noise to ensure well-conditioned
        diag.add(Ky, self.noise + 1e-8)
        Wi, LW, LWi, W_logdet = pdinv(Ky)

        self._woodbury_inv = Wi 
        self._woodbury_vector =  np.dot(self._woodbury_inv, self.zvals) 
        
        self._woodbury_chol = None 
        self._mean =  None
        self._covariance = None
        self._prior_mean = 0.
        self._K_chol = None

    def update_model(self, xvals, zvals, incremental = True):
        assert(self.xvals is not None)
        assert(self.zvals is not None)
        
        Kx = self.kern.K(self.xvals, xvals)

        # Update K matrix
        self._K = np.block([
            [self._K,    Kx],
            [Kx.T,      self.kern.K(xvals, xvals)] 
         ])

        # Update internal data
        self.xvals = np.vstack([self.xvals, xvals])
        self.zvals = np.vstack([self.zvals, zvals])

        # Update woodbury inverse, either incrementally or from scratch
        if incremental == True:
            Pinv = self.woodbury_inv
            Q = Kx
            R = Kx.T
            S = self.kern.K(xvals, xvals)
            M = S - np.dot(np.dot(R, Pinv), Q)
            # Adds some additional noise to ensure well-conditioned
            diag.add(M, self.noise + 1e-8)
            M, _, _, _ = pdinv(M)

            Pnew = Pinv + np.dot(np.dot(np.dot(np.dot(Pinv, Q), M), R), Pinv)
            Qnew = -np.dot(np.dot(Pinv, Q), M)
            Rnew = -np.dot(np.dot(M, R), Pinv)
            Snew = M

            self._woodbury_inv = np.block([
                [Pnew, Qnew],
                [Rnew, Snew]
            ])
        else:
            Ky = self.K.copy()
            # Adds some additional noise to ensure well-conditioned
            diag.add(Ky, self.noise + 1e-8)
            Wi, LW, LWi, W_logdet = pdinv(Ky)
            self._woodbury_inv = Wi 
        
        self._woodbury_vector = np.dot(self.woodbury_inv, self.zvals) 

        self._woodbury_chol = None 
        self._mean =  None
        self._covariance = None
        self._prior_mean = 0.
        self._K_chol = None

    def add_data(self, xvals, zvals):
        ''' Public method that adds data to an the GP model.
        Inputs:
            xvals (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2
            zvals (float array): an nparray of floats representing sensor observations, with dimension NUM_PTS x 1 
        ''' 
        # Reset the maximum values
        print "Resetting the sampled maximum values"
        self._maxima = None
        self._max_val = None

        if self.xvals is None:
            assert(self.zvals is None)
            self.init_model(xvals, zvals)
        else:
            assert(self.zvals is not None)
            self.update_model(xvals, zvals)

    def predict_value(self, xvals, include_noise = True, full_cov = False):
        # Calculate for the test point
        assert(xvals.shape[0] >= 1)            
        assert(xvals.shape[1] == self.dim)    
        n_points, input_dim = xvals.shape

        # With no observations, predict 0 mean everywhere and prior variance
        if self.xvals is None:
            return np.zeros((n_points, 1)), np.ones((n_points, 1)) * self.variance

        Kx = self.kern.K(self.xvals, xvals)
        mu = np.dot(Kx.T, self.woodbury_vector)
        if len(mu.shape)==1:
            mu = mu.reshape(-1,1)
        if full_cov:
            Kxx = self.kern.K(xvals)
            if self.woodbury_inv.ndim == 2:
                var = Kxx - np.dot(Kx.T, np.dot(self.woodbury_inv, Kx))
        else:
            Kxx = self.kern.Kdiag(xvals)
            var = (Kxx - np.sum(np.dot(self.woodbury_inv.T, Kx) * Kx, 0))[:,None]

        '''
        # Alternative way to compute predictions using the cholesky decompostion of the woodvery matrix
        Kx = self.kern.K(self.xvals, xvals)
        mu = np.dot(Kx.T, self.woodbury_vector)
        if len(mu.shape)==1:
            mu = mu.reshape(-1,1)
        if full_cov:
            Kxx = self.kern.K(xvals)
            tmp = dtrtrs(self.woodbury_chol, Kx)[0]
            var = Kxx - tdot(tmp.T)
        else:
            Kxx = self.kern.Kdiag(xvals)
            tmp = dtrtrs(self.woodbury_chol, Kx)[0]
            var = (Kxx - np.square(tmp).sum(0))[:,None]
        '''
 
        # If model noise should be inlcuded in the prediction
        if include_noise: 
            var += self.noise
        return mu, var
    
   
    ''' Sample from the Gaussian Process posterior '''
    def posterior_samples(self, xvals, size=10, full_cov = True):
        """
        Samples the posterior GP at the points X.

        :param X: The points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim)
        :param size: the number of a posteriori samples.
        :type size: int.
        :param full_cov: whether to return the full covariance matrix, or just the diagonal.
        :type full_cov: bool.
        :returns: fsim: set of simulations
        :rtype: np.ndarray (D x N x samples) (if D==1 we flatten out the first dimension)
        """
        m, v = self.predict_value(xvals, include_noise = True, full_cov = full_cov)

        def sim_one_dim(m, v):
            if not full_cov:
                return np.random.multivariate_normal(m.flatten(), np.diag(v.flatten()), size).T
            else:
                return np.random.multivariate_normal(m.flatten(), v, size).T

        num_data, input_dim = self.xvals.shape
        output_dim = self.zvals.shape[1]

        if output_dim == 1:
            return sim_one_dim(m, v)
        else:
            fsim = np.empty((output_dim, num_data, size))
            for d in range(output_dim):
                if (not full_cov) and v.ndim == 2:
                    fsim[d] = sim_one_dim(m[:, d], v[:, d])
                else:
                    fsim[d] = sim_one_dim(m[:, d], v)
        return fsim
    
    @property
    def K(self):
        if self._K is None:
            self._K = self.kern.K(self.xvals, self.xvals)
        return self._K
    
    @property
    def mean(self):
        """
        Posterior mean
        $$
        K_{xx}v
        v := \texttt{Woodbury vector}
        $$
        """
        if self._mean is None:
            self._mean = np.dot(self._K, self.woodbury_vector)
        return self._mean

    @property
    def covariance(self):
        """
        Posterior covariance
        $$
        K_{xx} - K_{xx}W_{xx}^{-1}K_{xx}
        W_{xx} := \texttt{Woodbury inv}
        $$
        """
        if self._covariance is None:
            #self._covariance = (np.atleast_3d(self._K) - np.tensordot(np.dot(np.atleast_3d(self.woodbury_inv).T, self._K), self._K, [1,0]).T).squeeze()
            self._covariance = self._K - self._K.dot(self.woodbury_inv).dot(self._K)
        return self._covariance

    @property
    def woodbury_chol(self):
        """
        return $L_{W}$ where L is the lower triangular Cholesky decomposition of the Woodbury matrix
        $$
        L_{W}L_{W}^{\top} = W^{-1}
        W^{-1} := \texttt{Woodbury inv}
        $$
        """
        if self._woodbury_chol is None:
            #compute woodbury chol from
            if self._woodbury_inv is not None:
                winv = np.atleast_3d(self._woodbury_inv)
                self._woodbury_chol = np.zeros(winv.shape)
                for p in range(winv.shape[-1]):
                    self._woodbury_chol[:,:,p] = pdinv(winv[:,:,p])[2]
            elif self._covariance is not None:
                raise NotImplementedError("TODO: check code here")
                B = self._K - self._covariance
                tmp, _ = dpotrs(self.K_chol, B)
                self._woodbury_inv, _ = dpotrs(self.K_chol, tmp.T)
                _, _, self._woodbury_chol, _ = pdinv(self._woodbury_inv)
            else:
                raise ValueError("insufficient information to compute posterior")
        return self._woodbury_chol

    @property
    def woodbury_inv(self):
        """
        The inverse of the woodbury matrix, in the gaussian likelihood case it is defined as
        $$
        (K_{xx} + \Sigma_{xx})^{-1}
        \Sigma_{xx} := \texttt{Likelihood.variance / Approximate likelihood covariance}
        $$
        """
        if self._woodbury_inv is None:
            if self._woodbury_chol is not None:
                self._woodbury_inv, _ = dpotri(self._woodbury_chol, lower=1)
                symmetrify(self._woodbury_inv)
            elif self._covariance is not None:
                B = np.atleast_3d(self._K) - np.atleast_3d(self._covariance)
                self._woodbury_inv = np.empty_like(B)
                for i in range(B.shape[-1]):
                    tmp, _ = dpotrs(self.K_chol, B[:,:,i])
                    self._woodbury_inv[:,:,i], _ = dpotrs(self.K_chol, tmp.T)
        return self._woodbury_inv

    @property
    def woodbury_vector(self):
        """
        Woodbury vector in the gaussian likelihood case only is defined as
        $$
        (K_{xx} + \Sigma)^{-1}Y
        \Sigma := \texttt{Likelihood.variance / Approximate likelihood covariance}
        $$
        """
        if self._woodbury_vector is None:
            self._woodbury_vector, _ = dpotrs(self.K_chol, self.mean - self._prior_mean)
        return self._woodbury_vector

    @property
    def K_chol(self):
        """
        Cholesky of the prior covariance K
        """
        if self._K_chol is None:
            self._K_chol = jitchol(self.K)
        return self._K_chol

class SpatialGPModel(GPModel):
    ''' This class inherits from the GP model class
        Implements a ``sparse'' GP using only nearest-neighbor points maintained using a KD tree
        Modifies the Posteior class from the GPy Library 
    '''
    def __init__(self, ranges, lengthscale, variance, xvals = None, zvals = None, noise = 0.0001, dimension = 2, kernel = 'rbf'):

        super(SpatialGPModel, self).__init__(ranges, lengthscale, variance, noise, dimension, kernel)

        self.batch_size = 20
        self.neighbor_radius = 1.5 #[meters]
        # self.neighbor_radius = 100 #[meters]
        self._prior_mean = 0.

        if xvals is not None:
            self.init_model(xvals, zvals)
    
    def init_model(self, xvals, zvals):
        # The Gaussian dataset; start with inital values
        self.xvals = xvals
        self.zvals = zvals

        # Create a KD tree
        self.spatial_tree = sp.spatial.KDTree(self.xvals, leafsize = 10)
        
        # The data waiting to be batch added to the tree once batch_size is reached
        self.xwait = None
        self.zwait = None

    def update_model(self, xvals, zvals, incremental = True):
        assert(self.xvals is not None)
        assert(self.zvals is not None)

        # Add data to the waiting queue
        if self.xwait is None:
            self.xwait = xvals
        else:
            self.xwait = np.vstack([self.xwait, xvals])
            
        if self.zwait is None:
            self.zwait = zvals
        else:
            self.zwait = np.vstack([self.zwait, zvals])

        if self.xwait.shape[0] >= self.batch_size:
            # Add data to the dataset queue
            if self.xvals is None:
                self.xvals = self.xwait.copy()
            else:
                self.xvals = np.vstack([self.xvals, self.xwait.copy()])
                
            if self.zvals is None:
                self.zvals = self.zwait.copy() 
            else:
                self.zvals = np.vstack([self.zvals, self.zwait.copy()])

            # Create a new KD tree
            self.spatial_tree = sp.spatial.KDTree(self.xvals, leafsize = 10)
            
            # Reset the waiting data
            self.xwait = None
            self.zwait = None


    def add_data(self, xvals, zvals):
        ''' Public method that adds data to an the GP model.
        Inputs:
            xvals (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2
            zvals (float array): an nparray of floats representing sensor observations, with dimension NUM_PTS x 1 
        ''' 
        # Reset the maximum values
        print "Resetting the sampled maximum values"
        self._maxima = None
        self._max_val = None

        if self.xvals is None:
            assert(self.zvals is None)
            self.init_model(xvals, zvals)
        else:
            assert(self.zvals is not None)
            self.update_model(xvals, zvals)

    def predict_value(self, xvals, include_noise = True, full_cov = False):
        # Calculate for the test point
        assert(xvals.shape[0] >= 1)            
        assert(xvals.shape[1] == self.dim)    
	n_points, input_dim = xvals.shape

        # With no observations, predict 0 mean everywhere and prior variance
        if self.xvals is None:
            return np.zeros((n_points, 1)), np.ones((n_points, 1)) * self.variance

        # Find neightbors within radius
        point_group = self.spatial_tree.query_ball_point(xvals, self.neighbor_radius)

        point_list = []
        for points in point_group:
            for index in points:
                point_list.append(index)

        point_set = Set(point_list)
        xpoints = [self.xvals[index] for index in point_set]
        zpoints = [self.zvals[index] for index in point_set]
        # print "Size before:", len(xpoints)

        # Brute force check the points in the waiting queue
        if self.xwait is not None and self.xwait.shape[0] > 0:
            wait_list = []
            for i, u in enumerate(self.xwait):
                for j, v in enumerate(xvals):
                    # if xvals.shape[0] < 10:
                    #     print "Comparing", i, j
                    #     print "Points:", u, v
                    dist = sp.spatial.distance.minkowski(u, v, p = 2.0)
                    if dist <= self.neighbor_radius:
                        wait_list.append(i)
                        # if xvals.shape[0] < 10:
                        #     print "Adding point", u

            
            # if xvals.shape[0] < 10:
            #     print "The wait list:", wait_list

            wait_set = Set(wait_list)
        
            xpoints = [self.xwait[index] for index in wait_set] + xpoints
            zpoints = [self.zwait[index] for index in wait_set] + zpoints
            # print "Size after:", len(xpoints)

        xpoints = np.array(xpoints).reshape(-1, 2)
        zpoints = np.array(zpoints).reshape(-1, 1)

        if xpoints.shape[0] == 0:
            "No nearby points!"
            return np.zeros((n_points, 1)), np.ones((n_points, 1)) * self.variance

        # if self.xvals is not None:
        #     print "Size of kernel array:", self.xvals
        # if self.xwait is not None:
        #     print "Size of wait array:", self.xwait.shape
        # if xpoints is not None:
        #     print "Size of returned points:", xpoints.shape

        Kx = self.kern.K(xpoints, xvals)
        K = self.kern.K(xpoints, xpoints)
        
        # Adds some additional noise to ensure well-conditioned
        Ky = K.copy()
        diag.add(Ky, self.noise + 1e-8)

        Wi, LW, LWi, W_logdet = pdinv(Ky)
        woodbury_inv = Wi 
        woodbury_vector =  np.dot(woodbury_inv, zpoints) 

        mu = np.dot(Kx.T, woodbury_vector)
        if len(mu.shape) == 1:
            mu = mu.reshape(-1,1)
        if full_cov:
            Kxx = self.kern.K(xvals)
            if self.woodbury_inv.ndim == 2:
                var = Kxx - np.dot(Kx.T, np.dot(woodbury_inv, Kx))
        else:
            Kxx = self.kern.Kdiag(xvals)
            var = (Kxx - np.sum(np.dot(woodbury_inv.T, Kx) * Kx, 0))[:,None]

        # If model noise should be included in the prediction
        if include_noise: 
            var += self.noise

        return mu, var
    
    ''' Sample from the Gaussian Process posterior '''
    def posterior_samples(self, xvals, size=10, full_cov = True):
        """
        Samples the posterior GP at the points X.

        :param X: The points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim)
        :param size: the number of a posteriori samples.
        :type size: int.
        :param full_cov: whether to return the full covariance matrix, or just the diagonal.
        :type full_cov: bool.
        :returns: fsim: set of simulations
        :rtype: np.ndarray (D x N x samples) (if D==1 we flatten out the first dimension)
        """
        m, v = self.predict_value(xvals, include_noise = True, full_cov = full_cov)

        def sim_one_dim(m, v):
            if not full_cov:
                return np.random.multivariate_normal(m.flatten(), np.diag(v.flatten()), size).T
            else:
                return np.random.multivariate_normal(m.flatten(), v, size).T

        num_data, input_dim = self.xvals.shape
        output_dim = self.zvals.shape[1]

        if output_dim == 1:
            return sim_one_dim(m, v)
        else:
            fsim = np.empty((output_dim, num_data, size))
            for d in range(output_dim):
                if (not full_cov) and v.ndim == 2:
                    fsim[d] = sim_one_dim(m[:, d], v[:, d])
                else:
                    fsim[d] = sim_one_dim(m[:, d], v)
        return fsim
