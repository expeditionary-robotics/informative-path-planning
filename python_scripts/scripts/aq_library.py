# !/usr/bin/python

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                    Aquisition Functions - should have the form:
    def alpha(time, xvals, robot_model, param), where:
        time (int): the current timestep of planning
        xvals (list of float tuples): representing a path i.e. [(3.0, 4.0), (5.6, 7.2), ... ])
        robot_model (GPModel object): the robot's current model of the environment
        param (mixed): some functions require specialized parameters, which is there this can be used
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib import cm
from sklearn import mixture
from IPython.display import display
from scipy.stats import multivariate_normal
import numpy as np
from functools import partial
import scipy as sp
import math
import os
import GPy as GPy
import copy
import dubins
import time
from itertools import chain
import pdb
import logging
import obstacles as obslib
logger = logging.getLogger('robot')

def info_gain(time, xvals, robot_model, param=None):
    ''' Compute the information gain of a set of potential sample locations with respect to the underlying function conditioned or previous samples xobs'''        
    data = np.array(xvals)
    x1 = data[:,0]
    x2 = data[:,1]

    # TODO: time will be constant during all simulations? 
    if robot_model.dimension == 2:
        queries = np.vstack([x1, x2]).T   
    elif robot_model.dimension == 3:
        queries = np.vstack([x1, x2, time * np.ones(len(x1))]).T   
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

def mean_UCB(time, xvals, robot_model, param=None, FVECTOR = False):
    ''' Computes the UCB for a set of points along a trajectory '''
    data = np.array(xvals)
    x1 = data[:,0]
    x2 = data[:,1]
    # TODO: time will be constant during all simulations? 
    if robot_model.dimension == 2:
        queries = np.vstack([x1, x2]).T   
    elif robot_model.dimension == 3:
        queries = np.vstack([x1, x2, time * np.ones(len(x1))]).T   
    
    if robot_model.xvals is None:
        if FVECTOR:
            return np.ones((data.shape[0], 1))
        else:
            return 1.0
                              
    # The GPy interface can predict mean and variance at an array of points; this will be an overestimate
    mu, var = robot_model.predict_value(queries)
    #mu_test, var_test = robot_model.predict_value_legacy(queries)
    #print "------Diff-----------:"
    #print mu - mu_test
    #print var - var_test

    delta = 0.9
    d = 20
    pit = np.pi**2 * (time + 1)**2 / 6.
    beta_t = 2 * np.log(d * pit / delta)

    if FVECTOR:
        return mu + np.sqrt(beta_t) * np.fabs(var)
    else:
        return np.sum(mu) + np.sqrt(beta_t) * np.sum(np.fabs(var))

    
def hotspot_info_UCB(time, xvals, robot_model, param=None):
    ''' The reward information gathered plus the estimated exploitation value gathered'''
    data = np.array(xvals)
    x1 = data[:,0]
    x2 = data[:,1]
    if robot_model.dimension == 2:
        queries = np.vstack([x1, x2]).T   
    elif robot_model.dimension == 3:
        queries = np.vstack([x1, x2, time * np.ones(len(x1))]).T   
                              
    LAMBDA = 1.0 # TOOD: should depend on time
    mu, var = robot_model.predict_value(queries)
    
    delta = 0.9
    d = 20
    pit = np.pi**2 * (time + 1)**2 / 6.
    beta_t = 2 * np.log(d * pit / delta)

    return info_gain(time, xvals, robot_model) + LAMBDA * np.sum(mu) + np.sqrt(beta_t) * np.sum(np.fabs(var))


def general_target(x, robot_model, nFeatures, W, theta, b):
    return np.dot(theta.T * np.sqrt(2.0 * robot_model.variance / nFeatures), np.cos(np.dot(W, x.T) + b)).T

def sample_max_vals(robot_model, t, nK = 3, nFeatures = 200, visualize = False, obstacles=obslib.FreeWorld(), f_rew='mes'):
    ''' The mutual information between a potential set of samples and the local maxima'''
    # If the robot has not samples yet, return a constant value
    if robot_model.xvals is None:
        return None, None, None

    d = robot_model.xvals.shape[1] # The dimension of the points (should be 2D)     

    ''' Sample Maximum values i.e. return sampled max values for the posterior GP, conditioned on 
    current observations. Construct random freatures and optimize functions drawn from posterior GP.'''
    samples = np.zeros((nK, 1))
    locs = np.zeros((nK, d))
    funcs = []
    delete_locs = []

    for i in xrange(nK):
        print "Starting global optimization", i, "of", nK
        logger.info("Starting global optimization {} of {}".format(i, nK))

        # Draw the weights for the random features
        # TODO: make sure this formula is correct
        if robot_model.dimension == 2:
            W = np.random.normal(loc = 0.0, scale = np.sqrt(1./(robot_model.lengthscale)), size = (nFeatures, d))
        elif robot_model.dimension == 3:
            W = np.random.normal(loc = 0.0, scale = np.sqrt(1./(robot_model.lengthscale[0])), size = (nFeatures, d))

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
        # target = lambda x: np.dot(theta.T * np.sqrt(2.0 * robot_model.variance / nFeatures), np.cos(np.dot(W, x.T) + b)).T
        # def target(x, W=W, theta=theta):
        #     W = copy.deepcopy(W)
        #     theta = copy.deepcopy(theta)
        #     return np.dot(theta.T * np.sqrt(2.0 * robot_model.variance / nFeatures), np.cos(np.dot(W, x.T) + b)).T
        # target = lambda x: np.dot(theta.T * np.sqrt(2.0 * robot_model.variance / nFeatures), np.cos(np.dot(W, x.T) + b)).T
        target = partial(general_target, robot_model=robot_model, nFeatures=nFeatures, theta=theta, W=W, b=b)
        target_vector_n = lambda x: -target(x.reshape(1, d))
        
        # Can only take a 1D input
        #def target_gradient(x): 
        #    return np.dot(theta.T * -np.sqrt(2.0 * robot_model.variance / nFeatures), np.sin(np.dot(W, x.reshape((2,1))) + b) * W)
        target_gradient = lambda x: np.dot(theta.T * -np.sqrt(2.0 * robot_model.variance / nFeatures), np.sin(np.dot(W, x.reshape((d, 1))) + b) * W)
        target_vector_gradient_n = lambda x: -np.asarray(target_gradient(x).reshape(d,))
                                                                    
        # Optimize the function
        status = False
        count = 0
        # Retry optimization up to 5 times; if hasn't converged, give up on this simulated world
        while status == False and count < 5:
            maxima, max_val, max_inv_hess, status = global_maximization(target,
                                                                        target_vector_n,
                                                                        target_gradient, 
                                                                        target_vector_gradient_n,
                                                                        robot_model.ranges,
                                                                        robot_model.xvals, 
                                                                        visualize, 
                                                                        't' + str(t) + '.nK' + str(i),
                                                                        obstacles,
                                                                        f_rew=f_rew,
                                                                        time=t)
            count += 1
        if status == False:
            delete_locs.append(i)
            continue
        
        samples[i] = np.array(max_val).reshape((1,1))
        funcs.append(copy.deepcopy(target))
        print "Max Value in Optimization \t \t", samples[i]
        logger.info("Max Value in Optimization \t {}".format(samples[i]))
        locs[i, :] = maxima.reshape((1,d))
        
        #if max_val < np.max(robot_model.zvals) + 5.0 * np.sqrt(robot_model.noise) or \
        #    maxima[0] == robot_model.ranges[0] or maxima[0] == robot_model.ranges[1] or \
        #    maxima[1] == robot_model.ranges[2] or maxima[1] == robot_model.ranges[3]:

        '''
        if max_val < np.max(robot_model.zvals) + 5.0 * np.sqrt(robot_model.noise):
            samples[i] = np.max(robot_model.zvals) + 5.0 * np.sqrt(robot_model.noise)
            print "Max observed is bigger than max in opt:", samples[i]
            logger.info("Max observed is bigger than max in opt: {}".format(samples[i]))
            locs[i, :] = robot_model.xvals[np.argmax(robot_model.zvals)]
        '''

    print "Deleting values at:", delete_locs
    samples = np.delete(samples, delete_locs, axis = 0)
    locs = np.delete(locs, delete_locs, axis = 0)

    # If all global optimizations fail, just return the max value seen so far
    if len(delete_locs) == nK:
        samples[0] = np.max(robot_model.zvals) + 5.0 * np.sqrt(robot_model.noise)
        locs[0, :] = robot_model.xvals[np.argmax(robot_model.zvals)]
  

    print "Returning:", samples.shape, locs.shape
    return samples, locs, funcs

def mves(time, xvals, robot_model, param, FVECTOR = False):
    ''' Define the Acquisition Function and the Gradient of MES'''
    # Compute the aquisition function value f and garident g at the queried point x using MES, given samples
    # function maxes and a previous set of functino maxes
    maxes = param[0]
    # If no max values are provided, return default value
    if maxes is None:
        if FVECTOR:
            return np.ones((xvals.shape[0], 1))
        else:
            return 1.0

    data = np.array(xvals)
    x1 = data[:,0]
    x2 = data[:,1]
    if robot_model.dimension == 2:
        queries = np.vstack([x1, x2]).T   
    elif robot_model.dimension == 3:
        queries = np.vstack([x1, x2, time * np.ones(len(x1))]).T   
    
    d = queries.shape[1] # The dimension of the points (should be 2D)     

    # Initialize f, g
    if FVECTOR:
        f = np.zeros((data.shape[0], 1))
    else:
        f = 0

    for i in xrange(maxes.shape[0]):
        # Compute the posterior mean/variance predictions and gradients.
        #[meanVector, varVector, meangrad, vargrad] = mean_var(x, xx, ...
        #    yy, KernelMatrixInv{i}, l(i,:), sigma(i), sigma0(i));
        mean, var = robot_model.predict_value(queries)
        
        # Compute the acquisition function of MES.        
        gamma = (maxes[i] - mean) / var
        pdfgamma = sp.stats.norm.pdf(gamma)
        cdfgamma = sp.stats.norm.cdf(gamma)
        utility = gamma * pdfgamma / (2.0 * cdfgamma) - np.log(cdfgamma)
        
        if FVECTOR:
            f += utility
        else:
            f += sum(utility)

        #if np.sum(utility) == 0.000:
        #    pdb.set_trace()

        f += sum(utility)
    # Average f
    f = f / maxes.shape[0]
    # f is an np array; return scalar value
    if FVECTOR:
        return f # TODO: make this better! Dummy retrun
    else:
        # f is an np array; return scalar value
        return f[0]

def naive(time, xvals, robot_model, param, FVECTOR = False):
    ''' The naive reward function for the MSS problem where param is number of samples to draw and range for reward'''

    _, max_locs, _ = param[0]
    if max_locs is None:
        if FVECTOR:
            return np.zeros((xvals.shape[0], 1))
        else:
            return 0.0

    data = np.array(xvals)
    x1 = data[:, 0]
    x2 = data[:, 1]

    # Initialize f
    # pdb.set_trace()
    f = np.zeros((data.shape[0], 1))

    for i in xrange(max_locs.shape[0]):
        d = np.sqrt(np.square(x1-max_locs[i][0]) + np.square(x2-max_locs[i][1]))
        count = d <= param[1]
        f += count.astype(float).reshape(f.shape)
    f = f / max_locs.shape[0]

    if FVECTOR:
        return f # TODO: make this better! Dummy retrun
    else:
        # f is an np array; return scalar value
        return np.sum(f)

def naive_value(time, xvals, robot_model, param, FVECTOR = False):
    ''' The naive reward function for the MSS problem where param is number of samples to draw and range for reward'''

    max_vals, _, funcs = param[0]
    # if time > 0:
    #     pdb.set_trace()
    if max_vals is None or funcs is None:
        if FVECTOR:
            return np.zeros((xvals.shape[0], 1))
        else:
            return 0.0

    data = np.array(xvals)
    x1 = data[:, 0]
    x2 = data[:, 1]
    if robot_model.dimension == 2:
        queries = np.vstack([x1, x2]).T   
    elif robot_model.dimension == 3:
        queries = np.vstack([x1, x2, time * np.ones(len(x1))]).T   

    # Initialize f
    f = np.zeros((data.shape[0], 1))
    for i in xrange(max_vals.shape[0]):
        #simple value distance check between the query point and the maximum
        # mean, var = robot_model.predict_value(queries)
        mean = funcs[i](queries)
        # if FVECTOR:
        #     plt.imshow(mean.reshape((100,100)))
        #     plt.show()
        #     plt.close()
        diff = np.fabs(mean - max_vals[i][0])
        count = diff <= param[1]
        f += count.astype(float).reshape(f.shape)
        # if FVECTOR:
        #     plt.imshow(f.reshape((100,100)))
        #     plt.show()
        #     plt.close()

    f = f / float(max_vals.shape[0])
    # f is an np array; return scalar value
    if FVECTOR:
        return f
    else:
        # f is an np array; return scalar value
        return np.sum(f)

    
def entropy_of_n(var):    
    return np.log(np.sqrt(2.0 * np.pi * var))

def entropy_of_tn(a, b, mu, var):
    ''' a (float) is the lower bound
        b (float) is the uppper bound '''
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

    Z = Phi_beta - Phi_alpha
    
    return np.log(Z * np.sqrt(2.0 * np.pi * var)) + (alpha * phi_alpha - beta * phi_beta) / (2.0 * Z)


def global_maximization(target, target_vector_n, target_grad, target_vector_gradient_n, ranges, guesses, visualize, filename, obstacles, f_rew, time = None):
    MIN_COLOR = -25.
    MAX_COLOR = 25.

    ''' Perform efficient global maximization'''
    gridSize = 300
    # Create a buffer around the boundary so the optmization doesn't always concentrate there
    hold_ranges = ranges
    bb = ((ranges[1] - ranges[0])*0.10, (ranges[3] - ranges[2]) * 0.10)
    ranges = (ranges[0] + bb[0], ranges[1] - bb[0], ranges[2] + bb[1], ranges[3] - bb[1])
    
    dim = guesses.shape[1]

    # Uniformly sample gridSize number of points in interval xmin to xmax
    x1 = np.random.uniform(ranges[0], ranges[1], size = gridSize)
    x2 = np.random.uniform(ranges[2], ranges[3], size = gridSize)
    x1, x2 = np.meshgrid(x1, x2, sparse = False, indexing = 'xy')

    if dim == 2:
        Xgrid_sample = np.vstack([x1.ravel(), x2.ravel()]).T 
        Xgrid = np.vstack([Xgrid_sample, guesses])   
    elif dim == 3:
        Xgrid_sample = np.vstack([x1.ravel(), x2.ravel(), time * np.ones(len(x1.ravel()))]).T    
        Xgrid = Xgrid_sample
        # TODO: could potentially add previously sampled points back in if fix time component; unclear if necessary
        # Xgrid = np.vstack([Xgrid_sample, guesses[:, ]])   
   
    # Care only about the locations of the maxima guesses
    # TODO: need to care less about previous maxima
    
    # Get the function value at Xgrid locations
    y = target(Xgrid)
    print "y shape:", y.shape
    max_index = np.argmax(y)   
    print "Max index:", max_index
    start = np.asarray(Xgrid[max_index, :])

    # If the highest sample point seen is ouside of the boundary, find the highest inside the boundary
    if start[0] < ranges[0] or start[0] > ranges[1] or start[1] < ranges[2] or start[1] > ranges[3]:
        y = target(Xgrid_sample)
        max_index = np.argmax(y)
        start = np.asarray(Xgrid_sample[max_index, :])

    # If highest sample point is inside an obstacle, find one outside of the obstacle
    # xobs = []
    # yobs = []
    # if obstacles.in_obstacle((start[0], start[1])) == True:
    #     x1 = np.random.uniform(ranges[0], ranges[1], size = gridSize)
    #     x2 = np.random.uniform(ranges[2], ranges[3], size = gridSize)
    #     for x in x1:
    #         for y in x2:
    #             if obstacles.in_obstacle((x,y)) == True:
    #                 pass
    #             else:
    #                 xobs.append(x)
    #                 yobs.append(y)
    #     xobs = np.array(xobs)
    #     yobs = np.array(yobs)
    #     Xgrid_exception = np.vstack([xobs.ravel(), yobs.ravel()]).T

    #     y = target(Xgrid_exception)
    #     max_index = np.argmax(y)
    #     start = np.asarray(Xgrid_exception[max_index, :])

    
    if visualize:
        # Generate a set of observations from robot model with which to make contour plots
        x1vals = np.linspace(hold_ranges[0], hold_ranges[1], 100)
        x2vals = np.linspace(hold_ranges[2], hold_ranges[3], 100)
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy') # dimension: NUM_PTS x NUM_PTS       
        if dim == 2: 
            data = np.vstack([x1.ravel(), x2.ravel()]).T
        elif dim == 3:
            data = np.vstack([x1.ravel(), x2.ravel(), time * np.ones(len(x1.ravel()))]).T

        observations = target(data)
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.set_xlim(hold_ranges[0:2])
        ax2.set_ylim(hold_ranges[2:])        
        ax2.set_title('Countour Plot of the Approximated World Model')     
        plot = ax2.contourf(x1, x2, observations.reshape(x1.shape), 25, cmap = 'viridis')

    if dim == 2:
        res = sp.optimize.minimize(fun = target_vector_n, x0 = start, method = 'SLSQP', \
                jac = target_vector_gradient_n, bounds = ((ranges[0], ranges[1]), (ranges[2], ranges[3])))
    elif dim == 3:
        res = sp.optimize.minimize(fun = target_vector_n, x0 = start, method = 'SLSQP', \
                jac = target_vector_gradient_n, bounds = ((ranges[0], ranges[1]), (ranges[2], ranges[3]), (time, time)))

    if res['success'] == False:
        print "Failed to converge!"
        #print res

        logger.warning("Failed to converge! \n")
        return 0, 0, 0, False
    
    if visualize:
        # Generate a set of observations from robot model with which to make contour plots
        scatter = ax2.scatter(guesses[:, 0], guesses[:, 1], color = 'k', s = 20.0)
        scatter = ax2.scatter(res['x'][0], res['x'][1], marker = '*', color = 'r', s = 500)      

        if not os.path.exists('./figures/'+str(f_rew)+'/opt'):
            os.makedirs('./figures/'+str(f_rew)+'/opt')
        fig2.savefig('./figures/'+str(f_rew)+'/opt/globalopt.' + str(filename) + '.png')
        # plt.show()
        plt.close()
        plt.close('all')

    # print res
    return res['x'], -res['fun'], res['jac'], True


def exp_improvement(time, xvals, robot_model, param = None):
    ''' The aquisition function using expected information, as defined in Hennig and Schuler Entropy Search'''
    data = np.array(xvals)
    x1 = data[:,0]
    x2 = data[:,1]
    if robot_model.dimension == 2:
        queries = np.vstack([x1, x2]).T   
    elif robot_model.dimension == 3:
        queries = np.vstack([x1, x2, time * np.ones(len(x1))]).T   

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
