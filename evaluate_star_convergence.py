# !/usr/bin/python

'''
Script for running myopic experiments using the run_sim bash script.
Generally a function of convenience in the event of parallelizing simulation runs.
Note: some of the parameters may need to be set prior to running the bash script.
=======

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''
import os
import time
import sys
import logging
import numpy as np
import scipy as sp

import pdb

import aq_library as aqlib
import mcts_library as mctslib
import gpmodel_library as gplib 
import evaluation_library as evalib 
import paths_library as pathlib 
import envmodel_library as envlib 
import robot_library as roblib
import obstacles as obslib
import bag_utils as baglib

from scipy.spatial import distance

import matplotlib.pyplot as plt

''' Predict the maxima of a GP model '''
def predict_max(GP):
    # If no observations have been collected, return default value
    if GP.xvals is None:
        return np.array([0., 0.]), 0.

    ''' First option, return the max value observed so far '''
    #return self.GP.xvals[np.argmax(GP.zvals), :], np.max(GP.zvals)

    ''' Second option: generate a set of predictions from model and return max '''
    # Generate a set of observations from robot model with which to predict mean
    x1vals = np.linspace(GP.ranges[0], GP.ranges[1], 30)
    x2vals = np.linspace(GP.ranges[2], GP.ranges[3], 30)
    x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy') 

    data = np.vstack([x1.ravel(), x2.ravel()]).T
    observations, var = GP.predict_value(data)        
    max_loc, max_val = data[np.argmax(observations), :], np.max(observations)

    fig2, ax2 = plt.subplots(figsize=(8, 8))
    plot = ax2.contourf(x1, x2, observations.reshape(x1.shape), 25, cmap = 'viridis')
    scatter = ax2.scatter(GP.xvals[:, 0], GP.xvals[:, 1], c='k', s = 20.0, cmap = 'viridis')

    scatter = ax2.scatter(data[:, 0], data[:, 1], c='b', s = 10.0, cmap = 'viridis')
    scatter = ax2.scatter(max_loc[0], max_loc[1], c='r', s = 20.0, cmap = 'viridis')
    plt.show()

    return max_loc, max_val

''' Quantify entropy of star distribution and visaulize the star heatmap '''
def star_max_dist(GP, true_loc, true_val):
    # If no observations have been collected, return default value
    if GP.xvals is None: #TODO: remember to change this
        print "Skipping star analysis prediction!"
        return 0.0, 0.0, 0.0, 0.0

    max_vals, max_locs, func = aqlib.sample_max_vals(GP, t = 0, nK = 20)
    max_vals = np.array(max_vals).reshape((-1, 1))
    max_locs = np.array(max_locs).reshape((-1, 2))
    np.savetxt('./sl_sampled_maxes.csv', np.vstack((max_locs.T, max_vals.T)))
    SAVE_FLAG = True 

    true_loc = np.array(true_loc).reshape((-1, 2))
    true_val = np.array(true_val).reshape((-1, 1))

    # Compute average distance from stars to true loc
    dist_loc = distance.cdist(max_locs, true_loc, 'euclidean')
    dist_val = distance.cdist(max_vals, true_val, 'euclidean')

    NBINS = 50
    RANGE = np.array([(ranges[0], ranges[1]), (ranges[2], ranges[3])])

    # Create the star heatmap
    if SAVE_FLAG:
        plt.figure(figsize=(8,8))
        # plt.hist2d(max_locs[:, 0], max_locs[:, 1], bins = NBINS, normed = True, range = RANGE, cmap = 'magma', norm=mcolors.LogNorm())
        plt.hist2d(max_locs[:, 0], max_locs[:, 1], bins = NBINS, normed = True, range = RANGE, cmap = 'viridis')
        plt.colorbar()
        plt.savefig('./star_heatmap_test.png')
        # plt.show()
        plt.close()

    # Compute the histrogram entropy of the star distribution
    ALPHA = 0.99
    hist, xbins, ybins = np.histogram2d(max_locs[:, 0], max_locs[:, 1], bins = NBINS, normed = True, range = RANGE)
    uniform = np.ones(hist.shape) / np.sum(np.ones(hist.shape))
    histnorm = ALPHA * hist + (1. - ALPHA) * uniform 
    histnorm = histnorm / np.sum(histnorm)
    entropy_x = -np.sum(histnorm[histnorm > 0.0] * np.log(histnorm[histnorm > 0.0]))

    # Uniform santiy check
    # uniform = np.ones(hist.shape) / np.sum(np.ones(hist.shape))
    # unifrom_entropy = -np.sum(uniform[uniform > 0.0] * np.log(uniform[uniform > 0.0]))
    # print "Entropy of a uniform distribution:", unifrom_entropy

    hist_z, xbins_z = np.histogram(max_vals, bins = NBINS, density = True)
    uniform = np.ones(hist_z.shape) / np.sum(np.ones(hist_z.shape))
    hist_z = hist_z / np.sum(hist_z)
    entropy_z = -np.mean(np.log(hist_z[hist_z > 0.0]))

    return np.mean(dist_loc), np.mean(dist_val), entropy_x, entropy_z


print "User specified options: SEED, REWARD_FUNCTION, PATHSET, USE_COST, NONMYOPIC, GOAL_ONLY, TREE_TYPE, RUN_REAL"
# Allow selection of seed world to be consistent, and to run through reward functions
SEED =  int(sys.argv[1])
# SEED = 0 
REWARD_FUNCTION = sys.argv[2]
PATHSET = sys.argv[3]
USE_COST = (sys.argv[4] == "True")
NONMYOPIC = (sys.argv[5] == "True")
GOAL_ONLY = (sys.argv[6] == "True")
TREE_TYPE = sys.argv[7] # one of dpw or belief
RUN_REAL_EXP = (sys.argv[8] == "True") # one of dpw or belief

if RUN_REAL_EXP:
    MAX_COLOR = 2.50
    MIN_COLOR = -3.00
    # MAX_COLOR = None
    # MIN_COLOR = None
    ranges = (0.0, 50.0, 0.0, 50.0)
else:
    MAX_COLOR = 25.0
    MIN_COLOR = -25.0
    ranges = (0.0, 0.0, 10.0, 10.0)
# MAX_COLOR = None
# MIN_COLOR = None

# Parameters for plotting based on the seed world information
# Set up paths for logging the data from the simulation run
if not os.path.exists('./figures/' + str(REWARD_FUNCTION)): 
    os.makedirs('./figures/' + str(REWARD_FUNCTION))
logging.basicConfig(filename = './figures/'+ REWARD_FUNCTION + '/robot.log', level = logging.INFO)
logger = logging.getLogger('robot')

# Create a random enviroment sampled from a GP with an RBF kernel and specified hyperparameters, mean function 0 
# The enviorment will be constrained by a set of uniformly distributed  sample points of size NUM_PTS x NUM_PTS

# Create obstacle world
ow = obslib.FreeWorld()
# ow = obslib.ChannelWorld(ranges, (3.5, 7.), 3., 0.3)
# ow = obslib.BugTrap(ranges, (2.2, 3.0), 4.6, orientation = 'left', width = 5.0)
# ow = obslib.BlockWorld(ranges,12, dim_blocks=(1., 1.), centers=[(2.5, 2.5), (7.,4.), (5., 8.), (8.75, 6.), (3.5,6.), (6.,1.5), (1.75,5.), (6.2,6.), (8.,8.5), (4.2, 3.8), (8.75,2.5), (2.2,8.2)])


if RUN_REAL_EXP:
    ''' Bagging '''
    xfull, zfull = baglib.read_fulldataset()

    # Add subsampled data from a previous bagifle
    seed_bag = '/home/genevieve/mit-whoi/barbados/rosbag_15Jan_slicklizard/slicklizard_2019-01-15-20-22-16.bag'
    xseed, zseed = baglib.read_bagfile(seed_bag, 20)
   
    # PLUMES trials
    seed_bag = '/home/genevieve/mit-whoi/barbados/rosbag_16Jan_slicklizard/slicklizard_2019-01-17-03-01-44.bag'
    xobs, zobs = baglib.read_bagfile(seed_bag, 1)
    trunc_index = baglib.truncate_by_distance(xobs, dist_lim = 1000.0)
    print "PLUMES trunc:", trunc_index
    xobs = xobs[0:trunc_index, :]
    zobs = zobs[0:trunc_index, :]

    # Myopic trials
    # seed_bag = '/home/genevieve/mit-whoi/barbados/rosbag_16Jan_slicklizard/slicklizard_2019-01-17-03-43-09.bag'
    # xobs, zobs = baglib.read_bagfile(seed_bag, 1)
    # trunc_index = baglib.truncate_by_distance(xobs, dist_lim = 1000.0)
    # print "myopic trunc:", trunc_index
    # xobs = xobs[0:trunc_index, :]
    # zobs = zobs[0:trunc_index, :]

    # Lawnmower trials
    seed_bag = '/home/genevieve/mit-whoi/barbados/rosbag_16Jan_slicklizard/slicklizard_2019-01-16-16-12-40.bag'
    xobs, zobs = baglib.read_bagfile(seed_bag, 1)
    trunc_index = baglib.truncate_by_distance(xobs, dist_lim = 1000.0)
    print "Lawnmower trunc:", trunc_index
    xobs = xobs[0:trunc_index, :]
    zobs = zobs[0:trunc_index, :]

    xobs = np.vstack([xseed, xobs])
    zobs = np.vstack([zseed, zobs])
    
    LEN = 2.0122 
    VAR = 5.3373 / 10.0
    NOISE = 0.19836 / 10.0

    # Create the GP model
    # gp_world = gplib.GPModel(ranges, lengthscale = 4.0543111858072445, variance = 0.3215773006606948, noise = 0.0862445597387173)
    gp_world = gplib.GPModel(ranges, lengthscale = LEN, variance = VAR, noise = NOISE)
    gp_world.add_data(xfull[::5], zfull[::5])

    # VAR = 0.3215773006606948
    # LEN = 4.0543111858072445
    # NOISE = 0.0862445597387173

    LEN = 2.0122 
    VAR = 5.3373 / 10.0
    NOISE = 0.19836 / 10.0
else:
    gp_world = None
    # VAR = 50.0
    # LEN = 5.0
    # NOISE = 0.1
    VAR = 100.0
    LEN = 1.0
    NOISE = 1.0

world = envlib.Environment(ranges = ranges,
                           NUM_PTS = 100, 
                           variance = VAR,
                           lengthscale = LEN,
                           noise = NOISE,
                           visualize = True,
                           seed = SEED,
                           MAX_COLOR = MAX_COLOR,
                           MIN_COLOR = MIN_COLOR,
			   model = gp_world,
                           obstacle_world = ow)

# Create the evaluation class used to quantify the simulation metrics
evaluation = evalib.Evaluation(world = world, reward_function = REWARD_FUNCTION)

# Generate a prior dataset
'''
x1observe = np.linspace(ranges[0], ranges[1], 5)
x2observe = np.linspace(ranges[2], ranges[3], 5)
x1observe, x2observe = np.meshgrid(x1observe, x2observe, sparse = False, indexing = 'xy')  
data = np.vstack([x1observe.ravel(), x2observe.ravel()]).T
observations = world.sample_value(data)
'''

print "Creating robot!"
# Create the point robot
robot = roblib.Robot(sample_world = world.sample_value, #function handle for collecting observations
                     start_loc = (5.0, 5.0, 0.0), #where robot is instantiated
                     extent = ranges, #extent of the explorable environment
                     MAX_COLOR = MAX_COLOR,
                     MIN_COLOR = MIN_COLOR,
                     kernel_file = None,
                     kernel_dataset = None,
                     # prior_dataset =  (data, observations), 
                     prior_dataset =  (xobs, zobs), 
                     # prior_dataset = None,
                     init_lengthscale = LEN,
                     init_variance = VAR,
                     noise = NOISE,
                     # init_lengthscale = 2.0122,
                     # init_variance = 5.3373,
                     # noise = 0.19836,
                     # noise = float(sys.argv[1]),
                     path_generator = PATHSET, #options: default, dubins, equal_dubins, fully_reachable_goal, fully_reachable_step
                     goal_only = GOAL_ONLY, #select only if using fully reachable step and you want the reward of the step to only be the goal
                     frontier_size = 10,
                     horizon_length = 1.50, 
                     turning_radius = 0.11,
                     sample_step = 0.1,
                     evaluation = evaluation, 
                     f_rew = REWARD_FUNCTION, 
                     create_animation = True, #logs images to the file folder
                     learn_params = False, #if kernel params should be trained online
                     nonmyopic = NONMYOPIC,
                     discretization = (20, 20), #parameterizes the fully reachable sets
                     use_cost = USE_COST, #select if you want to use a cost heuristic
                     computation_budget = 250,
                     rollout_length = 1,
                     obstacle_world = ow, 
                     tree_type = TREE_TYPE) 

print "Done creating robot!"


if RUN_REAL_EXP:
    print "Evaluating!"

    true_val = np.array(world.max_val).reshape((-1, 1))
    true_loc = np.array(world.max_loc).reshape((-1, 2))

    ''' Compute error in maxima prediction '''
    loc_guess, val_guess = predict_max(robot.GP)
    err_x = np.linalg.norm(loc_guess - true_loc)
    err_z = np.linalg.norm(val_guess - true_val)
    print "Estimation error in x, z:", err_x, err_z

    ''' Compute entropy of star samples '''
    dist_err, val_err, h_x, h_z = star_max_dist(robot.GP, true_loc, true_val)
    print "Star estimation error in x, z:", dist_err, val_err
    print "Star entropy error in x, z:", h_x, h_z 


    ''' Compute proportions of data witihin delta-epsilon regions '''
    samp_dist_loc = distance.cdist(robot.GP.xvals, true_loc, 'euclidean')
    samp_dist_val = distance.cdist(robot.GP.zvals, true_val, 'euclidean')
    pdb.set_trace()
    print samp_dist_loc[samp_dist_loc < 10.0]
    print samp_dist_val[samp_dist_val < 0.5]

    prop_x = float(len(samp_dist_loc[samp_dist_loc < 10.])) / float(len(samp_dist_loc))
    prop_z = float(len(samp_dist_val[samp_dist_val < 0.5])) / float(len(samp_dist_val))
    print "Proportion in x, z:", prop_x, prop_z


    max_vals, max_locs, func = aqlib.sample_max_vals(robot.GP, t = 0, nK = 20, obstacles = ow)

    max_vals=  np.array(max_vals).reshape((-1, 1))
    max_locs = np.array(max_locs).reshape((-1, 2))

    dist_loc = distance.cdist(max_locs, true_loc, 'euclidean')
    dist_val = distance.cdist(max_vals, true_val, 'euclidean')
    np.savetxt('./sampled_maxes.csv', np.vstack((max_locs.T, max_vals.T)))
    np.savetxt('./true_maxes.csv', np.vstack((true_loc.T, true_val.T)))


    print "Distance mean location:", np.mean(dist_loc), "\t Value:", np.mean(dist_val)

    loc_kernel = sp.stats.gaussian_kde(max_locs.T)
    loc_kernel.set_bandwidth(bw_method=LEN)

    density_loc = loc_kernel(max_locs.T)
    density_loc = density_loc / np.sum(density_loc)
    entropy_loc = -np.mean(np.log(density_loc))
    print density_loc
    print "Entropy of star location:", entropy_loc

    val_kernel = sp.stats.gaussian_kde(max_vals.T)
    val_kernel.set_bandwidth(bw_method='silverman')

    density_val = val_kernel(max_vals.T)
    density_val = density_val / np.sum(density_val)
    entropy_val = -np.mean(np.log(density_val))
    print "Entropy of star value:", entropy_val

else:
    robot.planner(T = 150)

    robot.visualize_world_model(screen = True)
    robot.visualize_trajectory(screen = False) #creates a summary trajectory image
    robot.plot_information() #plots all of the metrics of interest

