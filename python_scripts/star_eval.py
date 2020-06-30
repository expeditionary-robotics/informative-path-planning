
# !/usr/bin/python

''' Script for running myopic experiments using the run_sim bash script.  Generally a function of convenience in the event of parallelizing simulation runs.  Note: some of the parameters may need to be set prior to running the bash script.

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''
import os
import time
import sys
import logging
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

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
import matplotlib.colors as mcolors


from scipy.spatial import distance

prefix = '/home/genevieve/mit-whoi/temp/informative-path-planning/figures/'
samp_filename = 'sampled_maxes.csv'
max_filename = 'true_maxes.csv'

maxima_files = [prefix + 'mes_plumes/' + max_filename,
                prefix + 'mes_lawn/' + max_filename]

sample_files = [prefix + 'mes_plumes/' + samp_filename,
                prefix + 'mes_lawn/' + samp_filename]

labels = ['PLUMES',
          'LAWNMOWER']

NBINS = 50
RANGE = np.array([(0, 50), (0, 50)])
# st2d(xval, yval, bins=1000, range=np.array([(-6, 6), (-4.5, 4.5)]))
# xlim([-6, 6])
# ylim([-4.5, 4.5])

# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

for i, (label, max_file, samp_file) in enumerate(zip(labels, maxima_files, sample_files)):
    print "Analyzing", label
    sampled_maxes = np.loadtxt(samp_file).T
    true_maxes = np.loadtxt(max_file).T

    max_locs = sampled_maxes[:, 0:2].reshape((-1, 2))
    max_vals = sampled_maxes[:, 2].reshape((-1, 1))

    true_loc = true_maxes[0:2].reshape((-1, 2))
    true_val = true_maxes[2].reshape((-1, 1))

    dist_loc = distance.cdist(max_locs, true_loc, 'euclidean')
    dist_val = distance.cdist(max_vals, true_val, 'euclidean')

    print "Distance mean location:", np.mean(dist_loc), "\t Value:", np.mean(dist_val)

    plt.figure(figsize=(8,8))
    plt.hist2d(max_locs[:, 0], max_locs[:, 1], bins = NBINS, normed = True, range = RANGE, cmap = 'magma', norm=mcolors.LogNorm())
    # plt.hist2d(max_locs[:, 0], max_locs[:, 1], bins = NBINS, normed = True, range = RANGE, cmap = 'Greys' )
    # plt.hist2d(max_locs[:, 0], max_locs[:, 1], bins = NBINS, normed = True, range = RANGE, cmap = 'Greys', norm=mcolors.PowerNorm(0.01))
    # plt.hist2d(max_locs[:, 0], max_locs[:, 1], bins = NBINS, normed = True, cmap = 'plasma')
    # plt.xlim([0, 50])
    # plt.ylim([0, 50])
    plt.colorbar()
    plt.show()
    
    hist, xbins, ybins, _ = plt.hist2d(max_locs[:, 0], max_locs[:, 1], bins = NBINS, normed = True, range = RANGE)
    entropy_x = -np.mean(np.log(hist[hist > 0.0]))
    print "Entropy of star x-value distribution:", entropy_x

    hist_z, xbins_z, _ = plt.hist(max_vals, bins = NBINS, density = True)
    entropy_z = -np.mean(np.log(hist_z[hist_z > 0.0]))
    print "Entropy of star z-value distribution:", entropy_z

    # Uniform santiy check
    uniform = np.ones(hist.shape) / np.sum(np.ones(hist.shape))
    unifrom_entropy = -np.mean(np.log(uniform[uniform > 0.0]))
    print "Entropy of a uniform distribution:", unifrom_entropy

    # axes[i].set_title(label)
    # axes[i].hist2d(max_locs[:, 0], max_locs[:, 1], bins = NBINS)
    # # axes[i].imshow(hist)
    # axes[i].set_xlim(0, 50)
    # axes[i].set_ylim(0, 50)

plt.show()

    # loc_kernel = sp.stats.gaussian_kde(max_locs.T)
    # loc_kernel.set_bandwidth(bw_method=4.0)

    # density_loc = loc_kernel(max_locs.T)
    # density_loc = density_loc / np.sum(density_loc)
    # entropy_loc = -np.mean(np.log(density_loc))
    # print density_loc
    # print "Entropy of star location:", entropy_loc

    # val_kernel = sp.stats.gaussian_kde(max_vals.T)
    # val_kernel.set_bandwidth(bw_method='silverman')

    # density_val = val_kernel(max_vals.T)
    # density_val = density_val / np.sum(density_val)
    # entropy_val = -np.mean(np.log(density_val))
    # print "Entropy of star value:", entropy_val
