# !/usr/bin/python

import os
import time
import sys
import logging 
import numpy as np

from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib import cm

import aq_library as aqlib
import gpmodel_library as gplib
import evaluation_library as evalib
import paths_library as pathlib
from ipp_library import Environment
from ipp_library import GPModel

if __name__=='__main__':
    print("Hello World")
    
    ranges = (0., 10., 0., 10.)

    world = Environment(ranges = ranges, # x1min, x1max, x2min, x2max constraints
                    NUM_PTS = 20, 
                    variance = 100.0, 
                    lengthscale = 1.0, 
                    visualize = True,
                    seed = 3)
    x1observe = np.linspace(ranges[0], ranges[1], 8)
    x2observe = np.linspace(ranges[2], ranges[3], 8)
    x1observe, x2observe = np.meshgrid(x1observe, x2observe, sparse = False, indexing = 'xy')  
    data = np.vstack([x1observe.ravel(), x2observe.ravel()]).T
    observations = world.sample_value(data)

    # for each point in the path of the world, query for an observation, use that observation to update the GP, and
    # continue. log everything to make comparisons.
    # robot model
    rob_mod = GPModel(ranges = ranges, lengthscale = 1.0, variance = 100.0)
    
    #plotting params
    x1vals = np.linspace(ranges[0], ranges[1], 100)
    x2vals = np.linspace(ranges[2], ranges[3], 100)
    x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy') 
    data = np.vstack([x1.ravel(), x2.ravel()]).T

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(ranges[0:2])
    ax.set_ylim(ranges[2:])
    