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
from ipp_library import *

if __name__=='__main__':
    print("Hello World")
    #defining the path to take
    sample_step = 0.5
    ranges = (0., 10., 0., 10.)
    start = (0.25, 0.25, 0.0)
    path_length = 1.5*175
    coverage_path = [start]

    across = 9.75
    rise = 0.38
    cp = start
    waypoints = [cp]
    l = 0

    for i in range(0,51):
        if i%2 == 0:
            if cp[0] > ranges[1]/2:
                cp = (cp[0]-across+0.25, cp[1], cp[2])
                l += across
            else:
                cp = (cp[0]+across-0.25, cp[1], cp[2])
                l += across
        else:
            cp = (cp[0], cp[1]+rise, cp[2])
            l += rise
        waypoints.append(cp)

    x = [w[0] for w in waypoints]
    y = [w[1] for w in waypoints]

    samples = [start]
    extra = 0
    addit = 0
    last = start
    for i,w in enumerate(waypoints):
        if i%4 == 0:
            last = w[0]
            while last+sample_step <= waypoints[i+1][0]:
                samples.append((last+sample_step, w[1], w[2])) 
                last = samples[-1][0]
            remainder = across-last
        elif (i+1)%4 == 0:
            last = waypoints[i-1][0]
            while last-sample_step+(remainder) >= waypoints[i][0]:
                samples.append((last-sample_step+(remainder), w[1], w[2])) 
                last = samples[-1][0]
            remainder = across-last

    xs = [s[0] for s in samples]
    ys = [s[1] for s in samples]



    reward_function = 'mes'
    ranges = (0., 10., 0., 10.)

    world = Environment(ranges = ranges, # x1min, x1max, x2min, x2max constraints
                        NUM_PTS = 20, 
                        variance = 100.0, 
                        lengthscale = 1.0, 
                        visualize = True,
                        seed = 3)

    evaluation = Evaluation(world = world, 
                            reward_function = reward_function)

    # Gather some prior observations to train the kernel (optional)
    x1observe = np.linspace(ranges[0]+0.5, ranges[1]-0,5, 8)
    x2observe = np.linspace(ranges[2]+0.5, ranges[3]-0.5, 8)
    x1observe, x2observe = np.meshgrid(x1observe, x2observe, sparse = False, indexing = 'xy')  
    data = np.vstack([x1observe.ravel(), x2observe.ravel()]).T
    observations = world.sample_value(data)

    # Create the point robot
    robot = Nonmyopic_Robot(sample_world = world.sample_value, 
                start_loc = (5.0, 5.0, 0.0), 
                extent = ranges,
                kernel_file = None,
                kernel_dataset = None,
                prior_dataset =  None, 
                #prior_dataset =  (data, observations), 
                init_lengthscale = 1.0, 
                init_variance = 100.0, 
                noise = 0.0001,
                path_generator = 'default',
                frontier_size = 20, 
                horizon_length = 1.5, 
                turning_radius = 0.05, 
                sample_step = 0.5,
                evaluation = evaluation, 
                f_rew = reward_function, 
                create_animation = True,
                computation_budget = 5,
                rollout_length = 3) 

    robot.planner(T = 20)
    robot.visualize_world_model(screen = True)
    robot.visualize_trajectory(screen = True)
    robot.plot_information()

    # for p in samples:
    #     xobs = np.vstack([p[0], p[1]]).T
    #     zobs = world.sample_value(xobs)
    #     rob_mod.add_data(xobs, zobs)
    #     observations, var = rob_mod.predict_value(data)  
    #     # Plot the current robot model of the world
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     ax.set_xlim(ranges[0:2])
    #     ax.set_ylim(ranges[2:])
    #     plot = ax.contourf(x1, x2, observations.reshape(x1.shape), cmap = 'viridis', vmin = -25, vmax = 25, levels=np.linspace(-25, 25, 15))
    #     if rob_mod.xvals is not None:
    #         scatter = ax.scatter(rob_mod.xvals[:, 0], rob_mod.xvals[:, 1], c='k', s = 20.0, cmap = 'viridis')
    # # plt.show()

            