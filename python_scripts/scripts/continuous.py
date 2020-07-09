# Necessary imports
# %matplotlib inline

from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib import cm
from sklearn import mixture
# from IPython.display import display
from scipy.stats import multivariate_normal
import numpy as np
import math
import os
import GPy as GPy
import dubins
import time
from itertools import chain
# import continuous_traj
# import mcts_library as mc_lib
# import glog as log
import logging as log
# import gpmodel_library as gp_lib
# from continuous_traj import continuous_traj

from Environment import *
from Evaluation import *
from GPModel import *
from MCTS import *
from Path_Generator import *
from Robot import *

class Planning_Result():
    def __init__(self, planning_type, world, obstacle_world, evaluation, reward_function, ranges, start_loc, input_limit, sample_number, time_step, display, gradient_on, gradient_step, iteration):
        self.iteration = iteration
        self.type = planning_type
        self.world = world
        self.obstacle_world = obstacle_world
        self.evaluation = evaluation
        self.reward_function = reward_function

        if(planning_type=='coverage'):
            self.coverage_planning(ranges, start_loc, time_step)
        elif(planning_type=='non_myopic'):
            self.non_myopic_planning(ranges, start_loc, input_limit, sample_number, time_step, display, gradient_on, gradient_step)
        elif(planning_type=='myopic'):
            self.myopic_planning(ranges, start_loc, time_step)

    def myopic_planning(self, ranges_, start_loc_, time_step):
        robot = Robot(sample_world = world.sample_value, 
                    start_loc = start_loc_, 
                    ranges = ranges_,
                    kernel_file = None,
                    kernel_dataset = None,
                    prior_dataset =  None, 
                    init_lengthscale = 3.0, 
                    init_variance = 100.0, 
                    noise = 0.05,
                    path_generator = 'equal_dubins',
                    frontier_size = 20, 
                    horizon_length = 5.0, 
                    turning_radius = 0.5, 
                    sample_step = 2.0,
                    evaluation = evaluation, 
                    f_rew = reward_function)
        robot.myopic_planner(T = time_step)
        robot.visualize_world_model()
        robot.visualize_trajectory()
        # robot.plot_information()

    def non_myopic_planning(self, ranges_, start_loc_, input_limit_, sample_number_,time_step, display, gradient_on, gradient_step):
        robot = Nonmyopic_Robot(sample_world = self.world.sample_value, obstacle_world= self.obstacle_world, 
                        start_loc = start_loc_, 
                        ranges = ranges_,
                        kernel_file = None,
                        kernel_dataset = None,
                        prior_dataset =  None, 
                        init_lengthscale = 3.0, 
                        init_variance = 100.0, 
                        noise = 0.05,
                        path_generator = 'equal_dubins',
                        frontier_size = 20, 
                        horizon_length = 5.0, 
                        turning_radius = 0.5, 
                        sample_step = 2.0,
                        evaluation = self.evaluation, 
                        f_rew = self.reward_function,
                        computation_budget = 1.0,
                        rollout_length = 3, input_limit=input_limit_, sample_number=sample_number_,
                        step_time = 5.0, is_save_fig=display, gradient_on= gradient_on, grad_step = gradient_step)

        robot.nonmyopic_planner(T = time_step)
        # robot.visualize_world_model()
        # robot.visualize_trajectory()
        range_max = ranges_[1]
        robot.plot_information(self.iteration, range_max, gradient_step)
        # return MSE, regret, mean, hotspot_info, info_gain, UCB

    def coverage_planning(self, ranges_, start_loc_, time_step):
        sample_step = 0.5
        # ranges = (0., 10., 0., 10.)
        ranges = ranges_
        # start = (0.25, 0.25, 0.0)
        start = start_loc_
        path_length = 1.5*175
        coverage_path = [start]

        across = 19.5
        rise = 1.0
        cp = start
        waypoints = [cp]
        l = 0

        for i in range(0,51):
            if i%2 == 0:
                if cp[0] > ranges[1]/2:
                    cp = (cp[0]-across+0.5, cp[1], cp[2])
                    l += across
                else:
                    cp = (cp[0]+across-0.5, cp[1], cp[2])
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

        plt.plot(x, y)
        plt.plot(xs[0:30], ys[0:30], 'r*')

        # for each point in the path of the world, query for an observation, use that observation to update the GP, and
        # continue. log everything to make comparisons.
        # robot model
        rob_mod = GPModel(lengthscale = 1.0, variance = 100.0)

        #plotting params
        x1vals = np.linspace(ranges[0], ranges[1], 100)
        x2vals = np.linspace(ranges[2], ranges[3], 100)
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy') 
        data = np.vstack([x1.ravel(), x2.ravel()]).T
        t = 0
        for p in samples:
            xobs = np.vstack([p[0], p[1]]).T
            zobs = world.sample_value(xobs)
            rob_mod.add_data(xobs, zobs)
            print(p)
            observations, var = rob_mod.predict_value(data)  
            # Plot the current robot model of the world
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlim(ranges[0:2])
            ax.set_ylim(ranges[2:])
            plot = ax.contourf(x1, x2, observations.reshape(x1.shape), cmap = 'viridis', vmin = -25, vmax = 25, levels=np.linspace(-25, 25, 15))
            if rob_mod.xvals is not None:
                scatter = ax.scatter(rob_mod.xvals[:, 0], rob_mod.xvals[:, 1], c='k', s = 20.0, cmap = 'viridis')
            fig.savefig('./figures/coverage/' + str(t) + '.png')
            plt.close()
            t = t + 1
        plt.show()
        


# if __name__=="__main__":
#     # Create a random enviroment sampled from a GP with an RBF kernel and specified hyperparameters, mean function 0 
#     # The enviorment will be constrained by a set of uniformly distributed  sample points of size NUM_PTS x NUM_PTS

#     # logger = log.getLogger("crumbs")
#     # logger.setLevel(logger.debug)

#     # fileHandler = log.FileHandler('./log/file.log')
#     # streamHandler = log.StreamHandler()

#     range_exp = False
#     range_max_list = [20.0, 50.0, 100.0, 200.0]
#     if(range_exp):
#         for range_max in range_max_list:
#             ranges = (0., range_max, 0., range_max)
#             start_loc = (0.5, 0.5, 0.0)
#             time_step = 150
#             display = False
#             gradient_on = True

#             gradient_step_list = [0.0, 0.05, 0.1, 0.15, 0.20]

#             ''' Options include mean, info_gain, and hotspot_info, mes'''
#             reward_function = 'mean'

#             world = Environment(ranges = ranges, # x1min, x1max, x2min, x2max constraints
#                                 NUM_PTS = 20, 
#                                 variance = 100.0, 
#                                 lengthscale = 3.0, 
#                                 visualize = False,
#                                 seed = 1)

#             evaluation = Evaluation(world = world, 
#                                     reward_function = reward_function)

#             # Gather some prior observations to train the kernel (optional)

#             x1observe = np.linspace(ranges[0], ranges[1], 5)
#             x2observe = np.linspace(ranges[2], ranges[3], 5)
#             x1observe, x2observe = np.meshgrid(x1observe, x2observe, sparse = False, indexing = 'xy')  
#             data = np.vstack([x1observe.ravel(), x2observe.ravel()]).T
#             observations = world.sample_value(data)

#             input_limit = [0.0, 10.0, -30.0, 30.0] #Limit of actuation 
#             sample_number = 10 #Number of sample actions 

#             planning_type = 'non_myopic'
            
#             for iteration in range(5):
#                 for gradient_step in gradient_step_list:
#                     print('range_max ' + str(range_max)+ ' iteration ' + str(iteration) + ' gradient_step ' + str(gradient_step))
#                     planning = Planning_Result(planning_type, ranges, start_loc, input_limit, sample_number, time_step, display, gradient_on, gradient_step, iteration)

#     else:
#         ranges = (0., 20., 0., 20.)
#         start_loc = (0.5, 0.5, 0.0)
#         time_step = 150
#         display = True
#         gradient_on = False

#         gradient_step_list = [0.0, 0.05, 0.1, 0.15, 0.20]

#         ''' Options include mean, info_gain, and hotspot_info, mes'''
#         reward_function = 'mean'

#         world = Environment(ranges = ranges, # x1min, x1max, x2min, x2max constraints
#                             NUM_PTS = 20, 
#                             variance = 100.0, 
#                             lengthscale = 3.0, 
#                             visualize = False,
#                             seed = 1)

#         evaluation = Evaluation(world = world, 
#                                 reward_function = reward_function)

#         # Gather some prior observations to train the kernel (optional)

#         # ranges = (0., 20., 0., 20.)
#         x1observe = np.linspace(ranges[0], ranges[1], 5)
#         x2observe = np.linspace(ranges[2], ranges[3], 5)
#         x1observe, x2observe = np.meshgrid(x1observe, x2observe, sparse = False, indexing = 'xy')  
#         data = np.vstack([x1observe.ravel(), x2observe.ravel()]).T
#         observations = world.sample_value(data)

#         input_limit = [0.0, 10.0, -30.0, 30.0] #Limit of actuation 
#         sample_number = 10 #Number of sample actions 

#         planning_type = 'myopic'
#         planning = Planning_Result(planning_type, ranges, start_loc, input_limit, sample_number, time_step, display, gradient_on, 0, 0)
        
#         # for iteration in range(5):
#         #     for gradient_step in gradient_step_list:
#         #         print('iteration ' + str(iteration) + ' gradient_step ' + str(gradient_step))
#         #         planning = Planning_Result(planning_type, ranges, start_loc, input_limit, sample_number, time_step, display, gradient_on, gradient_step, iteration)


    
