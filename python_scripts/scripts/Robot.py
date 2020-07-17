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
# import glog as log
import logging as log
# import gpmodel_library as gp_lib
# from continuous_traj import continuous_traj

from Environment import *
from Evaluation import *
from GPModel import *
from MCTS import *
import Path_Generator as pg
import grid_map_ipp_module as grid 

class Robot:
    '''The Robot class, which includes the vehicles current model of the world, path set represetnation, and
        infromative path planning algorithm.
        
        * sample_world (method) a function handle that takes a set of locations as input and returns a set of observations
        * start_loc (tuple of floats) the location of the robot initially in 2-D space e.g. (0.0, 0.0)
        * ranges (tuple of floats): a tuple representing the max/min of 2D rectangular domain i.e. (-10, 10, -50, 50)
        * kernel_file (string) a filename specifying the location of the stored kernel values
        * kernel_dataset (tuple of nparrays) a tuple (xvals, zvals), where xvals is a Npoint x 2 nparray of type float
            and zvals is a Npoint x 1 nparray of type float 
        * prior_dataset (tuple of nparrays) a tuple (xvals, zvals), where xvals is a Npoint x 2 nparray of type float
            and zvals is a Npoint x 1 nparray of type float                
        * init_variance (float) the variance parameter of the squared exponential kernel
        * init_lengthscale (float) the lengthscale parameter of the squared exponential kernel
        * noise (float) the sensor noise parameter of the squared exponential kernel '''
    
    def __init__(self, sample_world, obstacle_world, start_loc = (0.0, 0.0, 0.0), ranges = (-10., 10., -10., 10.), kernel_file = None, 
            kernel_dataset = None, prior_dataset = None, init_lengthscale = 10.0, init_variance = 100.0, noise = 0.05, 
            path_generator = 'default', frontier_size = 6, horizon_length = 5, turning_radius = 1, sample_step = 0.5, 
            evaluation = None , f_rew = 'mean', computation_budget = 60, rollout_length = 6, input_limit = [0.0, 10.0, -30.0, 30.0],
             sample_number= 10, step_time = 5.0, grid_map = None, lidar = None, is_save_fig = False, gradient_on = False, grad_step = 0.05):
        ''' Initialize the robot class with a GP model, initial location, path sets, and prior dataset'''
        self.ranges = ranges
        self.eval = evaluation
        self.loc = start_loc # Initial location of the robot      
        self.sample_world = sample_world # A function handel that allows the robot to sample from the environment 
        self.obstacle_World = obstacle_world

        if f_rew == 'hotspot_info':
            self.aquisition_function = hotspot_info_UCB
        elif f_rew == 'mean':
            self.aquisition_function = mean_UCB  
        elif f_rew == 'info_gain':
            self.aquisition_function = info_gain
        else:
            raise ValueError('Only \'hotspot_info\' and \'mean\' and \'info_gain\' reward fucntions supported.')

        # Initialize the robot's GP model with the initial kernel parameters
        self.GP = GPModel( lengthscale = init_lengthscale, variance = init_variance)
                
        # If both a kernel training dataset and a prior dataset are provided, train the kernel using both
        if  kernel_dataset is not None and prior_dataset is not None:
            data = np.vstack([prior_dataset[0], kernel_dataset[0]])
            observations = np.vstack([prior_dataset[1], kernel_dataset[1]])
            self.GP.train_kernel(data, observations, kernel_file) 
        # Train the kernel using the provided kernel dataset
        elif kernel_dataset is not None:
            self.GP.train_kernel(kernel_dataset[0], kernel_dataset[1], kernel_file)
        # If a kernel file is provided, load the kernel parameters
        elif kernel_file is not None:
            self.GP.load_kernel()
        # No kernel information was provided, so the kernel will be initialized with provided values
        else:
            pass
        
        # Incorporate the prior dataset into the model
        if prior_dataset is not None:
            self.GP.set_data(prior_dataset[0], prior_dataset[1]) 

        # Ego vehicle's Grid Map
        if grid_map is not None:
            self.grid_map = grid_map
        
        #Lidar sensor class 
        if lidar is not None:
            self.lidar = lidar
            self.is_lidar = True
        else:
            self.is_lidar = False

        # The path generation class for the robot
        path_options = {'default':Path_Generator(frontier_size, horizon_length, turning_radius, sample_step, ranges),
                        'dubins': Dubins_Path_Generator(frontier_size, horizon_length, turning_radius, sample_step, ranges),
                        'equal_dubins': Dubins_EqualPath_Generator(frontier_size, horizon_length, turning_radius, sample_step, ranges)}
        self.path_generator = path_options[path_generator]
        
    def visualize_world_model(self):
        ''' Visaulize the robots current world model by sampling points uniformly in space and plotting the predicted
            function value at those locations. '''
        # Generate a set of observations from robot model with which to make contour plots
        x1vals = np.linspace(self.ranges[0], self.ranges[1], 100)
        x2vals = np.linspace(self.ranges[2], self.ranges[3], 100)
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy') # dimension: NUM_PTS x NUM_PTS       
        data = np.vstack([x1.ravel(), x2.ravel()]).T
        observations, var = self.GP.predict_value(data)        
        
        fig2 = plt.figure(figsize=(4, 3))
        ax2 = fig2.add_subplot(111)
        ax2.set_xlim(self.ranges[0:2])
        ax2.set_ylim(self.ranges[2:])        
        ax2.set_title('Countour Plot of the Robot\'s World Model')     
    
        plot = ax2.contourf(x1, x2, observations.reshape(x1.shape), cmap = 'viridis')
        # Plot the samples taken by the robot
        scatter = ax2.scatter(self.GP.xvals[:, 0], self.GP.xvals[:, 1], c = self.GP.zvals.ravel(), s = 10.0, cmap = 'viridis')        
        plt.show()           

    def choose_trajectory(self, T, t):
        ''' Select the best trajectory avaliable to the robot at the current pose, according to the aquisition function.
        Input: 
        * T (int > 0): the length of the planning horization (number of planning iterations) 
        * t (int > 0): the current planning iteration (value of a point can change with algortihm progress)'''
        paths = self.path_generator.get_path_set(self.loc)
        value = {}        

        for path, points in paths.items():
            value[path] =  self.aquisition_function(time = t, xvals = points, robot_model = self.GP)            
        try:
            return paths[max(value, key = value.get)], value[max(value, key = value.get)], paths
        except:
            return None
    
    def collect_observations(self, xobs):
        ''' Gather noisy samples of the environment and updates the robot's GP model.
        Input: 
        * xobs (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2 '''
        zobs = self.sample_world(xobs)       
        self.GP.add_data(xobs, zobs)

    def myopic_planner(self, T):
        ''' Gather noisy samples of the environment and updates the robot's GP model  
        Input: 
        * T (int > 0): the length of the planning horization (number of planning iterations)'''
        self.trajectory = []
        
        for t in xrange(T):
            # Select the best trajectory according to the robot's aquisition function
            best_path, best_val, all_paths = self.choose_trajectory(T = T, t = t)
            
            # Given this choice, update the evaluation metrics 
            self.eval.update_metrics(t, self.GP, all_paths, best_path)            
            
            if best_path == None:
                break
            data = np.array(best_path)
            x1 = data[:,0]
            x2 = data[:,1]
            xlocs = np.vstack([x1, x2]).T           
            
            if len(best_path) != 1:
                self.collect_observations(xlocs)
                
            self.trajectory.append(best_path)
            self.save_figure(t)
            # self.visualize_world_model()
            if len(best_path) == 1:
                self.loc = (best_path[-1][0], best_path[-1][1], best_path[-1][2]-0.45)
            else:
                self.loc = best_path[-1]
    
    def save_figure(self, t):
        rob_mod = self.GP
        ranges = self.ranges
        
        x1vals = np.linspace(self.ranges[0], self.ranges[1], 100)
        x2vals = np.linspace(self.ranges[2], self.ranges[3], 100)
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy') # dimension: NUM_PTS x NUM_PTS       
        data = np.vstack([x1.ravel(), x2.ravel()]).T
        observations, var = self.GP.predict_value(data) 

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(ranges[0:2])
        ax.set_ylim(ranges[2:])
        plot = ax.contourf(x1, x2, observations.reshape(x1.shape), cmap = 'viridis', vmin = -25, vmax = 25, levels=np.linspace(-25, 25, 15))
        if rob_mod.xvals is not None:
            scatter = ax.scatter(rob_mod.xvals[:, 0], rob_mod.xvals[:, 1], c='k', s = 20.0, cmap = 'viridis')
            fig.savefig('./figures/myopic/' + str(t) + '.png')
        plt.close()


    def visualize_trajectory(self):      
        ''' Visualize the set of paths chosen by the robot '''
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_xlim(self.ranges[0:2])
        ax.set_ylim(self.ranges[2:])
        
        color = iter(plt.cm.cool(np.linspace(0,1,len(self.trajectory))))
        
        for i,path in enumerate(self.trajectory):
            c = next(color)
            f = np.array(path)
            plt.plot(f[:,0], f[:,1], c=c, marker='*')
        plt.show()
    
    def plot_information(self, iteration, range_max, grad_step):
        ''' Visualizes the accumulation of reward and aquisition functions ''' 
        self.eval.plot_metrics(iteration, range_max, grad_step)

    def save_information(self):
        MSE, regret, mean, hotspot_info, info_gain, UCB = self.eval.save_metric()
        

        return MSE, regret, mean, hotspot_info, info_gain, UCB

class Nonmyopic_Robot(Robot):
    '''This robot inherits from the Robot class, but uses a MCTS in order to perform global horizon planning'''
    
    def __init__(self, sample_world, obstacle_world, start_loc = (0.0, 0.0, 0.0), ranges = (-10., 10., -10., 10.), kernel_file = None, 
            kernel_dataset = None, prior_dataset = None, init_lengthscale = 10.0, init_variance = 100.0, noise = 0.05, 
            path_generator = 'default', frontier_size = 6, horizon_length = 5, turning_radius = 1, sample_step = 0.5, 
            evaluation = None , f_rew = 'mean', computation_budget = 60, rollout_length = 6, input_limit = [0.0, 10.0, -30.0, 30.0],
             sample_number= 10, step_time = 5.0, grid_map = None, lidar = None, is_save_fig = False, gradient_on = False, grad_step = 0.05):
        ''' Initialize the robot class with a GP model, initial location, path sets, and prior dataset'''

        self.ranges = ranges
        self.eval = evaluation
        self.loc = start_loc # Initial location of the robot      
        self.sample_world = sample_world # A function handel that allows the robot to sample from the environment 
        self.obstacle_World = obstacle_world # Environment model which needs to check collision 
        self.total_value = {}
        self.fs = frontier_size
        self.save_fig = is_save_fig
        self.f_rew = f_rew
        self.gradient_on = gradient_on
        self.grad_step = grad_step

        if f_rew == 'hotspot_info':
            self.aquisition_function = hotspot_info_UCB
        elif f_rew == 'mean':
            self.aquisition_function = mean_UCB  
        elif f_rew == 'info_gain':
            self.aquisition_function = info_gain
        elif f_rew == 'mes':
            self.aquisition_function = aqlib.mves
            # self.f_rew = self.mean_reward 
        else:
            raise ValueError('Only \'hotspot_info\' and \'mean\' and \'info_gain\' reward fucntions supported.')
        
        # Initialize the robot's GP model with the initial kernel parameters
        self.GP = GPModel(lengthscale = init_lengthscale, variance = init_variance)
                
        # If both a kernel training dataset and a prior dataset are provided, train the kernel using both
        if  kernel_dataset is not None and prior_dataset is not None:
            data = np.vstack([prior_dataset[0], kernel_dataset[0]])
            observations = np.vstack([prior_dataset[1], kernel_dataset[1]])
            self.GP.train_kernel(data, observations, kernel_file) 
        # Train the kernel using the provided kernel dataset
        elif kernel_dataset is not None:
            self.GP.train_kernel(kernel_dataset[0], kernel_dataset[1], kernel_file)
        # If a kernel file is provided, load the kernel parameters
        elif kernel_file is not None:
            self.GP.load_kernel()
        # No kernel information was provided, so the kernel will be initialized with provided values
        else:
            pass
        
        # Incorporate the prior dataset into the model
        if prior_dataset is not None:
            self.GP.set_data(prior_dataset[0], prior_dataset[1]) 
        
        # Ego vehicle's Grid Map
        if grid_map is not None:
            self.grid_map = grid_map
        
        #Lidar sensor class 
        if lidar is not None:
            self.lidar = lidar
            self.is_lidar = True
        else:
            self.is_lidar = False

        # The path generation class for the robot
        path_options = {'default':pg.Path_Generator(frontier_size, horizon_length, turning_radius, sample_step, ranges),
                        'dubins': pg.Dubins_Path_Generator(frontier_size, horizon_length, turning_radius, sample_step, ranges),
                        'equal_dubins': pg.Dubins_EqualPath_Generator(frontier_size, horizon_length, turning_radius, sample_step, ranges)}
                        # 'continuous_traj': continuous_traj.continuous_traj_sampler( input_limit, sample_number, frontier_size, horizon_length, step_time, sample_step,ranges)}
        self.path_generator = path_options[path_generator]
        
        # Computation limits
        self.comp_budget = computation_budget
        self.roll_length = rollout_length

    def nonmyopic_planner(self, T=3):
        ''' Use a monte carlo tree search in order to perform long-horizon planning'''
        
        self.trajectory = []
                 
        for t in xrange(T):
            #computation_budget, belief, initial_pose, planning_limit, frontier_size, path_generator, aquisition_function, time
            # FIXME MCTS
            mcts = MCTS(self.ranges, self.obstacle_World, self.comp_budget, self.GP, self.loc, self.roll_length, self.fs, self.path_generator, self.aquisition_function, t, self.gradient_on, self.grad_step)
            best_path, cost = mcts.get_actions()

            # mcts = mc_lib.cMCTS(self.comp_budget, self.GP, self.loc, self.roll_length, self.path_generator, self.aquisition_function, self.f_rew, t, None, False, 'dpw')
            # # best_path, best, cost = mcts.get_best_child()      
            # sampling_path, best_path, best_val, all_paths, all_values, self.max_locs, self.max_val, self.target = mcts.choose_trajectory(t=t)
#             print best_path
            data = np.array(best_path)
            x1 = data[:,0]
            x2 = data[:,1]
            xlocs = np.vstack([x1, x2]).T
            all_paths, _ = self.path_generator.get_path_set(self.loc)

            free_paths = self.collision_check(all_paths)
            
            self.eval.update_metrics(t, self.GP, free_paths, best_path) 
            self.collect_observations(xlocs)
            self.collect_lidar_observations(xlocs)
            self.trajectory.append(best_path)

            if(self.save_fig == True):
                self.save_figure(t)

            if len(best_path) == 1:
                self.loc = (best_path[-1][0],best_path[-1][1],best_path[-1][2]-1.14)
            elif best_path[-1][0] < self.ranges[0] + 0.5 or best_path[-1][0] > self.ranges[1] - 0.5:
                self.loc = (best_path[-1][0],best_path[-1][1],best_path[-1][2]-1.14)
            elif best_path[-1][1] < self.ranges[2] + 0.5 or best_path[-1][0] > self.ranges[3] - 0.5:
                self.loc = (best_path[-1][0],best_path[-1][1],best_path[-1][2]-1.14)
            else:
                self.loc = best_path[-1]
    
    def collect_lidar_observations(self, xlocs):
        ''' Gather lidar observations of the environment and updates the robot's belief grid map.
        Input: 
        * xobs (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2 '''

        zobs = self.sample_world(xobs)       
        self.GP.add_data(xobs, zobs)

    def collision_check(self, path_dict):
        free_paths = {}
                    
        for key,path in path_dict.items():
            is_collision = 0
            for pt in path:
                if self.is_lidar:
                    #Get occupancy prob. form current belief map 
                    x, y = pt[0] - self.ranges[0]/2.0, pt[1] - self.ranges[2]/2.0 # (0,0) pt is Occupancy grid map's center
                    occ_val = self.lidar.get_occ_value(x,y)
                    if(occ_val > 0.15):
                        is_collision = 1 
                        print("Collision Occured!")
                else:
                    if(self.obstacle_World.in_obstacle(pt, 3.0)):
                        is_collision = 1
                        print("Collision Occured!")
            if(is_collision == 0):
                free_paths[key] = path
        
        return free_paths 
        
    def save_figure(self, t):
        rob_mod = self.GP
        ranges = self.ranges
        
        x1vals = np.linspace(self.ranges[0], self.ranges[1], 100)
        x2vals = np.linspace(self.ranges[2], self.ranges[3], 100)
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy') # dimension: NUM_PTS x NUM_PTS       
        data = np.vstack([x1.ravel(), x2.ravel()]).T
        observations, var = self.GP.predict_value(data) 

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(ranges[0:2])
        ax.set_ylim(ranges[2:])
        plot = ax.contourf(x1, x2, observations.reshape(x1.shape), cmap = 'viridis', vmin = -25, vmax = 25, levels=np.linspace(-25, 25, 15))
        # self.obstacle_World.draw_obstacles()
        for obs in self.obstacle_World.obstacles:
            x,y = obs.exterior.xy
            ax.plot(x,y)
        
        if rob_mod.xvals is not None:
            scatter = ax.scatter(rob_mod.xvals[:, 0], rob_mod.xvals[:, 1], c='k', s = 20.0, cmap = 'viridis')
            fig.savefig('./figures/nonmyopic/' + str(t) + '.png')
        plt.close()