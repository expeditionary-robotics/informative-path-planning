# !/usr/bin/python

'''
This script can be used as a library file for a simulated robot which selects either myopically or nonmyopically the most "rewarding" place to visit in a discretized belief model, and takes a step toward that location, using a Dubins curve path generator.

This script draws heavily from elements in the ipp_library.py file.
''' 

from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib import cm
from sklearn import mixture
from IPython.display import display
from scipy.stats import multivariate_normal
import numpy as np
import scipy as sp
import math
import os
import sys
import GPy as GPy
import dubins
import time
from itertools import chain
import pdb
import logging
logger = logging.getLogger('robot')
import ipp_library as il


# globals for plotting
# MIN_COLOR = 3.0
# MAX_COLOR = 7.5
MIN_COLOR = -25.
MAX_COLOR = 25.


class Reachable_Robot():
    '''This robot inherits from the Robot class, but uses a MCTS in order to perform global horizon planning'''
    
    def __init__(self, sample_world, start_loc = (0.0, 0.0, 0.0), extent = (-10., 10., -10., 10.), 
            kernel_file = None, kernel_dataset = None, prior_dataset = None, init_lengthscale = 10.0, 
            init_variance = 100.0, noise = 0.05, step_size = 1.5, turning_radius = 1, sample_step = 0.5, discretization = (10,10), evaluation = None, 
            f_rew = 'mean', create_animation = False, use_mcts = False, learn_params = False, computation_budget = 60, rollout_length = 6):
        ''' Initialize the robot class with a GP model, initial location, path sets, and prior dataset'''
                   
        # General params
        self.ranges = extent
        self.create_animation = create_animation
        self.eval = evaluation
        self.loc = start_loc
        self.sample_world = sample_world
        self.f_rew = f_rew
        self.maxes = []
        self.current_max = -1000
        self.current_max_loc = [-1,-1]
        self.max_locs = None
        self.max_vals = None
        self.learn_params = learn_params

        # Handle the type of reward function
        if f_rew == 'hotspot_info':
            self.aquisition_function = il.hotspot_info_UCB
        elif f_rew == 'mean':
            self.aquisition_function = mean_UCB_2  
        elif f_rew == 'info_gain':
            self.aquisition_function = il.info_gain
        elif f_rew == 'mes':
            self.aquisition_function = il.mves
        elif f_rew == 'exp_improve':
            self.aquisition_function = il.exp_improvement
        else:
            raise ValueError('Only \'hotspot_info\' and \'mean\' and \'info_gain\' and \'mes\' and \'exp_improve\' reward fucntions supported.')
        

        # Initialize the belief model
        self.GP = il.GPModel(ranges = extent, lengthscale = init_lengthscale, variance = init_variance)
                
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
            self.GP.add_data(prior_dataset[0], prior_dataset[1]) 

        # Params for nonmyopic planning
        self.use_mcts = use_mcts
        self.comp_budget = computation_budget
        self.roll_length = rollout_length

        # Params for navigation
        self.step_size = step_size
        self.sample_step = sample_step
        self.turning_radius = turning_radius

        x1vals = np.linspace(extent[0], extent[1], discretization[0])
        x2vals = np.linspace(extent[2], extent[3], discretization[1])
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy')
        self.goals = np.vstack([x1.ravel(), x2.ravel()]).T



    def choose_destination(self, t):
        '''
        Select the best location in the discretized world to navigate to
        Input: t, step number
        Output: None or location index
        '''

        value = {}
        param = None

        max_locs = max_vals = None
        if self.f_rew == 'mes':
            self.max_val, self.max_locs = il.sample_max_vals(self.GP, t = t)

        for i,goal in enumerate(self.goals):
            if self.f_rew == 'mes':
                param = self.max_val
            elif self.f_rew == 'exp_improve':
                param = [self.current_max]
            value[i] = self.aquisition_function(time=t,
                                                   xvals=[goal[0], goal[1]],
                                                   robot_model=self.GP,
                                                   param=param)

        try:
            return self.goals[max(value, key=value.get)], value[max(value, key=value.get)], self.goals, value, self.max_locs
        except:
            return None

    def take_step(self, goal):
        '''
        Create an intermediary goal towards the point of interest such that the robot only translates the step size specified
        Input: Goal
        Output: Navigable points to the intermediary goal
        '''
        coords = {}

        dist = np.sqrt((self.loc[0]-goal[0])**2 + (self.loc[1]-goal[1])**2)
        angle_to_goal = np.arcsin((goal[1]-self.loc[1])/(dist)) * np.sign(goal[0]-self.loc[0])
        print 'here'
        print self.loc
        print 'goal'
        print goal
        if dist > self.step_size:
            new_goal = (self.loc[0]+self.step_size*np.sin(np.pi/2-angle_to_goal), self.loc[1]+self.step_size*np.sin(angle_to_goal), angle_to_goal)
        else:
            print 'too close'
            new_goal = (goal[0], goal[1], angle_to_goal)
        print 'new_goal'
        print new_goal


        path = dubins.shortest_path(self.loc, new_goal, self.turning_radius)
        configurations, _ = path.sample_many(self.sample_step)
        configurations.append(new_goal)

        temp = []
        for i,config in enumerate(configurations):
            if config[0] > self.ranges[0] and config[0] < self.ranges[1] and config[1] > self.ranges[2] and config[1] < self.ranges[3]:
                # return config
                temp.append(config)
            else:
                print config
                temp = []
                # break

        # if len(temp) < 2:
        #     pass
        # else:
        #     coords[i] = temp

        return temp 

    def collect_observations(self, xobs):
        ''' Gather noisy samples of the environment and updates the robot's GP model.
        Input: 
            xobs (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2 '''
        zobs = self.sample_world(xobs)       
        self.GP.add_data(xobs, zobs)

        for z, x in zip (zobs, xobs):
            if z[0] > self.current_max:
                self.current_max = z[0]
                self.current_max_loc = [x[0],x[1]]

    def predict_max(self):
        # If no observations have been collected, return default value
        if self.GP.xvals is None:
            return np.array([0., 0.]).reshape(1,2), 0.

        ''' First option, return the max value observed so far '''
        #return self.GP.xvals[np.argmax(self.GP.zvals), :], np.max(self.GP.zvals)

        ''' Second option: generate a set of predictions from model and return max '''
        # Generate a set of observations from robot model with which to predict mean
        x1vals = np.linspace(self.ranges[0], self.ranges[1], 30)
        x2vals = np.linspace(self.ranges[2], self.ranges[3], 30)
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy') 
        data = np.vstack([x1.ravel(), x2.ravel()]).T
        observations, var = self.GP.predict_value(data)        

        return data[np.argmax(observations), :], np.max(observations)

    def planner(self, T):
        '''
        Select the best point to navigate to myopically or nonmyopically; and perform the simulated navigation
        '''
        self.trajectory = []

        for t in xrange(T):
            print "[", t, "] Current Location:  ", self.loc
            logger.info("[{}] Current Location: {}".format(t, self.loc))
            best_location, best_val, all_locations, all_values, max_locs = self.choose_destination(t = t)
            
            # Given this choice, update the evaluation metrics 
            pred_loc, pred_val = self.predict_max()
            print "Current predicted max and value: \t", pred_loc, "\t", pred_val
            logger.info("Current predicted max and value: {} \t {}".format(pred_loc, pred_val))
             # try:
            #     self.eval.update_metrics(len(self.trajectory), self.GP, all_paths, best_path, \
            #     value = best_val, max_loc = pred_loc, max_val = pred_val, params = [self.current_max, self.current_max_loc, self.max_val, self.max_locs]) 
            # except:
            #     max_locs = [[-1, -1], [-1, -1]]
            #     max_val = [-1,-1]
            #     self.eval.update_metrics(len(self.trajectory), self.GP, all_paths, best_path, \
            #             value = best_val, max_loc = pred_loc, max_val = pred_val, params = [self.current_max, self.current_max_loc, max_val, max_locs]) 


            # Given this choice, take a step in the right direction, obeying to the dynamics of the vehicle
            sampling_path = self.take_step(best_location)
            print sampling_path
            if len(sampling_path) == 0:
                break
            data = np.array(sampling_path)
            x1 = data[:,0]
            x2 = data[:,1]
            xlocs = np.vstack([x1, x2]).T

            self.collect_observations(xlocs)

            if t < T/3 and self.learn_params == True:
                self.GP.train_kernel()

            self.trajectory.append(sampling_path)

            
            # if self.create_animation:
            #     self.visualize_trajectory(screen = False, filename = t, best_path = sampling_path, 
            #             maxes = max_locs, all_paths = all_locations, all_vals = all_values)            

            self.loc = sampling_path[-1]
        # np.savetxt('./figures/' + self.f_rew+ '/robot_model.csv', (self.GP.xvals[:, 0], self.GP.xvals[:, 1], self.GP.zvals[:, 0]))

    def visualize_trajectory(self, screen = True, filename = 'SUMMARY', best_path = None, 
        maxes = None, all_paths = None, all_vals = None):      
        ''' Visualize the set of paths chosen by the robot 
        Inputs:
            screen (boolean): determines whether the figure is plotted to the screen or saved to file
            filename (string): substring for the last part of the filename i.e. '0', '1', ...
            best_path (path object)
            maxes (list of locations)
            all_paths (list of path objects)
            all_vals (list of all path rewards) 
            T (string or int): string append to the figure filename
        '''
        
        # Generate a set of observations from robot model with which to make contour plots
        x1vals = np.linspace(self.ranges[0], self.ranges[1], 100)
        x2vals = np.linspace(self.ranges[2], self.ranges[3], 100)
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy') 
        data = np.vstack([x1.ravel(), x2.ravel()]).T
        observations, var = self.GP.predict_value(data)        
        
       
        # Plot the current robot model of the world
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(self.ranges[0:2])
        ax.set_ylim(self.ranges[2:])
        plot = ax.contourf(x1, x2, observations.reshape(x1.shape), cmap = 'viridis', vmin = MIN_COLOR, vmax = MAX_COLOR, levels=np.linspace(MIN_COLOR, MAX_COLOR, 15))
        if self.GP.xvals is not None:
            scatter = ax.scatter(self.GP.xvals[:, 0], self.GP.xvals[:, 1], c='k', s = 20.0, cmap = 'viridis')                
        color = iter(plt.cm.cool(np.linspace(0,1,len(self.trajectory))))
       
        # Plot the current trajectory
        for i, path in enumerate(self.trajectory):
            c = next(color)
            f = np.array(path)
            plt.plot(f[:,0], f[:,1], c=c, marker='*')

        # If available, plot the current set of options available to robot, colored
        # by their value (red: low, yellow: high)
        if all_paths is not None:
            all_vals = [x for x in all_vals.values()]   
            path_color = iter(plt.cm.autumn(np.linspace(0, max(all_vals),len(all_vals))/ max(all_vals)))        
            path_order = np.argsort(all_vals)
            
            for index in path_order:
                c = next(path_color)                
                points = all_paths[all_paths.keys()[index]]
                f = np.array(points)
                plt.plot(f[:,0], f[:,1], c = c, marker='.')
               
        # If available, plot the selected path in green
        if best_path is not None:
            f = np.array(best_path)
            plt.plot(f[:,0], f[:,1], c = 'g', marker='*')
           
        # If available, plot the current location of the maxes for mes
        if maxes is not None:
            for coord in maxes:
                plt.scatter(coord[0], coord[1], color = 'r', marker = '*', s = 500.0)
            # plt.scatter(maxes[:, 0], maxes[:, 1], color = 'r', marker = '*', s = 500.0)
           
        # Either plot to screen or save to file
        if screen:
            plt.show()           
        else:
            if not os.path.exists('./figures/' + str(self.f_rew)):
                os.makedirs('./figures/' + str(self.f_rew))
            fig.savefig('./figures/' + str(self.f_rew)+ '/trajectory-N.' + str(filename) + '.png')
            #plt.show()
            plt.close()
            
    def visualize_world_model(self, screen = True, filename = 'SUMMARY'):
        ''' Visaulize the robots current world model by sampling points uniformly in space and 
        plotting the predicted function value at those locations.
        Inputs:
            screen (boolean): determines whether the figure is plotted to the screen or saved to file 
            filename (String): name of the file to be made
            maxes (locations of largest points in the world)
        '''
        # Generate a set of observations from robot model with which to make contour plots
        x1vals = np.linspace(self.ranges[0], self.ranges[1], 100)
        x2vals = np.linspace(self.ranges[2], self.ranges[3], 100)
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy') # dimension: NUM_PTS x NUM_PTS       
        data = np.vstack([x1.ravel(), x2.ravel()]).T
        observations, var = self.GP.predict_value(data)        
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.set_xlim(self.ranges[0:2])
        ax2.set_ylim(self.ranges[2:])        
        ax2.set_title('Countour Plot of the Robot\'s World Model')     
        plot = ax2.contourf(x1, x2, observations.reshape(x1.shape), cmap = 'viridis', vmin = MIN_COLOR, vmax = MAX_COLOR, levels=np.linspace(MIN_COLOR, MAX_COLOR, 15))

        # Plot the samples taken by the robot
        if self.GP.xvals is not None:
            scatter = ax2.scatter(self.GP.xvals[:, 0], self.GP.xvals[:, 1], c=self.GP.zvals.ravel(), s = 10.0, cmap = 'viridis')        
        if screen:
            plt.show()           
        else:
            if not os.path.exists('./figures/' + str(self.f_rew)):
                os.makedirs('./figures/' + str(self.f_rew))
            fig.savefig('./figures/' + str(self.f_rew)+ '/world_model.' + str(filename) + '.png')
            plt.close()
    
    def plot_information(self):
        ''' Visualizes the accumulation of reward and aquisition functions ''' 
        self.eval.plot_metrics()


def mean_UCB_2(time, xvals, robot_model, param=None):
    ''' Computes the UCB for a set of points along a trajectory '''
    data = xvals
    x1 = data[0]
    x2 = data[1]
    queries = np.vstack([x1, x2]).T   
                              
    # The GPy interface can predict mean and variance at an array of points; this will be an overestimate
    mu, var = robot_model.predict_value(queries)
    
    delta = 0.9
    d = 20
    pit = np.pi**2 * (time + 1)**2 / 6.
    beta_t = 2 * np.log(d * pit / delta)

    return np.sum(mu) + np.sqrt(beta_t) * np.sum(np.fabs(var))


if __name__ == '__main__':
    seed = 0#int(sys.argv[1])
    reward_function = 'mean'#sys.argv[2]

    if not os.path.exists('./figures/' + str(reward_function)): 
        os.makedirs('./figures/' + str(reward_function))
    logging.basicConfig(filename = './figures/'+ reward_function + '/robot.log', level = logging.INFO)
    logger = logging.getLogger('robot')
    from ipp_library import *

    # Create a random enviroment sampled from a GP with an RBF kernel and specified hyperparameters, mean function 0 
    # The enviorment will be constrained by a set of uniformly distributed  sample points of size NUM_PTS x NUM_PTS
    ''' Options include mean, info_gain, hotspot_info, and mes'''
    ranges = (0., 10., 0., 10.)

    world = il.Environment(ranges = ranges, # x1min, x1max, x2min, x2max constraints
                        NUM_PTS = 20, 
                        variance = 100.0, 
                        lengthscale = 1.0, 
                        visualize = True,
                        seed = seed)

    evaluation = il.Evaluation(world = world, reward_function = reward_function)

    # Create the point robot
    robot = Reachable_Robot(sample_world = world.sample_value, 
                  start_loc = (5.0, 5.0, 0.0), 
                  extent = ranges,
                  kernel_file = None,
                  kernel_dataset = None,
                  prior_dataset =  None, 
                  init_lengthscale = 1.0, 
                  init_variance = 100.0, 
                  noise = 0.0001,
                  step_size = 1.5, 
                  turning_radius = 0.05,
                  sample_step = 0.5,
                  evaluation = evaluation, 
                  f_rew = reward_function, 
                  create_animation = True) 

    robot.planner(T = 30)
    #robot.visualize_world_model(screen = True)
    robot.visualize_trajectory(screen = False)
    # robot.plot_information()