# !/usr/bin/python

'''
This library alows access to the evaluation metrics and criteria used to assess the PLUMES framework.
'''
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib import cm
from sklearn import mixture
from scipy.stats import multivariate_normal
import numpy as np
import scipy as sp
import math
import os
import GPy as GPy
import time
from itertools import chain
import pdb
import logging
logger = logging.getLogger('robot')

import heuristic_rewards as aqlib

class Evaluation:
    ''' The Evaluation class, which includes the ground truth world model and a selection of reward criteria.
    
    Inputs:
        world (Environment object): an environment object that represents the ground truth environment
        f_rew (string): the reward function. One of {hotspot_info, mean, info_gain, mes, exp_improve} 
    '''
    def __init__(self, world, reward_function='mean', num_stars=3):
        ''' Initialize the evaluation module and select reward function'''
        self.world = world

        if world.dim == 2:
            self.max_loc = world.GP.xvals[np.argmax(world.GP.zvals), :]
            self.max_val = np.max(world.GP.zvals)
        elif world.dim == 3:
            self.max_loc = self.world.models[0].xvals[np.argmax(self.world.models[0].zvals), 0:-1]
            self.max_val = np.max(self.world.models[0].zvals)

        self.reward_function = reward_function
        self.num_stars = num_stars

        print "World max value", self.max_val, "at location", self.max_loc
        logger.info("World max value {} at location {}".format(self.max_val, self.max_loc))
        
        self.metrics = {'aquisition_function': {},
                        'mean_reward': {}, 
                        'info_gain_reward': {},                         
                        'hotspot_info_reward': {}, 
                        'MSE': {},                         
                        'hotspot_error': {},                         
                        'max_loc_error': {},
                        'max_val_error': {},
                        'current_highest_obs': {},
                        'current_highest_obs_loc_x': {},
                        'current_highest_obs_loc_y': {},
                        'robot_location_x': {},
                        'robot_location_y': {},
                        'robot_location_a': {},
                        'distance_traveled': {},
                        'planning_iteration': {},
                        'robot_samples': {}
                       }
        
        for i in range(0,num_stars):
            self.metrics['star_obs_'+str(i)] = {}
            self.metrics['star_obs_loc_x_'+str(i)] = {}
            self.metrics['star_obs_loc_y_'+str(i)] = {}
        
        if reward_function == 'hotspot_info':
            self.f_rew = self.hotspot_info_reward
            self.f_aqu = aqlib.hotspot_info_UCB
        elif reward_function == 'mean':
            self.f_rew = self.mean_reward
            self.f_aqu = aqlib.mean_UCB      
        elif reward_function == 'info_gain':
            self.f_rew = self.info_gain_reward
            self.f_aqu = aqlib.info_gain   
        elif reward_function == 'mes':
            self.f_aqu = aqlib.mves
            self.f_rew = self.mean_reward 
        elif reward_function == 'maxs-mes':
            self.f_aqu = aqlib.mves_maximal_set
            self.f_rew = self.mean_reward 
        elif reward_function == 'exp_improve':
            self.f_aqu = aqlib.exp_improvement
            self.f_rew = self.mean_reward
        elif reward_function == 'gumbel':
            self.f_aqu = aqlib.gumbel_mves
            self.f_rew = self.mean_reward
        else:
            raise ValueError('Only \'mean\' and \'hotspot_info\' and \'info_gain\' and \' mes\' and \'maxs-mes\' and \'exp_improve\' reward functions currently supported.')    
    
    '''Reward Functions - should have the form (def reward(time, xvals, robot_model)), where:
        time (int): the current timestep of planning
        xvals (list of float tuples): representing a path i.e. [(3.0, 4.0), (5.6, 7.2), ... ])
        robot_model (GPModel)
    '''
    def mean_reward(self, time, xvals, robot_model):
        ''' Predcited mean (true) reward function'''
        data = np.array(xvals)
        x1 = data[:,0]
        x2 = data[:,1]
        if self.world.dim == 2:
            queries = np.vstack([x1, x2]).T   
            mu, var = self.world.GP.predict_value(queries)
        elif self.world.dim == 3:
            queries = np.vstack([x1, x2, time * np.ones(len(x1))]).T   
            mu, var = self.world.models[time].predict_value(queries)
       
        return np.sum(mu)   

    def hotspot_info_reward(self, time, xvals, robot_model):
        ''' The reward information gathered plus the exploitation value gathered'''    
        LAMBDA = 1.0 # TOOD: should depend on time
        data = np.array(xvals)
        x1 = data[:,0]
        x2 = data[:,1]
        if self.world.dim == 2:
            queries = np.vstack([x1, x2]).T   
            mu, var = self.world.GP.predict_value(queries)    
        elif self.world.dim == 3:
            queries = np.vstack([x1, x2, time * np.ones(len(x1))]).T   
            mu, var = self.world.models[time].predict_value(queries)
        
        return self.info_gain_reward(time, xvals, robot_model) + LAMBDA * np.sum(mu)
    
    def info_gain_reward(self, time, xvals, robot_model):
        ''' The information reward gathered '''
        return aqlib.info_gain(time, xvals, robot_model)
    
    def max_error(self, max_loc, max_val):
        ''' The error of the current best guess for the global maximizer
        Input:
            max_loc (nparray 1 x 2)
            max_val (float)
        '''
        return np.linalg.norm(max_loc[0:-1] - self.max_loc), np.linalg.norm(max_val - self.max_val)

    def hotspot_error(self, time, robot_model, NTEST = 100, NHS = 100):
        ''' Compute the hotspot error on a set of test points, randomly distributed throughout the environment'''
        x1 = np.random.random_sample((NTEST, 1)) * (self.world.x1max - self.world.x1min) + self.world.x1min
        x2 = np.random.random_sample((NTEST, 1)) * (self.world.x2max - self.world.x2min) + self.world.x2min
        x1 = x1.reshape((NTEST,))
        x2 = x2.reshape((NTEST,))

        if self.world.dim == 2:
            data = np.vstack([x1, x2]).T   
            pred_world, var_world = self.world.GP.predict_value(data)
            pred_robot, var_robot = robot_model.predict_value(data)      
        elif self.world.dim == 3:
            data = np.vstack([x1, x2, time * np.ones(NTEST)]).T   
            pred_world, var_world = self.world.models[time].predict_value(data)
            pred_robot, var_robot = robot_model.predict_value(data)      

        # Get the NHOTSPOT most "valuable" points
        order = np.argsort(pred_world, axis = None)
        pred_world = pred_world[order[0:NHS]]
        pred_robot = pred_robot[order[0:NHS]]
        
        return ((pred_world - pred_robot) ** 2).mean()
        
    def MSE(self, time, robot_model, NTEST = 100):
        ''' Compute the MSE on a set of test points, randomly distributed throughout the environment'''
        x1 = np.random.random_sample((NTEST, 1)) * (self.world.x1max - self.world.x1min) + self.world.x1min
        x2 = np.random.random_sample((NTEST, 1)) * (self.world.x2max - self.world.x2min) + self.world.x2min
        x1 = x1.reshape((NTEST,))
        x2 = x2.reshape((NTEST,))
        if self.world.dim == 2:
            data = np.vstack([x1, x2]).T   
            pred_world, var_world = self.world.GP.predict_value(data)
            pred_robot, var_robot = robot_model.predict_value(data)      
        elif self.world.dim == 3:
            data = np.vstack([x1, x2, time * np.ones(NTEST)]).T   
            pred_world, var_world = self.world.models[time].predict_value(data)
            pred_robot, var_robot = robot_model.predict_value(data)      
        
        return ((pred_world - pred_robot) ** 2).mean()
    
    ''' Helper functions '''

    def update_metrics(self, t, robot_model, robot_loc, selected_path, value=None, max_loc=None, max_val=None, params=None, dist=0):
        ''' Function to update avaliable metrics'''    

        # Update with this timestamps max value and location (only spatial location)
        if self.world.dim == 3:
            self.max_loc = self.world.models[t].xvals[np.argmax(self.world.models[t].zvals), 0:-1]
            self.max_val = np.max(self.world.models[t].zvals)

        self.metrics['planning_iteration'][t] = t
        self.metrics['aquisition_function'][t] = value
        self.metrics['max_loc_error'][t], self.metrics['max_val_error'][t] = self.max_error(max_loc, max_val)

        if params[2] is None:
            for i in range(0,self.num_stars):
                self.metrics['star_obs_'+str(i)][t] = -1.
                self.metrics['star_obs_loc_x_'+str(i)][t] = -1.
                self.metrics['star_obs_loc_y_'+str(i)][t] = -1.
        else:
            for i, s in enumerate(params[2]):
                self.metrics['star_obs_'+str(i)][t] = s
                self.metrics['star_obs_loc_x_'+str(i)][t] = params[3][i][0]
                self.metrics['star_obs_loc_y_'+str(i)][t] = params[3][i][1]

        self.metrics['info_gain_reward'][t] = self.info_gain_reward(t, selected_path, robot_model)
        self.metrics['MSE'][t] = self.MSE(t, robot_model, NTEST = 200)
        self.metrics['hotspot_error'][t] = self.hotspot_error(t, robot_model, NTEST = 200, NHS = 100)

        self.metrics['current_highest_obs'][t] = params[0]
        self.metrics['current_highest_obs_loc_x'][t] = params[1][0]
        self.metrics['current_highest_obs_loc_y'][t] = params[1][1]
        self.metrics['robot_location_x'][t] = robot_loc[0]
        self.metrics['robot_location_y'][t] = robot_loc[1]
        self.metrics['robot_location_a'][t] = robot_loc[2]
        self.metrics['robot_samples'][t] = selected_path

        self.metrics['distance_traveled'][t] = dist
    
    def plot_metrics(self):
        ''' Plots the performance metrics computed over the course of a info'''
        time = np.array(self.metrics['planning_iteration'].values())
        
        ''' Metrics that require a ground truth global model to compute'''        
        info_gain = np.cumsum(np.array(self.metrics['info_gain_reward'].values()))        
        aqu_fun = np.cumsum(np.array(self.metrics['aquisition_function'].values()))
        MSE = np.array(self.metrics['MSE'].values())
        hotspot_error = np.array(self.metrics['hotspot_error'].values())

        max_loc_error = np.array(self.metrics['max_loc_error'].values())
        max_val_error = np.array(self.metrics['max_val_error'].values())

        current_highest_obs = np.array(self.metrics['current_highest_obs'].values())
        current_highest_obs_loc_x = np.array(self.metrics['current_highest_obs_loc_x'].values())
        current_highest_obs_loc_y = np.array(self.metrics['current_highest_obs_loc_y'].values())
        robot_location_x = np.array(self.metrics['robot_location_x'].values())
        robot_location_y = np.array(self.metrics['robot_location_y'].values())
        robot_location_a = np.array(self.metrics['robot_location_a'].values())
        robot_samples = np.array(self.metrics['robot_samples'].values())
        
        star_obs = []
        star_obs_loc_x = []
        star_obs_loc_y = []

        for i in range(0,self.num_stars):
            star_obs.append(np.array(self.metrics['star_obs_'+str(i)].values()))
            star_obs_loc_x.append(np.array(self.metrics['star_obs_loc_x_'+str(i)].values()))
            star_obs_loc_y.append(np.array(self.metrics['star_obs_loc_y_'+str(i)].values()))

        distance = np.array(self.metrics['distance_traveled'].values())

        if not os.path.exists('./figures/' + str(self.reward_function)):
            os.makedirs('./figures/' + str(self.reward_function))
        ''' Save the relevent metrics as csv files '''
        np.savetxt('./figures/' + self.reward_function + '/metrics.csv', \
            (time.T, info_gain.T, aqu_fun.T, MSE.T, hotspot_error.T, max_loc_error.T, \
            max_val_error.T, current_highest_obs.T, current_highest_obs_loc_x.T, current_highest_obs_loc_y.T, \
            robot_location_x.T, robot_location_y.T, robot_location_a.T, \
            distance.T))

        for i in range(0, self.num_stars):
            f = open('./figures/'+self.reward_function + '/stars.csv', "a")
            np.savetxt(f, (star_obs[i].T, star_obs_loc_x[i].T, star_obs_loc_y[i].T))
            f.close()

        np.save('./figures/'+self.reward_function + '/samples', robot_samples)

        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax.set_title('Accumulated Information Gain')                             
        # plt.plot(time, info_gain, 'k')        
        # fig.savefig('./figures/' + self.reward_function + '/information_gain.png')
        
        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax.set_title('Accumulated Aquisition Function')             
        # plt.plot(time, aqu_fun, 'g')
        # fig.savefig('./figures/' + self.reward_function + '/aqu_fun.png')
        
        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax.set_title('Max Location Error')                             
        # plt.plot(time, max_loc_error, 'k')        
        # fig.savefig('./figures/' + self.reward_function + '/error_location.png')
        
        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax.set_title('Max Value Error')                             
        # plt.plot(time, max_val_error, 'k')        
        # fig.savefig('./figures/' + self.reward_function + '/error_value.png')
        
        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax.set_title('Map MSE at 100 Random Test Points')                             
        # plt.plot(time, MSE, 'r')  
        # fig.savefig('./figures/' + self.reward_function + '/mse.png')
        
        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax.set_title('Map Hotspot Error at 100 Random Test Points')                             
        # plt.plot(time, hotspot_error, 'r')  
        # fig.savefig('./figures/' + self.reward_function + '/hotspot_mse.png')
        
        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax.set_title('Average sample loc distance to Maximizer')                             
        # plt.plot(time, sample_regret_loc, 'r')
        # fig.savefig('./figures/' + self.reward_function + '/sample_regret_loc.png')
  
        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax.set_title('Average sample val distance to Maximizer')
        # plt.plot(time, sample_regret_val, 'r')  
        # fig.savefig('./figures/' + self.reward_function + '/sample_regret_val.png')

        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax.set_title('Distance Traveled in Time')
        # plt.plot(time, distance, 'r')  
        # fig.savefig('./figures/' + self.reward_function + '/distance_traveled.png')
        
        # #plt.show() 
        # plt.close()
