# !/usr/bin/python

'''
This library alows access to the evaluation metrics and criteria used to assess the PLUMES framework.
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
import GPy as GPy
import time
from itertools import chain
import pdb
import logging
logger = logging.getLogger('robot')

import aq_library as aqlib

class Evaluation:
    ''' The Evaluation class, which includes the ground truth world model and a selection of reward criteria.
    
    Inputs:
        world (Environment object): an environment object that represents the ground truth environment
        f_rew (string): the reward function. One of {hotspot_info, mean, info_gain, mes, exp_improve} 
    '''
    def __init__(self, world, reward_function = 'mean'):
        ''' Initialize the evaluation module and select reward function'''
        self.world = world
        self.max_val = np.max(world.GP.zvals)
        self.max_loc = world.GP.xvals[np.argmax(world.GP.zvals), :]
        self.reward_function = reward_function

        print "World max value", self.max_val, "at location", self.max_loc
        logger.info("World max value {} at location {}".format(self.max_val, self.max_loc))
        
        self.metrics = {'aquisition_function': {},
                        'mean_reward': {}, 
                        'info_gain_reward': {},                         
                        'hotspot_info_reward': {}, 
                        'MSE': {},                         
                        'hotspot_error': {},                         
                        'instant_regret': {},
                        'max_val_regret': {},
                        'regret_bound': {},
                        'simple_regret': {},
                        'sample_regret_loc': {},
                        'sample_regret_val': {},
                        'max_loc_error': {},
                        'max_val_error': {},
                        'current_highest_obs': {},
                        'current_highest_obs_loc_x': {},
                        'current_highest_obs_loc_y': {},
                        'star_obs_0': {},
                        'star_obs_1': {},
                        'star_obs_loc_x_0': {},
                        'star_obs_loc_x_1': {},
                        'star_obs_loc_y_0': {},
                        'star_obs_loc_y_1': {},
                        'robot_location_x': {},
                        'robot_location_y': {},
                        'robot_location_a': {},
                        'distance_traveled': {},
                       }
        self.reward_function = reward_function
        
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
        elif reward_function == 'exp_improve':
            self.f_aqu = aqlib.exp_improvement
            self.f_rew = self.mean_reward
        else:
            raise ValueError('Only \'mean\' and \'hotspot_info\' and \'info_gain\' and \' mew\' and \'exp_improve\' reward functions currently supported.')    
    
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
        queries = np.vstack([x1, x2]).T   
        
        mu, var = self.world.GP.predict_value(queries)
        return np.sum(mu)   

    def hotspot_info_reward(self, time, xvals, robot_model):
        ''' The reward information gathered plus the exploitation value gathered'''    
        LAMBDA = 1.0# TOOD: should depend on time
        data = np.array(xvals)
        x1 = data[:,0]
        x2 = data[:,1]
        queries = np.vstack([x1, x2]).T   
        
        mu, var = self.world.GP.predict_value(queries)    
        return self.info_gain_reward(time, xvals, robot_model) + LAMBDA * np.sum(mu)
    
    def info_gain_reward(self, time, xvals, robot_model):
        ''' The information reward gathered '''
        return aqlib.info_gain(time, xvals, robot_model)
    
    def inst_regret(self, t, all_paths, selected_path, robot_model, param = None):
        ''' The instantaneous Kapoor regret of a selected path, according to the specified reward function
        Input:
            all_paths: the set of all avalaible paths to the robot at time t
            selected path: the path selected by the robot at time t 
            robot_model (GP Model)
        '''

        value_omni = {}        
        for path, points in all_paths.items():           
            if param is None:
                value_omni[path] =  self.f_rew(time = t, xvals = points, robot_model = robot_model)  
            else:
                value_omni[path] =  aqlib.mves(time = t, xvals = points, robot_model = robot_model, param = (self.max_val).reshape(1,1))  

        value_max = value_omni[max(value_omni, key = value_omni.get)]
        if param is None:
            value_selected = self.f_rew(time = t, xvals = selected_path, robot_model = robot_model)
        else:
            value_selected =  aqlib.mves(time = t, xvals = selected_path, robot_model = robot_model, param = (self.max_val).reshape(1,1))  
        return value_max - value_selected
    
    def simple_regret(self, xvals):
        ''' The simple regret of a selected trajecotry
        Input:
            max_loc (nparray 1 x 2)
        '''
        error = 0.0
        for point in xvals:
            error += np.linalg.norm(np.array(point[0:-1]) -  self.max_loc)
        error /= float(len(xvals))

        return error

    def sample_regret(self, robot_model):
        if robot_model.xvals is None:
            return 0., 0.

        global_max_val = np.reshape(np.array(self.max_val), (1,1))
        global_max_loc = np.reshape(np.array(self.max_loc), (1,2))
        avg_loc_dist = sp.spatial.distance.cdist(global_max_loc, robot_model.xvals)
        avg_val_dist = sp.spatial.distance.cdist(global_max_val, robot_model.zvals)
        return np.mean(avg_loc_dist), np.mean(avg_val_dist)
    
    def max_error(self, max_loc, max_val):
        ''' The error of the current best guess for the global maximizer
        Input:
            max_loc (nparray 1 x 2)
            max_val (float)
        '''
        return np.linalg.norm(max_loc - self.max_loc), np.linalg.norm(max_val - self.max_val)

    def hotspot_error(self, robot_model, NTEST = 100, NHS = 100):
        ''' Compute the hotspot error on a set of test points, randomly distributed throughout the environment'''
        x1 = np.random.random_sample((NTEST, 1)) * (self.world.x1max - self.world.x1min) + self.world.x1min
        x2 = np.random.random_sample((NTEST, 1)) * (self.world.x2max - self.world.x2min) + self.world.x2min
        data = np.hstack((x1, x2))
        
        pred_world, var_world = self.world.GP.predict_value(data)
        pred_robot, var_robot = robot_model.predict_value(data)      

        # Get the NHOTSPOT most "valuable" points
        #print pred_world
        order = np.argsort(pred_world, axis = None)
        pred_world = pred_world[order[0:NHS]]
        pred_robot = pred_robot[order[0:NHS]]

        #print pred_world
        #print pred_robot
        #print order
        
        return ((pred_world - pred_robot) ** 2).mean()
    
    def regret_bound(self, t, T):
        pass
        
    def MSE(self, robot_model, NTEST = 100):
        ''' Compute the MSE on a set of test points, randomly distributed throughout the environment'''
        x1 = np.random.random_sample((NTEST, 1)) * (self.world.x1max - self.world.x1min) + self.world.x1min
        x2 = np.random.random_sample((NTEST, 1)) * (self.world.x2max - self.world.x2min) + self.world.x2min
        data = np.hstack((x1, x2))
        
        pred_world, var_world = self.world.GP.predict_value(data)
        pred_robot, var_robot = robot_model.predict_value(data)      
        
        return ((pred_world - pred_robot) ** 2).mean()
    
    ''' Helper functions '''

    def update_metrics(self, t, robot_model, all_paths, selected_path, value = None, max_loc = None, max_val = None, params = None, dist = 0):
        ''' Function to update avaliable metrics'''    
        #self.metrics['hotspot_info_reward'][t] = self.hotspot_info_reward(t, selected_path, robot_model, max_val)
        #self.metrics['mean_reward'][t] = self.mean_reward(t, selected_path, robot_model)
        self.metrics['aquisition_function'][t] = value

        self.metrics['simple_regret'][t] = self.simple_regret(selected_path)
        self.metrics['sample_regret_loc'][t], self.metrics['sample_regret_val'][t] = self.sample_regret(robot_model)
        self.metrics['max_loc_error'][t], self.metrics['max_val_error'][t] = self.max_error(max_loc, max_val)
        
        self.metrics['instant_regret'][t] = self.inst_regret(t, all_paths, selected_path, robot_model)
        self.metrics['max_val_regret'][t] = self.inst_regret(t, all_paths, selected_path, robot_model, param = 'info_regret')

        self.metrics['star_obs_0'][t] = params[2][0]
        self.metrics['star_obs_1'][t] = params[2][1]
        self.metrics['star_obs_loc_x_0'][t] = params[3][0][0]
        self.metrics['star_obs_loc_x_1'][t] = params[3][1][0]
        self.metrics['star_obs_loc_y_0'][t] = params[3][0][1]
        self.metrics['star_obs_loc_y_1'][t] = params[3][1][1]

        self.metrics['info_gain_reward'][t] = self.info_gain_reward(t, selected_path, robot_model)
        self.metrics['MSE'][t] = self.MSE(robot_model, NTEST = 200)
        self.metrics['hotspot_error'][t] = self.hotspot_error(robot_model, NTEST = 200, NHS = 100)

        self.metrics['current_highest_obs'][t] = params[0]
        self.metrics['current_highest_obs_loc_x'][t] = params[1][0]
        self.metrics['current_highest_obs_loc_y'][t] = params[1][1]
        self.metrics['robot_location_x'][t] = selected_path[0][0]
        self.metrics['robot_location_y'][t] = selected_path[0][1]
        self.metrics['robot_location_a'][t] = selected_path[0][2]

        self.metrics['distance_traveled'][t] = dist
    
    def plot_metrics(self):
        ''' Plots the performance metrics computed over the course of a info'''
        # Asumme that all metrics have the same time as MSE; not necessary
        time = np.array(self.metrics['MSE'].keys())
        
        ''' Metrics that require a ground truth global model to compute'''        
        info_gain = np.cumsum(np.array(self.metrics['info_gain_reward'].values()))        
        aqu_fun = np.cumsum(np.array(self.metrics['aquisition_function'].values()))
        MSE = np.array(self.metrics['MSE'].values())
        hotspot_error = np.array(self.metrics['hotspot_error'].values())
        
        regret = np.cumsum(np.array(self.metrics['instant_regret'].values()))
        info_regret = np.cumsum(np.array(self.metrics['max_val_regret'].values()))

        max_loc_error = np.array(self.metrics['max_loc_error'].values())
        max_val_error = np.array(self.metrics['max_val_error'].values())
        simple_regret = np.array(self.metrics['simple_regret'].values())

        sample_regret_loc = np.array(self.metrics['sample_regret_loc'].values())
        sample_regret_val = np.array(self.metrics['sample_regret_val'].values())

        current_highest_obs = np.array(self.metrics['current_highest_obs'].values())
        current_highest_obs_loc_x = np.array(self.metrics['current_highest_obs_loc_x'].values())
        current_highest_obs_loc_y = np.array(self.metrics['current_highest_obs_loc_y'].values())
        robot_location_x = np.array(self.metrics['robot_location_x'].values())
        robot_location_y = np.array(self.metrics['robot_location_y'].values())
        robot_location_a = np.array(self.metrics['robot_location_a'].values())
        star_obs_0 = np.array(self.metrics['star_obs_0'].values())
        star_obs_1 = np.array(self.metrics['star_obs_1'].values())
        star_obs_loc_x_0 = np.array(self.metrics['star_obs_loc_x_0'].values())
        star_obs_loc_x_1 = np.array(self.metrics['star_obs_loc_x_1'].values())
        star_obs_loc_y_0 = np.array(self.metrics['star_obs_loc_y_0'].values())
        star_obs_loc_y_1 = np.array(self.metrics['star_obs_loc_y_1'].values())

        distance = np.array(self.metrics['distance_traveled'].values())
        # star_obs_loc = np.array(self.metrics['star_obs_loc'].values())

        #mean = np.cumsum(np.array(self.metrics['mean_reward'].values()))
        #hotspot_info = np.cumsum(np.array(self.metrics['hotspot_info_reward'].values()))


        if not os.path.exists('./figures/' + str(self.reward_function)):
            os.makedirs('./figures/' + str(self.reward_function))
        ''' Save the relevent metrics as csv files '''
        np.savetxt('./figures/' + self.reward_function + '/metrics.csv', \
            (time.T, info_gain.T, aqu_fun.T, MSE.T, hotspot_error.T, max_loc_error.T, \
            max_val_error.T, simple_regret.T,  sample_regret_loc.T, sample_regret_val.T, \
            regret.T, info_regret.T, current_highest_obs.T, current_highest_obs_loc_x.T,current_highest_obs_loc_y.T, \
            robot_location_x.T, robot_location_y.T, robot_location_a.T, \
            star_obs_0.T, star_obs_loc_x_0.T, star_obs_loc_y_0.T, \
            star_obs_1.T, star_obs_loc_x_1.T, star_obs_loc_y_1.T, distance.T))
        #np.savetxt('./figures/' + self.reward_function + '/aqu_fun.csv', aqu_fun)
        #np.savetxt('./figures/' + self.reward_function + '/MSE.csv', MSE)
        #np.savetxt('./figures/' + self.reward_function + '/hotspot_MSE.csv', hotspot_error)
        #np.savetxt('./figures/' + self.reward_function + '/max_loc_error.csv', max_loc_error)
        #np.savetxt('./figures/' + self.reward_function + '/max_val_error.csv', max_val_error)
        #np.savetxt('./figures/' + self.reward_function + '/simple_regret.csv', simple_regret)
        
        
        #fig, ax = plt.subplots(figsize=(8, 6))
        #ax.set_title('Accumulated Mean Reward')                     
        #plt.plot(time, mean, 'b')      
        
        #fig, ax = plt.subplots(figsize=(8, 6))
        #ax.set_title('Accumulated Hotspot Information Gain Reward')                             
        #plt.plot(time, hotspot_info, 'r')          
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Average Regret w.r.t. ' + self.reward_function + ' Reward')                     
        plt.plot(time, regret/time, 'b')
        fig.savefig('./figures/' + self.reward_function + '/snapping_regret.png')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Average Info Regret w.r.t. ' + self.reward_function + ' Reward')                     
        plt.plot(time, info_regret/time, 'b')
        fig.savefig('./figures/' + self.reward_function + '/snapping_info_regret.png')

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Accumulated Information Gain')                             
        plt.plot(time, info_gain, 'k')        
        fig.savefig('./figures/' + self.reward_function + '/information_gain.png')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Accumulated Aquisition Function')             
        plt.plot(time, aqu_fun, 'g')
        fig.savefig('./figures/' + self.reward_function + '/aqu_fun.png')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Max Location Error')                             
        plt.plot(time, max_loc_error, 'k')        
        fig.savefig('./figures/' + self.reward_function + '/error_location.png')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Max Value Error')                             
        plt.plot(time, max_val_error, 'k')        
        fig.savefig('./figures/' + self.reward_function + '/error_value.png')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Simple Regret w.r.t. Global Maximizer')                     
        plt.plot(time, simple_regret, 'b')        
        fig.savefig('./figures/' + self.reward_function + '/simple_regret.png')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Map MSE at 100 Random Test Points')                             
        plt.plot(time, MSE, 'r')  
        fig.savefig('./figures/' + self.reward_function + '/mse.png')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Map Hotspot Error at 100 Random Test Points')                             
        plt.plot(time, hotspot_error, 'r')  
        fig.savefig('./figures/' + self.reward_function + '/hotspot_mse.png')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Average sample loc distance to Maximizer')                             
        plt.plot(time, sample_regret_loc, 'r')
        fig.savefig('./figures/' + self.reward_function + '/sample_regret_loc.png')
  
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Average sample val distance to Maximizer')
        plt.plot(time, sample_regret_val, 'r')  
        fig.savefig('./figures/' + self.reward_function + '/sample_regret_val.png')

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Distance Traveled in Time')
        plt.plot(time, distance, 'r')  
        fig.savefig('./figures/' + self.reward_function + '/distance_traveled.png')
        
        #plt.show() 
        plt.close()
