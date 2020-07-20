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
import aq_library as aqlib
# import glog as log
import logging as log
# import gpmodel_library as gp_lib
# from continuous_traj import continuous_traj

class Evaluation:
    def __init__(self, world, reward_function = 'mean'):
        self.world = world
        
        self.metrics = {'aquisition_function': {},
                        'mean_reward': {}, 
                        'info_gain_reward': {},                         
                        'hotspot_info_reward': {}, 
                        'MSE': {},                         
                        'instant_regret': {},   
                        'mes_reward_robot': {},                     
                       }
        
        self.reward_function = reward_function
        
        if reward_function == 'hotspot_info':
            self.f_rew = self.hotspot_info_reward
            self.f_aqu = hotspot_info_UCB
        elif reward_function == 'mean':
            self.f_rew = self.mean_reward
            self.f_aqu = mean_UCB      
        elif reward_function == 'info_gain':
            self.f_rew = self.info_gain_reward
            self.f_aqu = info_gain             
        elif reward_function == 'mes':
            self.f_aqu = aqlib.mves
            self.f_rew = self.mean_reward 
        else:
            raise ValueError('Only \'mean\' and \'hotspot_info\' reward functions currently supported.')    
    
    ''' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                Reward Functions - should have the form:
    def reward(time, xvals), where:
    * time (int): the current timestep of planning
    * xvals (list of float tuples): representing a path i.e. [(3.0, 4.0), (5.6, 7.2), ... ])
    * robot_model (GPModel)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
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
        LAMBDA = 0.5
        data = np.array(xvals)
        x1 = data[:,0]
        x2 = data[:,1]
        queries = np.vstack([x1, x2]).T   
        
        mu, var = self.world.GP.predict_value(queries)    
        return self.info_gain_reward(time, xvals, robot_model) + LAMBDA * np.sum(mu)
    
    def info_gain_reward(self, time, xvals, robot_model):
        ''' The information reward gathered '''
        return info_gain(time, xvals, robot_model)
    
    ''' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                               End Reward Functions
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''        
    def inst_regret(self, t, all_paths, selected_path, robot_model):
        ''' The instantaneous Kapoor regret of a selected path, according to the specified reward function
        Input:
        * all_paths: the set of all avalaible paths to the robot at time t
        * selected path: the path selected by the robot at time t '''
        
        value_omni = {}        
        for path, points in all_paths.items():           
            value_omni[path] =  self.f_rew(time = t, xvals = points, robot_model = robot_model)  
        value_max = value_omni[max(value_omni, key = value_omni.get)]
        
        value_selected = self.f_rew(time = t, xvals = selected_path, robot_model = robot_model)

        #assert(value_max - value_selected >= 0)
        return value_max - value_selected
        
    def MSE(self, robot_model, NTEST = 100):
        ''' Compute the MSE on a set of test points, randomly distributed throughout the environment'''
        np.random.seed(0)
        x1 = np.random.random_sample((NTEST, 1)) * (self.world.x1max - self.world.x1min) + self.world.x1min
        x2 = np.random.random_sample((NTEST, 1)) * (self.world.x2max - self.world.x2min) + self.world.x2min
        data = np.hstack((x1, x2))
        
        pred_world, var_world = self.world.GP.predict_value(data)
        pred_robot, var_robot = robot_model.predict_value(data)      
        
        return ((pred_world - pred_robot) ** 2).mean()
    
    def update_metrics(self, t, robot_model, all_paths, selected_path):
        ''' Function to update avaliable metrics'''    
        # Compute aquisition function
        if(self.f_aqu == aqlib.mves):
            self.metrics['aquisition_function'][t] = self.f_aqu(t, selected_path, robot_model, [None])
        else:
            self.metrics['aquisition_function'][t] = self.f_aqu(t, selected_path, robot_model)
        
        # Compute reward functions
        self.metrics['mean_reward'][t] = self.mean_reward(t, selected_path, robot_model)
        self.metrics['info_gain_reward'][t] = self.info_gain_reward(t, selected_path, robot_model)
        self.metrics['hotspot_info_reward'][t] = self.hotspot_info_reward(t, selected_path, robot_model)
        self.metrics['mes_reward_robot'][t] = aqlib.mves(t, selected_path, robot_model, [None])
        # Compute other performance metrics
        self.metrics['MSE'][t] = self.MSE(robot_model, NTEST = 25)
        # self.metrics['instant_regret'][t] = self.inst_regret(t, all_paths, selected_path, robot_model)
    
    def save_metric(self):
        time = np.array(self.metrics['MSE'].keys())
        
        ''' Metrics that require a ground truth global model to compute'''        
        MSE = np.array(self.metrics['MSE'].values())
        regret = np.cumsum(np.array(self.metrics['instant_regret'].values()))
        mean = np.cumsum(np.array(self.metrics['mean_reward'].values()))
        hotspot_info = np.cumsum(np.array(self.metrics['hotspot_info_reward'].values()))
        
        ''' Metrics that the robot can compute online '''
        info_gain = np.cumsum(np.array(self.metrics['info_gain_reward'].values()))        
        UCB = np.cumsum(np.array(self.metrics['aquisition_function'].values()))
        
        return MSE, regret, mean, hotspot_info, info_gain, UCB
        
    def plot_metrics(self, iteration, range_max, grad_step):
        # Asumme that all metrics have the same time as MSE; not necessary
        time = np.array(self.metrics['MSE'].keys())
        
        ''' Metrics that require a ground truth global model to compute'''        
        MSE = np.array(self.metrics['MSE'].values())
        regret = np.cumsum(np.array(self.metrics['instant_regret'].values()))
        mean = np.cumsum(np.array(self.metrics['mean_reward'].values()))
        hotspot_info = np.cumsum(np.array(self.metrics['hotspot_info_reward'].values()))
        
        ''' Metrics that the robot can compute online '''
        info_gain = np.cumsum(np.array(self.metrics['info_gain_reward'].values()))        
        UCB = np.cumsum(np.array(self.metrics['aquisition_function'].values()))
        

        if not os.path.exists('./result/' + str(self.reward_function)):
            os.makedirs('./result/' + str(self.reward_function))
        ''' Save the relevent metrics as csv files '''
        np.savetxt('./result/' + self.reward_function + '/metrics_grad_step_' + str(grad_step)+ 'range_max_' + str(range_max) \
            + ' iter_' + str(iteration) +'_time' + '.txt', time.T, fmt='%s')
        np.savetxt('./result/' + self.reward_function + '/metrics_grad_step_' + str(grad_step)+ 'range_max_' + str(range_max) \
            + ' iter_' + str(iteration) +'_info_gain' + '.txt', info_gain.T, fmt='%s')
        np.savetxt('./result/' + self.reward_function + '/metrics_grad_step_' + str(grad_step)+ 'range_max_' + str(range_max) \
            + ' iter_' + str(iteration) +'_MSE' + '.txt', MSE.T, fmt='%s')
        np.savetxt('./result/' + self.reward_function + '/metrics_grad_step_' + str(grad_step)+ 'range_max_' + str(range_max) \
            + ' iter_' + str(iteration) +'_hotspot_info' + '.txt', hotspot_info.T, fmt='%s')
        np.savetxt('./result/' + self.reward_function + '/metrics_grad_step_' + str(grad_step)+ 'range_max_' + str(range_max) \
            + ' iter_' + str(iteration) +'_UCB' + '.txt', UCB.T, fmt='%s')
        np.savetxt('./result/' + self.reward_function + '/metrics_grad_step_' + str(grad_step)+ 'range_max_' + str(range_max) \
            + ' iter_' + str(iteration) +'_mean' + '.txt', mean.T, fmt='%s')
# , info_gain.T, MSE.T, hotspot_info.T,UCB.T, regret.T, mean.T )
        # for i in range(0, self.num_stars):
        #     f = open('./figures/'+self.reward_function + '/stars.csv', "a")
        #     np.savetxt(f, (star_obs[i].T, star_obs_loc_x[i].T, star_obs_loc_y[i].T))
        #     f.close()


        # fig, ax = plt.subplots(figsize=(4, 3))
        # ax.set_title('Accumulated UCB Aquisition Function')             
        # plt.plot(time, UCB, 'g')
        # fig.savefig('./figures/gradient/' + self.reward_function + '/UCB.png')

        # fig, ax = plt.subplots(figsize=(4, 3))
        # ax.set_title('Accumulated Information Gain')                             
        # plt.plot(time, info_gain, 'k')        
        # fig.savefig('./figures/gradient/' + self.reward_function + '/Accumul_Info_Gain.png')

        # fig, ax = plt.subplots(figsize=(4, 3))
        # ax.set_title('Accumulated Mean Reward')                     
        # plt.plot(time, mean, 'b')      
        # fig.savefig('./figures/gradient/' + self.reward_function + '/Accumul_Mean_reward.png')


        # fig, ax = plt.subplots(figsize=(4, 3))
        # ax.set_title('Accumulated Hotspot Information Gain Reward')                             
        # plt.plot(time, hotspot_info, 'r')          
        # fig.savefig('./figures/gradient/' + self.reward_function + '/Accumul_Hotspot_Info_Gain.png')

        # # fig, ax = plt.subplots(figsize=(4, 3))
        # # ax.set_title('Average Regret w.r.t. ' + self.reward_function + ' Reward')                     
        # # plt.plot(time, regret/time, 'b')        
        # # fig.savefig('./figures/' + self.reward_function + '/Regret_Time.png')

        # fig, ax = plt.subplots(figsize=(4, 3))
        # ax.set_title('Map MSE at 100 Random Test Points')                             
        # plt.plot(time, MSE, 'r')  
        # fig.savefig('./figures/gradient/' + self.reward_function + '/Map_MSE.png')

        # plt.show()          
    
                             
'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                    Aquisition Functions - should have the form:
    def alpha(time, xvals, robot_model), where:
    * time (int): the current timestep of planning
    * xvals (list of float tuples): representing a path i.e. [(3.0, 4.0), (5.6, 7.2), ... ])
    * robot_model (GPModel object): the robot's current model of the environment
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% '''

def info_gain(time, xvals, robot_model):
    ''' Compute the information gain of a set of potential sample locations with respect to the underlying fucntion
        conditioned or previous samples xobs'''        
    data = np.array(xvals)
    x1 = data[:,0]
    x2 = data[:,1]
    queries = np.vstack([x1, x2]).T   
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

    
def mean_UCB(time, xvals, robot_model):
    ''' Computes the UCB for a set of points along a trajectory '''
    data = np.array(xvals)
    x1 = data[:,0]
    x2 = data[:,1]
    queries = np.vstack([x1, x2]).T   
                              
    # The GPy interface can predict mean and variance at an array of points; this will be an overestimate
    mu, var = robot_model.predict_value(queries)
    
    delta = 0.9
    d = 20
    pit = np.pi**2 * (time + 1)**2 / 6.
    beta_t = 2 * np.log(d * pit / delta)

    return np.sum(mu) + np.sqrt(beta_t) * np.sum(np.fabs(var))

def hotspot_info_UCB(time, xvals, robot_model):
    ''' The reward information gathered plus the exploitation value gathered'''
    data = np.array(xvals)
    x1 = data[:,0]
    x2 = data[:,1]
    queries = np.vstack([x1, x2]).T   
                              
    LAMBDA = 0.5
    mu, var = robot_model.predict_value(queries)
    
    delta = 0.9
    d = 20
    pit = np.pi**2 * (time + 1)**2 / 6.
    beta_t = 2 * np.log(d * pit / delta)

    return info_gain(time, xvals, robot_model) + LAMBDA * np.sum(mu) + np.sqrt(beta_t) * np.sum(np.fabs(var))