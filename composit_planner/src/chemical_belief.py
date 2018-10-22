#!/usr/bin/python

# Copyright 2018 Massachusetts Institute of Technology
# System includes
import numpy as np
import scipy as sp
import math
import os
import threading

# ROS includes
import rospy
from std_msgs.msg import *
from composit_planner.srv import *
from composit_planner.msg import *

# GPy and GP includes
import GPy as GPy
from gpmodel_library import GPModel, OnlineGPModel
from scipy.stats import multivariate_normal

import aq_library as aq

'''
This is a ROS node that performs the following task(s): 
    * publish the sample maxima of a current belief map (SRV)
    * publish the mean/var prediction at a specific point (SRV)
    * subscribes to the node and updated belief model

    TODO:
        * Port ability to initialize from a prior dataset
        * Think about how to queue measurements/when to update kernel
'''

class ChemicalBelief:
    '''The ChemicalBelief class, which represents a rectangular Gaussian world.
            prior_dataset (tuple of nparrays) a tuple (xvals, zvals), where xvals is a Npoint x 2 nparray of type float and zvals is a Npoint x 1 nparray of type float
    ''' 
    def __init__(self):
        self.current_max = -float("inf")
        self.data_queue = list()
        self._maxima = None

        self.data_lock = threading.Lock()

        ''' Get ROS parameters '''
        # The kernel hyperparmeters for the robot's GP model 
        self.variance = float(rospy.get_param('model_variance','100'))
        self.lengthscale = float(rospy.get_param('model_lengthscale','0.1'))
        self.noise = float(rospy.get_param('model_noise', '0.0001'))
        
        # This range parameter will be shared by both the world and the robot
        self.ranges = rospy.get_param('ranges', [0, 10, 0, 10]) 

        # Initialize the robot's GP model with the initial kernel parameters
        self.GP = OnlineGPModel(ranges = self.ranges, lengthscale = self.lengthscale, variance = self.variance, noise = self.noise)

        # Define ROS service
        self.srv_replan = rospy.Service('update_model', RequestReplan, self.update_model)
        self.srv_maxima= rospy.Service('pred_value', GetValue, self.predict_value)
       
        # Subscribe to the chem data topic
        self.data = rospy.Subscriber("chem_data", ChemicalSample, self.get_sensordata)
        
        rospy.spin()

    @property
    def maxima(self):
        if self._maxima is None:
            max_vals, max_locs, func = aq.sample_max_vals(self.GP) 
            self._maxima = (max_vals, max_locs, func)
        return self._maxima
    
    def update_model(self, _):
        ''' Adds all data currently in the data queue into the GP model and clears the data queue. Threadsafe. 
        Input:
            None
        Output: 
            Boolean success service response''' 
        try:
            # Aquire the data lock
            self.data_lock.acquire()

            # Add all current observations in the data queue to the current model
            NUM_PTS = len(self.data_queue)
            zobs = np.array([msg.data for msg in self.data_queue]).reshape(NUM_PTS, 1)
            xobs = np.array([[msg.loc.x, msg.loc.y] for msg in self.data_queue]).reshape(NUM_PTS, 2)
            self.GP.add_data(xobs, zobs)
        
            # Update the current best max for EI
            for z, x in zip (zobs, xobs):
                if z[0] > self.current_max:
                    self.current_max = z[0]
                    self.current_max_loc = [x[0],x[1]]

            # Delete data from the data_queue
            del self.data_queue[:] #in Python3, would be self.data_queue.clear()
            self._maxima = None

            # Release the data lock
            self.data_lock.release()

            return RequestReplanResponse(True)
        except:
            return RequestReplanResponse(False)

    def predict_value(self, req):
        ''' Samples a set of K maxima from the current GP belief.  
        Input:
            None
        Output: 
            Boolean success service response''' 

        xvals = np.array([req.pose.x, req.pose.y]).reshape(1,2)
        aq_func = req.aq_func 
        if aq_func == 'ei':
            value = aq.exp_improvement(time = 0, xvals = xvals, robot_model = self.GP, param = self.current_max)
        elif aq_func == 'ucb':
            value = aq.mean_ucb(time = 0, xvals = xvals, robot_model = self.GP, param = None)
        elif aq_func == 'mes':
            value = aq.mves(time = 0, xvals = xvals, robot_model = self.GP, param = self.maxima)
        elif aq_func == 'ig':
            value = aq.info_gain(time = 0, xvals = xvals, robot_model = self.GP, param = None)
        else:
            print aq_func
            raise ValueError('Aqusition function must be one of ei, ucb, ig, or mes')

        return GetValueResponse(value)

    def get_sensordata(self, msg):
        ''' Gather noisy samples of the environment and updates the robot's GP model.
        Input: 
            xobs (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2 '''

        self.data_lock.acquire()
        self.data_queue.append(msg)
        self.data_lock.release()    

def main():
	#initialize node
	rospy.init_node('chemical_belief')
	try:
		ChemicalBelief()
	except rospy.ROSInterruptException:
		pass
if __name__ == '__main__':
	main()
