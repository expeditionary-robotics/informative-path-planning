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
from nav_msgs.msg import Odometry  
from sensor_msgs.msg import PointCloud, PointField, ChannelFloat32
from composit_planner.srv import *
from composit_planner.msg import *

# GPy and GP includes
import GPy as GPy
from gpmodel_library import GPModel, OnlineGPModel
from scipy.stats import multivariate_normal
from scipy.stats import norm

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
        # Initialize member variables
        self.current_max = -float("inf")
        self.data_queue = list()
        self.pose_queue = list()
        self._maxima = None
        self.pose = Odometry() 
        
        # Create mutex for the data queue
        self.data_lock = threading.Lock()

        ''' Get ROS parameters '''
        # The kernel hyperparmeters for the robot's GP model 
        self.variance = float(rospy.get_param('model_variance','100'))
        self.lengthscale = float(rospy.get_param('model_lengthscale','0.1'))
        self.noise = float(rospy.get_param('model_noise', '0.0001'))
        
        # This range parameter will be shared by both the world and the robot
        self.x1min = float(rospy.get_param('xmin', '0'))
        self.x1max = float(rospy.get_param('xmax', '10'))
        self.x2min = float(rospy.get_param('ymin', '0'))
        self.x2max = float(rospy.get_param('ymax', '10'))
        
        # Initialize the robot's GP model with the initial kernel parameters
        self.GP = OnlineGPModel(ranges = [self.x1min, self.x1max, self.x2min, self.x2max], lengthscale = self.lengthscale, variance = self.variance, noise = self.noise)

        # Define ROS service
        self.srv_replan = rospy.Service('replan', RequestReplan, self.update_model)
        self.srv_maxima= rospy.Service('pred_value', GetValue, self.predict_value)
       
        #  Subscriptions and publiscations
        self.data = rospy.Subscriber("chem_data", ChemicalSample, self.get_sensordata)
        self.odom = rospy.Subscriber("/odom", Odometry, self.update_pose)

        self.pub = rospy.Publisher('chem_map', PointCloud, queue_size = 100)
        
        # Set sensing loop rate
        #r = rospy.Rate(rate)
        r = rospy.Rate(2)

        while not rospy.is_shutdown():
            # Pubish current belief map
            self.publish_gpbelief()
            r.sleep()

    def publish_gpbelief(self):
        # Generate a set of observations from robot model with which to make contour plots
        grid_size = 8.0 # grid size in meters
        num_pts = 100 # number of points to visaulzie in grid (num_pts x num_pts)
        x1max = self.pose.pose.pose.position.x + grid_size / 2.0
        x1min = self.pose.pose.pose.position.x - grid_size / 2.0
        x2max = self.pose.pose.pose.position.y + grid_size / 2.0
        x2min = self.pose.pose.pose.position.y - grid_size / 2.0

        x1 = np.linspace(x1min, x1max, num_pts)
        x2 = np.linspace(x2min, x2max, num_pts)
        x1, x2 = np.meshgrid(x1, x2, sparse = False, indexing = 'xy') # dimension: NUM_PTS x NUM_PTS       
        data = np.vstack([x1.ravel(), x2.ravel()]).T

        if self.GP.xvals is not None:
            observations, var = self.GP.predict_value(data)

            max_val = norm.ppf(q = 0.90, loc = 0.0, scale = np.sqrt(self.GP.variance))
            min_val = norm.ppf(q = 0.10, loc = 0.0, scale = np.sqrt(self.GP.variance))

            if max_val == min_val and max_val == 0.00: 
                topixel = lambda val: 0.0
            else:
                # Define lambda for transforming from observation to 0-255 range
                topixel = lambda val: int((val - min_val) / (max_val - min_val) * 255.0)

            pt_vals = np.array([topixel(c) for c in observations]).reshape(num_pts *  num_pts)
            pt_locs = data.T
            pt_cloud = np.array(np.hstack([pt_locs[0, :], pt_locs[1, :], pt_vals]), dtype = np.float32)
            
        msg = PointCloud()
        msg.header.frame_id = 'map' # Global frame

        val = ChannelFloat32()
        val.name = 'intensity'
        
        #msg.header.stamp = rospy.get_rostime()
        for i, d in enumerate(data):
            pt = geometry_msgs.msg.Point32()
            pt.x = data[i, 0]
            pt.y = data[i, 1]
            pt.z = 1.0
            msg.points.append(pt)

            if self.GP.xvals is None:
                val.values.append(255./2.)
            else:
                val.values.append(topixel(observations[i, :]))

        msg.channels.append(val)
        self.pub.publish(msg)
    
    def update_model(self, _):
        ''' Adds all data currently in the data queue into the GP model and clears the data queue. Threadsafe. 
        Input:
            None
        Output: 
            Boolean success service response''' 
        try:
            # Cannot update model if no data has been collected
            if len(self.data_queue) == 0:
                return RequestReplanResponse(False)

            # Aquire the data lock
            self.data_lock.acquire()

            # Add all current observations in the data queue to the current model
            NUM_PTS = len(self.data_queue)
            zobs = np.array([msg.data for msg in self.data_queue]).reshape(NUM_PTS, 1)
            xobs = np.array([[msg.pose.pose.position.x, msg.pose.pose.position.y] for msg in self.pose_queue]).reshape(NUM_PTS, 2)

            self.GP.add_data(xobs, zobs)
            rospy.loginfo("Number of sample points in belief model %d", self.GP.zvals.shape[0])
        
            # Update the current best max for EI
            for z, x in zip (zobs, xobs):
                if z[0] > self.current_max:
                    self.current_max = z[0]
                    self.current_max_loc = [x[0],x[1]]

            # Delete data from the data_queue
            del self.data_queue[:] 
            del self.pose_queue[:] 
            self._maxima = None

            # Release the data lock
            self.data_lock.release()

            return RequestReplanResponse(True)
        except ValueError as e:
            print e 
            return RequestReplanResponse(False)

    def predict_value(self, req):
        ''' Samples a set of K maxima from the current GP belief.  
        Input:
            None
        Output: 
            Boolean success service response''' 

        xvals = np.array([req.pose.position.x, req.pose.position.y]).reshape(1,2)
        aq_func = req.aq_func 
        if aq_func == 'ei':
            value = aq.exp_improvement(time = req.time, xvals = xvals, robot_model = self.GP, param = self.current_max)
        elif aq_func == 'ucb':
            value = aq.mean_ucb(time = req.time, xvals = xvals, robot_model = self.GP, param = None)
        elif aq_func == 'mes':
            value = aq.mves(time = req.time, xvals = xvals, robot_model = self.GP, param = self.maxima)
        elif aq_func == 'ig':
            value = aq.info_gain(time = req.time, xvals = xvals, robot_model = self.GP, param = None)
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
        self.pose_queue.append(self.pose)
        self.data_lock.release()    

    def update_pose(self, msg):
        self.pose = msg

    @property
    def maxima(self):
        if self._maxima is None:
            max_vals, max_locs, func = aq.sample_max_vals(self.GP) 
            self._maxima = (max_vals, max_locs, func)
        return self._maxima

def main():
	#initialize node
	rospy.init_node('chemical_belief')
	try:
		ChemicalBelief()
	except rospy.ROSInterruptException:
		pass
if __name__ == '__main__':
	main()
