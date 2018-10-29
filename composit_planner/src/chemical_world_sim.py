#!/usr/bin/python

# Copyright 2018 Massachusetts Institute of Technology

# System includes
import numpy as np
import math
import os

# GPy and GP includes
import GPy as GPy
from gpmodel_library import GPModel, OnlineGPModel

# ROS includes
import rospy
from std_msgs.msg import *
from nav_msgs.msg import Odometry
from composit_planner.srv import *
from composit_planner.msg import *

'''
This is a ROSSERVICE tos perform the following task(s): 
    * publish noisy sensor information on request at the current robot pose.
'''

class Environment:
    '''The Environment class, which represents a retangular Gaussian world.
    ''' 
    def __init__(self):
        ''' Initialize a random Gaussian environment using the input kernel, 
            assuming zero mean function.
        Input:
        num_pts (int): the number of points in each dimension to sample for 
            initialization, resulting in a sample grid of size num_pts x num_pts
        variance (float): the variance parameter of the kernel
        lengthscale (float): the lengthscale parameter of the kernel
        noise (float): the sensor noise parameter of the kernel
        seed (int): an integer seed for the random draws. If set to \'None\', 
            no seed is used 
        '''
        self.num_pts = int(rospy.get_param('num_pts','10'))
        self.variance = float(rospy.get_param('variance','100'))
        self.lengthscale = float(rospy.get_param('lengthscale','0.1'))
        self.noise = float(rospy.get_param('noise', '0.0001'))
        self.seed = int(rospy.get_param('seed','0'))
        self.x1min = float(rospy.get_param('xmin', '0'))
        self.x1max = float(rospy.get_param('xmax', '10'))
        self.x2min = float(rospy.get_param('ymin', '0'))
        self.x2max = float(rospy.get_param('ymax', '10'))
        #self.delta = float(rospy.get_param('delta', '0.05'))
        self.delta = 0.5 # TODO: make seperate paraam 

        # Keeps track of current pose of the robot so to report the correct sensor measurement
        self.current_pose = Odometry()
        self.pose = rospy.Subscriber("/odom", Odometry, self.update_pose)

        # Generate the world
        # Generate a set of discrete grid points, uniformly spread across the environment
        x1 = np.linspace(self.x1min, self.x1max, (self.x1max - self.x1min) / self.delta)
        x2 = np.linspace(self.x2min, self.x2max, (self.x1max - self.x1min) / self.delta)
        # dimension: num_pts x num_pts
        x1vals, x2vals = np.meshgrid(x1, x2, sparse = False, indexing = 'xy') 
        # dimension: num_pts*num_pts x 2
        data = np.vstack([x1vals.ravel(), x2vals.ravel()]).T 

        bb = ((self.x1max - self.x1min)*0.05, (self.x2max - self.x2min) * 0.05)
        ranges = (self.x1min + bb[0], self.x1max - bb[0], self.x2min + bb[1], self.x2max - bb[1])
        # Initialize maxima arbitrarily to violate boundary constraints
        maxima = [self.x1min, self.x2min]

        # Continue to generate random environments until the global maximia 
        # lives within the boundary constraints
        while maxima[0] < ranges[0] or maxima[0] > ranges[1] or \
              maxima[1] < ranges[2] or maxima[1] > ranges[3]:
            # print "Current environment in violation of boundary constraint. Regenerating!"
            # logger.warning("Current environment in violation of boundary constraint. Regenerating!")

            # Intialize a GP model of the environment
            self.GP = OnlineGPModel(ranges = ranges, lengthscale = self.lengthscale, variance = self.variance)         
            data = np.vstack([x1vals.ravel(), x2vals.ravel()]).T 

            # Take an initial sample in the GP prior, conditioned on no other data
            xsamples = np.reshape(np.array(data[0, :]), (1, 2)) # dimension: 1 x 2        
            mean, var = self.GP.predict_value(xsamples, include_noise = False)   
            if self.seed is not None:
                np.random.seed(self.seed)
                self.seed += 1
            zsamples = np.random.normal(loc = 0, scale = np.sqrt(var))
            zsamples = np.reshape(zsamples, (1,1)) # dimension: 1 x 1 
                                
            # Add initial sample data point to the GP model
            self.GP.add_data(xsamples, zsamples)                            
            np.random.seed(self.seed)
            observations = self.GP.posterior_samples(data[1:, :], full_cov = True, size=1)
            self.GP.add_data(data[1:, :], observations) 
            maxima = self.GP.xvals[np.argmax(self.GP.zvals), :]

        print "World generated! Size:", x1.shape, ",", x2.shape
        # Save the map for later comparison
        # np.save('../world_map_x1vals', x1vals)
        # np.save('../world_map_x2vals', x2vals)
        # np.save('../world_map_zvals', self.GP.zvals.reshape(x1vals.shape))
        np.savez('../world_map', x1=x1vals, x2=x2vals, z=self.GP.zvals.reshape(x1vals.shape))

        # Define ROS service
        self.srv = rospy.Service('query_chemical', SimMeasurement, self.sample_value)
        
        rospy.spin()

    def update_pose(self, msg):
        self.current_pose = msg

    def sample_value(self, _):
        ''' The public interface to the Environment class. Returns a noisy sample of the true value of environment at a set of point. 
        Input:
            xvals (float array): an nparray of floats representing observation locations, with dimension num_pts x 2 
        
        Returns:
            mean (float array): an nparray of floats representing predictive mean, with dimension num_pts x 1 
        '''

        # In simulation, the chemical sensor must know the pose of the robot to report sensor measurement
        xvals = np.array([self.current_pose.pose.pose.position.x, self.current_pose.pose.pose.position.y]).reshape(1,2)

        mean, var = self.GP.predict_value(xvals, include_noise = False)
        return SimMeasurementResponse(mean + np.random.normal(loc = 0, scale = np.sqrt(self.noise)))


def main():
	#initialize node
	rospy.init_node('chemical_world_sim')
	try:
		Environment()
	except rospy.ROSInterruptException:
		pass


if __name__ == '__main__':
	main()
