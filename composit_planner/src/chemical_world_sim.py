#!/usr/bin/python

# Copyright 2018 Massachusetts Institute of Technology

# System includes
import numpy as np
import math
import os
from scipy.stats import norm

# GPy and GP includes
import GPy as GPy
from gpmodel_library import GPModel, OnlineGPModel

# ROS includes
import rospy
from std_msgs.msg import *
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped 
from sensor_msgs.msg import PointCloud, PointField, ChannelFloat32
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


        self.global_max = None

        # Keeps track of current pose of the robot so to report the correct sensor measurement
        self.current_pose = PoseStamped()
        self.pose = rospy.Subscriber("/pose", PoseStamped, self.update_pose)

        # Generate the world
        self.make_world(None)

        # Define ROS service
        self.srv = rospy.Service('query_chemical', SimMeasurement, self.sample_value)
        self.visualize_map = rospy.Service('vis_chemworld', RequestReplan, self.publish_gp)
        self.regen_map = rospy.Service('regen_map', RequestRegen, self.make_world)
        self.pub_vis = rospy.Publisher('vis_chemworld', PointCloud, queue_size = 100)
        self.pub_max = rospy.Publisher('/true_maxima', PointCloud, queue_size = 100)
        
        rospy.spin()

    def make_world(self, msg):
        if msg is not None:
            seed = msg.seed
        # Generate the world
        # Generate a set of discrete grid points, uniformly spread across the environment
        x1 = np.linspace(self.x1min, self.x1max, self.num_pts)
        x2 = np.linspace(self.x2min, self.x2max, self.num_pts)
        # dimension: num_pts x num_pts
        x1vals, x2vals = np.meshgrid(x1, x2, sparse = False, indexing = 'xy') 
        # dimension: num_pts*num_pts x 2
        data = np.vstack([x1vals.ravel(), x2vals.ravel()]).T 

        bb = ((self.x1max - self.x1min)*0.01, (self.x2max - self.x2min) * 0.01)
        ranges = (self.x1min + bb[0], self.x1max - bb[0], self.x2min + bb[1], self.x2max - bb[1])
        # Initialize maxima arbitrarily to violate boundary constraints
        maxima = [self.x1min, self.x2min]


        # Seed the GP with a single "high" value in the middle
        xmax = np.array([ranges[0] + (ranges[1] - ranges[0])/2.0, ranges[2] + (ranges[3] - ranges[2])/ 2.0]).reshape((1, 2))
        zmax = np.array([norm.ppf(q = 0.999, loc = 0.0, scale = np.sqrt(self.variance))]).reshape((1, 1))

        print "Ranges:", ranges
        print "Setting maxima to :", zmax, "at location", xmax

        # Continue to generate random environments until the global maximia 
        # lives within the boundary constraints
        while maxima[0] < ranges[0] or maxima[0] > ranges[1] or \
              maxima[1] < ranges[2] or maxima[1] > ranges[3]:
            # logger.warning("Current environment in violation of boundary constraint. Regenerating!")

            # Intialize a GP model of the environment
            # self.GP = OnlineGPModel(ranges = ranges, lengthscale = self.lengthscale, variance = self.variance)         
            self.GP = GPModel(ranges = ranges, lengthscale = self.lengthscale, variance = self.variance)         
            self.GP.add_data(xmax, zmax)

            # Take an initial sample in the GP prior, conditioned on no other data
            data = np.vstack([x1vals.ravel(), x2vals.ravel()]).T 
            #xsamples = np.reshape(np.array(data[0, :]), (1, 2)) # dimension: 1 x 2        
            #mean, var = self.GP.predict_value(xsamples, include_noise = False)   
            #if msg is not None:
            #    np.random.seed(seed)
            #    seed += 1
            #elif self.seed is not None:
            #    np.random.seed(self.seed)
            #    self.seed += 1

            #zsamples = np.random.normal(loc = 0, scale = np.sqrt(var))
            #zsamples = np.reshape(zsamples, (1,1)) # dimension: 1 x 1 
                                
            # Add initial sample data point to the GP model
            #self.GP.add_data(xsamples, zsamples)                            

            np.random.seed(self.seed)
            observations = self.GP.posterior_samples(data, full_cov = True, size=1)

            self.GP = GPModel(ranges = ranges, lengthscale = self.lengthscale, variance = self.variance)         
            self.GP.add_data(data, observations) 
            maxima = self.GP.xvals[np.argmax(self.GP.zvals), :]
            self.global_max = (maxima, np.max(self.GP.zvals))
            print "Generated maxima:", self.global_max

        print "World generated! Size:", x1.shape, ",", x2.shape
        # Save the map for later comparison
        # np.save('../world_map_x1vals', x1vals)
        # np.save('../world_map_x2vals', x2vals)
        # np.save('../world_map_zvals', self.GP.zvals.reshape(x1vals.shape))
        np.savez('../world_map', x1=x1vals, x2=x2vals, z=self.GP.zvals.reshape(x1vals.shape))

        # self.publish_gp(None)
        
        return RequestRegenResponse(True)

    def publish_gp(self, _):
        # Generate a set of observations from robot model with which to make contour plots
        max_val = np.max(self.GP.zvals)
        min_val = np.min(self.GP.zvals)
        #print "Max val:", max_val
        #print "Min val:", min_val
        if max_val == min_val and max_val == 0.00: 
            topixel = lambda val: 0.0
        else:
            # Define lambda for transforming from observation to 0-255 range
            # topixel = lambda val: int((val - min_val) / (max_val - min_val) * 255.0)
            topixel = lambda val: float(val) 
        msg = PointCloud()
        msg.header.frame_id = 'map' # Global frame

	val = ChannelFloat32()
        val.name = 'ground_truth'
        #msg.header.stamp = rospy.get_rostime()
        for i, d in enumerate(self.GP.xvals):
            pt = geometry_msgs.msg.Point32()
            pt.x = self.GP.xvals[i, 0]
            pt.y = self.GP.xvals[i, 1]
            pt.z = 2.0
            msg.points.append(pt)
            val.values.append(topixel(self.GP.zvals[i, :]))
        msg.channels.append(val)
        self.pub_vis.publish(msg)

        ''' Publish the ground truth global maxima '''
        msg = PointCloud()
        msg.header.frame_id = 'map' # Global frame

	val = ChannelFloat32()
        val.name = 'global_maxima'
        #msg.header.stamp = rospy.get_rostime()
        pt = geometry_msgs.msg.Point32()
        pt.x = self.global_max[0][0]
        pt.y = self.global_max[0][1]
        pt.z = 2.0
        msg.points.append(pt)
        val.values.append(topixel(self.global_max[1]))
        msg.channels.append(val)
        self.pub_max.publish(msg)
        
        return RequestReplanResponse(True)

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
        xvals = np.array([self.current_pose.pose.position.x, self.current_pose.pose.position.y]).reshape(1,2)

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
