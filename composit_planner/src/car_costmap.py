#!/usr/bin/python

# Copyright 2018 Massachusetts Institute of Technology

import numpy as np
import math
import scipy.ndimage
import skimage.graph
import os
import GPy as GPy
from obstacles import *
import rospy
from std_msgs.msg import *
from composit_planner.srv import *
from geometry_msgs.msg import *
from trajectory_msgs.msg import *
from nav_msgs.msg import *
from map_msgs.msg import *


'''
This is a node which keeps a physical map of obstacles in the world, and provides a ROSSERVICE when queried to report whether or not a trajectory is in an obstacle
'''

class CostMap:
    ''' The Obstacle_Map class, which represents the physical location of barriers and obstacles in a 3D world. Can optionally take in a predetermined map, or can generate a map live from a LIDAR subscription. On query, returns safe trajectories based on current status of the world. 
    '''
    def __init__(self):
        ''' Listens to the costmap topic and will provide a service to send the map when queried 
        '''

        # make sure the map object has been instantiated
        self.map = None

        # create a service that responds to queries about the current costmap
        self.srv = rospy.Service('obstacle_map', GetCostMap, self.return_map)
        
        # subscribe to the occupancy grid
        self.sub = rospy.Subscriber('/projected_map', OccupancyGrid, self.get_map, queue_size=1)

        # temp pub
        self.temp_pub = rospy.Publisher('/costmap', OccupancyGrid)

        # costmap params
        self.obstacle_threshold = 80.0
        inflation_radius_m = rospy.get_param('obstacle_buffer_m',0.25)
        map_resolution = 0.05 #TODO Read this automatically
        self.inflation_radius = np.round(inflation_radius_m/map_resolution)

        # spin until interrupt
        rospy.spin()

    def return_map(self, req): 
 	'''
        The service request listened, forms the response
        '''
        # print 'Map Queried'
	return GetCostMapResponse(self.map)

    def get_map(self, msg):
        '''
        Clears the current map from periodic full map updates
        '''
        # print 'Map Established'
        self.map = self.process_map(msg)
        self.temp_pub.publish(self.map)

    def process_map(self, msg):
        '''
        Inflates the obstacles in the map to give rough costmap system
        '''

        # get the map data and isolate edges of interest
        data = self.make_array(msg.data, msg.info.height, msg.info.width)
        operation_grid = np.zeros(data.shape)
        operation_grid[data>=self.obstacle_threshold] = 1

        global_grid = data.copy()
        r = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        radii = map(lambda x: x*self.inflation_radius, r)
        for i,radius in enumerate(radii):
            inflated_mask = self.inflate(operation_grid, radius)
            inflated_grid = data.copy()
            inflated_grid[inflated_mask] = 100
            global_grid = np.add(global_grid,inflated_grid)

        global_grid = np.multiply(global_grid, [1./(i+2)])

        costmap = OccupancyGrid()
        costmap.header.stamp = rospy.Time(0)
        costmap.header.frame_id = 'world'
        costmap.data = global_grid.flatten('C')
        costmap.info = msg.info

        return costmap

    def make_array(self,data,height,width):
        return np.array(data).reshape((height,width),order='C')#self.make_array(msg.data, msg.info.height, msg.info.width)

        # output = np.zeros((height,width))
        # for i in range(width):
        #     for j in range(height):
        #         output[i,j] = data[i+j*width]
        # return output

    def inflate(self, operation_grid, inflation_radius):
        kernel_size = int(1+2*math.ceil(inflation_radius))
        cind = int(math.ceil(inflation_radius))
        x, y = np.ogrid[-cind:kernel_size-cind, -cind:kernel_size-cind]
        kernel = np.zeros((kernel_size,kernel_size))
        kernel[y*y+x*x <= inflation_radius*inflation_radius] = 1
        inflated_mask = scipy.ndimage.filters.convolve(operation_grid,
                                                       kernel,
                                                       mode='constant',
                                                       cval=0)
        inflated_mask = inflated_mask >= 1.0
        return inflated_mask


def main():
	#initialize node
	rospy.init_node('costmap_server')
	try:
		CostMap()
	except rospy.ROSInterruptException:
		pass


if __name__ == '__main__':
	main()
