#!/usr/bin/python

# Copyright 2018 Massachusetts Institute of Technology

import numpy as np
import math
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
        
        # subscribe to the initialization and periodic updates of the costmap
        self.sub = rospy.Subscriber('move_base/global_costmap/costmap', OccupancyGrid, self.get_map, queue_size=1)
        
        # subscribe to the partial updates of the costmap
        self.proc = rospy.Subscriber('move_base/global_costmap/costmap_updates', OccupancyGridUpdate, self.process_map, queue_size=1)
        
        # spin until interrupt
        rospy.spin()

    def return_map(self, req):
        '''
        The service request listened, forms the response
        '''
        print 'Map Queried'
        return GetCostMapResponse(self.map)

    def get_map(self, msg):
        '''
        Clears the current map from periodic full map updates
        '''
        print 'Map Established'
        self.map = msg

    def process_map(self, msg):
        '''
        Updates the full map with the "slices" posted by the update message
        '''
        print 'Map Updated'

        # get the update information
        idx = msg.x
        idy = msg.y
        width = msg.width
        height = msg.height
        data = np.asarray(msg.data)

        # get the map to update
        map_data = np.asarray(self.map.data)

        # update the map data; remember in row-major order
        index = 0
        for j in range(idy,idy+height):
            for i in range(idx,idx+width):
                map_data[j*self.map.info.width+i] = data[index]
                index += 1

        # update the occupancy grid object to return by the service
        new_map = OccupancyGrid()
        new_map.header = self.map.header
        new_map.info = self.map.info
        new_map.data = map_data
        self.map = new_map


def main():
	#initialize node
	rospy.init_node('costmap_server')
	try:
		CostMap()
	except rospy.ROSInterruptException:
		pass


if __name__ == '__main__':
	main()