#!/usr/bin/python

'''
This is a node which keeps a physical map of obstacles in the world, and provides a ROSSERVICE 
when queried to report whether or not a trajectory is in an obstacle. Assumes that there is a
map representation that is being published by some slam mapping system.

Copyright 2018 Massachusetts Institute of Technology
'''

import scipy.ndimage
import numpy as np
from obstacles import *
import rospy
from std_msgs.msg import *
from composit_planner.srv import *
from geometry_msgs.msg import *
from trajectory_msgs.msg import *
from nav_msgs.msg import *
from map_msgs.msg import *

class CostMap(object):
    '''
    The Obstacle_Map class, which represents the physical location of barriers and obstacles
    in a 3D world. Can optionally take in a predetermined map, or can generate a map live from
    a LIDAR subscription. On query, returns safe trajectories based on current status of the world.
    '''
    def __init__(self):
        '''
        Listens to the costmap topic and will provide a service to send the map when queried
        '''

        # make sure the map object has been instantiated
        self.map = None

        # pull in metamap details, if there are any
        self.bounding_box = rospy.get_param('bounding_box', None)
        self.origin = rospy.get_param('origin', None)

        # create a service that responds to queries about the current costmap
        self.srv = rospy.Service('obstacle_map', GetCostMap, self.return_map)

        # subscribe to the occupancy grid
        self.sub = rospy.Subscriber('/projected_map', OccupancyGrid, self.get_map, queue_size=1)

        # temp pub
        self.temp_pub = rospy.Publisher('/costmap', OccupancyGrid)

        # costmap params
        self.obstacle_threshold = 80.0
        inflation_radius_m = rospy.get_param('obstacle_buffer_m', 0.25)
        map_resolution = rospy.get_param('map_resolution', 0.1)
        self.inflation_radius = np.round(inflation_radius_m/map_resolution)

        # spin until interrupt
        rospy.spin()

    def return_map(self, req):
        '''
        The service request listened, forms the response
        '''
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
        data = make_array(msg.data, msg.info.height, msg.info.width)
        operation_grid = np.zeros(data.shape)
        operation_grid[data >= self.obstacle_threshold] = 1

        global_grid = data.copy()
        global_grid = inflate(global_grid, self.inflation_radius)

        costmap = OccupancyGrid()
        costmap.header.stamp = rospy.Time(0)
        costmap.header.frame_id = 'world'
        costmap.data = global_grid.flatten('C')
        costmap.info = msg.info

        return costmap

def make_array(data, height, width):
    '''
    Puts the map into a useable matrix form
    '''
    return np.array(data).reshape((height, width), order='C')

def inflate(operation_grid, inflation_radius):
    '''
    Applies the kernel of interest
    '''
    inflated_mask = scipy.ndimage.filters.maximum_filter(
        operation_grid, size=(inflation_radius, inflation_radius), cval=-1, mode='constant')
    return inflated_mask



if __name__ == '__main__':
	#initialize node
    rospy.init_node('costmap_server')
    try:
        CostMap()
    except rospy.ROSInterruptException:
        pass
