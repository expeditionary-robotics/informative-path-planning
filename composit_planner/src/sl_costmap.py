#!/usr/bin/python

'''
This node makes a Slick Lizard or BlueROV map of the world, in which it is assumed that there is
mostly freespace but a bounding box may apply and need to be queried.

Copyright 2018 Massachusetts Institute of Technology
'''

import scipy.ndimage
import scipy.spatial
import numpy as np
import cv2
from obstacles import *
import rospy
import navpy
from std_msgs.msg import *
from composit_planner.srv import *
from geometry_msgs.msg import *
from trajectory_msgs.msg import *
from nav_msgs.msg import *
from map_msgs.msg import *

class CostMap(object):
    '''
    The Obstacle_Map class, which projects "barriers" based on the bounding box and origin
    provided by the system. Interface is shared with other vehicle representations.
    '''
    def __init__(self):
        '''
        Listens to the costmap topic and will provide a service to send the map when queried
        '''

        # make sure the map object has been instantiated
        self.map = None

        # costmap params
        self.obstacle_threshold = 80.0
        inflation_radius_m = rospy.get_param('obstacle_buffer_m', 0.25)
        self.map_resolution = rospy.get_param('map_resolution', 0.1)
        self.inflation_radius = np.round(inflation_radius_m/self.map_resolution)

        # self.bounding_box = rospy.get_param('bounding_box', None) #in lat lon coords
        self.bounding_box = rospy.get_param('bounding_box', None) #in NED (m) coordinates
        origin = rospy.get_param('origin', None) #in lat lon coords
        if self.bounding_box and origin is not None:
            # go through the points and convert to meters
            lat_ref = origin[0]
            lon_ref = origin[1]
            loc_points = []
            loc_north = []
            loc_east = []
            for point in self.bounding_box:
                # rel_point = navpy.lla2ned(point[0], point[1], 0.0, lat_ref, lon_ref, 0.0)
                # loc_points.append(rel_point)
                # loc_north.append(rel_point[0])
                # loc_east.append(rel_point[1])
                loc_points.append(point)
                loc_north.append(point[0])
                loc_east.append(point[1])

            self.origin = Pose()
            self.origin.position.x = 0.0
            self.origin.position.y = 0.0
            self.origin.position.z = 0.0
            self.origin.orientation.x = 0.
            self.origin.orientation.y = 0.
            self.origin.orientation.z = 0.
            self.origin.orientation.w = 1.
            self.height = int(np.fabs((np.nanmax(loc_north))/self.map_resolution)) #in cells
            self.width = int(np.fabs((np.nanmax(loc_east))/self.map_resolution)) #in cells
            rospy.loginfo('Costmap height and width (in cells) [%f x %f]'%(self.height, self.width))
            
            self.points = loc_points #in meters
        else: 
            rospy.logerr('No origin and geofence provided')

        # create a service that responds to queries about the current costmap
        self.srv = rospy.Service('obstacle_map', GetCostMap, self.return_map)

        # temp pub
        self.temp_pub = rospy.Publisher('/costmap', OccupancyGrid, queue_size=1)

        # make the static map
        self.process_map()

        # spin until interrupt
        rospy.spin()

    def return_map(self, req):
        '''
        The service request listened, forms the response
        '''
        self.temp_pub.publish(self.map)
        return GetCostMapResponse(self.map)

    def process_map(self):
        '''
        Inflates the obstacles in the map to give rough costmap system
        '''

        # get the map data and isolate edges of interest
        data = self.make_array(self.height, self.width)
        # operation_grid = np.zeros(data.shape)
        # operation_grid[data >= self.obstacle_threshold] = 1

        # global_grid = data.copy()
        # global_grid = inflate(global_grid, self.inflation_radius)

        costmap = OccupancyGrid()
        costmap.header.stamp = rospy.Time(0)
        costmap.header.frame_id = 'world'
        costmap.data = data.flatten('C')
        costmap.info.width = self.width
        costmap.info.height = self.height
        costmap.info.resolution = self.map_resolution
        costmap.info.origin = self.origin

        self.map = costmap

    def make_array(self, height, width):
        '''
        Puts the map into a useable matrix form
        '''
        base_map = np.zeros((int(height), int(width)), dtype=float)
        indices = []
        for p in self.points:
            #convert to cells
            project_x = p[0]/self.map_resolution
            project_y = p[1]/self.map_resolution
            indices.append([int(project_y), int(project_x)])
        indices = np.array(indices)
        indices = indices.reshape((-1, 1, 2))
        base_map = cv2.polylines(base_map, [np.array(indices)], True, color = 100, thickness = min(1, int(1.0/self.map_resolution)))
        #connection the points
        #return the matrix
        return base_map

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
