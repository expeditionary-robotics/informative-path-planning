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
from sensor_msgs.msg import *
from scipy.ndimage import gaussian_filter, convolve
from tf import TransformListener


'''
This is a node which prevents crashes into nearby obstacles
'''

class Hindbrain:
    ''' The ObstacleCheck class, which represents the physical location of barriers and obstacles in a 3D world. Can optionally take in a predetermined map, or can generate a map live from a LIDAR subscription. On query, returns safe trajectories based on current status of the world. 
    '''
    def __init__(self):
        ''' Initialize the environment either from a parameter input or by subscribing to a topic of interest. 
        '''

        # get params
        self.safety_radius = rospy.get_param('turning_radius', 0.25)

        # subscribe to lasers
        self.lasers_sub = rospy.Subscriber('/scan',LaserScan,self.check_for_obstacles)

        # publish to generate new plan
        self.replan = rospy.ServiceProxy('replan', RequestReplan)

        #create polygon object to kill current trajectory
        self.path_pub = rospy.Publisher("/trajectory/current", PolygonStamped,
                                        queue_size=1)

        rospy.spin()

    def check_for_obstacles(self, req):
        ''' Given a list of Paths, check agains the current map for truncating
        Input: list of nav_msg/Path 
        Output: list of nav_msg/Path
        '''
        ang_info = [req.angle_min, req.angle_max, req.angle_increment]
        range_info = [req.range_min, req.range_max]

        ranges = map(self.check_scan, req.ranges)

        if sum(ranges) > length(req.ranges/2):
            msg = PolygonStamped()
            msg.header.stamp = rospy.Time(0)
            msg.header.frame_id = 'world'
            points = []
            self.path_pub.publish(msg)
            self.replan()

    def check_scan(self,r):
        if r < self.safety_radius:
            return 1
        else:
            return 0




def main():
	#initialize node
	rospy.init_node('hindbrain')
	try:
		Hindbrain()
	except rospy.ROSInterruptException:
		pass


if __name__ == '__main__':
	main()
