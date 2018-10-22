#!/usr/bin/python

# Copyright 2018 Massachusetts Institute of Technology

import numpy as np
import math
import os
import rospy
from std_msgs.msg import *
from geometry_msgs.msg import *
from composit_planner.srv import *
from composit_planner.msg import *

'''
This node spoofs a localization node. Publishes a random 2D pose at a fixed rate
'''

if __name__ == '__main__':
    rospy.init_node('odom_spoofer')
    try:
        # Initialize ros publisher, set queue size to be 1 so only the freshest chem measurement is processed 
        pub = rospy.Publisher('odom_spoof', geometry_msgs.msg.Pose2D, queue_size = 1)
        
        # Set odom loop rate
        rate = float(rospy.get_param('odom_rate','100'))
        r = rospy.Rate(rate)

        current_pose = geometry_msgs.msg.Pose2D()

        while not rospy.is_shutdown():
            # Sample a uniformly random pose
            current_pose.x = np.random.uniform(low = 0.0, high = 10.0)
            current_pose.y = np.random.uniform(low = 0.0, high = 10.0)
            current_pose.theta  = 0.0

            # Publish data
            pub.publish(current_pose)
            r.sleep()


    except rospy.ROSInterruptException:
            pass
