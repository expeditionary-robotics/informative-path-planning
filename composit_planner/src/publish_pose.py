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
This node spoofs a localization node. Can be queried and will respond with a 2D pose
'''

def write_pose(req):
    pose = geometry_msgs.msg.Pose2D()
    # Sample a uniformly random pose
    pose.x = np.random.uniform(low = 0.0, high = 10.0)
    pose.y = np.random.uniform(low = 0.0, high = 10.0)
    pose.theta  = 0.0

    return SimPoseResponse(pose)


if __name__ == '__main__':
	rospy.init_node('pose_spoofer')
	try:
            rospy.Service('query_pose', SimPose, write_pose)
            rospy.spin()

	except rospy.ROSInterruptException:
		pass
