#!/usr/bin/python

# Copyright 2018 Massachusetts Institute of Technology

import numpy as np
import math
import os
import rospy
from std_msgs.msg import *
from geometry_msgs.msg import *
from composit_planner.srv import *

'''
This node runs at 5Hz and queries noisy sensor measurements from the world and publishes.
'''

def get_measurements():
	rospy.wait_for_service('query_chemical')
	query_chemical = rospy.ServiceProxy('query_chemical', SimMeasurement)
	rate = float(rospy.get_param('rate','10'))
	r = rospy.Rate(rate)
	while not rospy.is_shutdown():
		query = Pose()
		query.position.x = 0.
		query.position.y = 0.
		query.position.z = 0.
		resp = query_chemical(query)
		print resp
		r.sleep()

if __name__ == '__main__':
	rospy.init_node('sniffer')
	try:
		get_measurements()
	except rospy.ROSInterruptException:
		pass
