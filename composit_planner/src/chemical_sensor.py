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
This node runs at 5Hz and queries noisy sensor measurements from the world and publishes.
    * TODO: should localization be in this node?
    * TOOD: revisit the data flow of localization and sensor measurements
'''

def get_measurements():
        # Wait for the (simulated) environment to initialize
	rospy.wait_for_service('query_chemical')
        # Wait for the robot localization service to initialize
	rospy.wait_for_service('query_pose')

        # Initialize pose and chemical sensor services
	query_pose = rospy.ServiceProxy('query_pose', SimPose)
	query_chemical = rospy.ServiceProxy('query_chemical', SimMeasurement)

        # Initialize ros publisher
        pub = rospy.Publisher('chem_data', ChemicalSample, queue_size = 100)

        # Set sampling rate
	rate = float(rospy.get_param('rate','10'))

        # Set sensing loop rate
	r = rospy.Rate(rate)

	while not rospy.is_shutdown():
                # Get pose and chemical measurment
                pose = query_pose()
		resp = query_chemical(pose.pose)

                # Publish data
	        pub.publish(loc = pose.pose, data = float(resp.sig))
                r.sleep()

if __name__ == '__main__':
	rospy.init_node('sniffer')
	try:
		get_measurements()
	except rospy.ROSInterruptException:
		pass
