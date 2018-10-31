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
This node runs at a fixed rate, queries noisy sensor measurements from the world, and publishes.
'''

def get_measurements():
    # Wait for the (simulated) environment to initialize
    rospy.wait_for_service('query_chemical')

    # Initialize pose and chemical sensor services
    query_chemical = rospy.ServiceProxy('query_chemical', SimMeasurement)

    # Initialize ros publisher, set queue size to be 1 so only the freshest chem measurement is processed 
    pub = rospy.Publisher('chem_data', ChemicalSample, queue_size = 1)

    # Set sampling rate
    rate = float(rospy.get_param('sensor_rate','10'))

    # Set sensing loop rate
    r = rospy.Rate(rate)

    while not rospy.is_shutdown():
        # Get pose and chemical measurment
        resp = query_chemical()

        # Publish data
        pub.publish(data = float(resp.sig))
        r.sleep()

if __name__ == '__main__':
	rospy.init_node('sniffer')
	try:
            get_measurements()
	except rospy.ROSInterruptException:
		pass
