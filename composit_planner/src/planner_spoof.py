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
This node runs at a fixed rate, and triggers replanning. A stand-node for the true planner
'''

def get_measurements():
    # Wait for the (simulated) environment to initialize
    rospy.wait_for_service('replan')

    # Initialize pose and chemical sensor services
    replan = rospy.ServiceProxy('replan', RequestReplan)

    # Set sampling rate
    rate = 1./45. 

    # Set sensing loop rate
    r = rospy.Rate(rate)

    while not rospy.is_shutdown():
        # Get pose and chemical measurment
        resp = replan()
        print "Replan sucess?", resp.success

        # Publish data
        r.sleep()

if __name__ == '__main__':
	rospy.init_node('planner_spoof')
	try:
            get_measurements()
	except rospy.ROSInterruptException:
		pass
