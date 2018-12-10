#!/usr/bin/python

# Copyright 2018 Massachusetts Institute of Technology

import numpy as np
import math
import os
import rospy
import time
import serial

from std_msgs.msg import *
from geometry_msgs.msg import *
from composit_planner.srv import *
from composit_planner.msg import *

'''
This node runs at a fixed rate, queries noisy sensor measurements from the world, and publishes.
'''

def get_measurements():
    # configure the serial connections (the parameters differs on the device you are connecting to)
    ser = serial.Serial(
	port='/dev/ttyACM2',
	baudrate=9600,
	parity=serial.PARITY_ODD,
	stopbits=serial.STOPBITS_TWO,
	bytesize=serial.SEVENBITS
    )

    if ser.isOpen() is False:
	ser.open()
    
    # Initialize ros publisher, set queue size to be 1 so only the freshest chem measurement is processed 
    pub = rospy.Publisher('chem_data', ChemicalSample, queue_size = 1)

    # Set sampling rate
    rate = float(rospy.get_param('sensor_rate','2'))

    # Set sensing loop rate
    r = rospy.Rate(rate)

    while not rospy.is_shutdown():
        # Get pose and chemical measurment
	loc = -1
	while loc == -1:
	    resp = ser.readline()
	    loc = resp.find('TVOC: ')

	fin = resp.find('p', loc)
	val = resp[loc + 6: fin]
	
        # Publish data
        pub.publish(data = float(val))

	print resp,
        r.sleep()

if __name__ == '__main__':
	rospy.init_node('real_sniffer')
	try:
            get_measurements()
	except rospy.ROSInterruptException:
		pass
