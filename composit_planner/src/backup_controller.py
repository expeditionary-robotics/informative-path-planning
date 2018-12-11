#!/usr/bin/python

'''
This library can be used to access the multiple ways in which path sets can be generated for the simulated vehicle in the PLUMES framework.

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''

import numpy as np
import math
import dubins
import rospy
from geometry_msgs.msg import *
from nav_msgs.msg import * 
from sensor_msgs.msg import *
from std_msgs.msg import *
from composit_planner.srv import *
from composit_planner.msg import *
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf import TransformListener
import copy

class Backup_Controller():
    def __init__(self):
        ''' Initialize a path generator
        Input:
            frontier_size (int) the number of points on the frontier we should consider for navigation
            horizon_length (float) distance between the vehicle and the horizon to consider
            turning_radius (float) the feasible turning radius for the vehicle
            sample_step (float) the unit length along the path from which to draw a sample
        '''
        rospy.init_node("backup_controller")

        self.sub = rospy.Subscriber('call_backup', Bool, self.handle_backup, queue=1)
        # Publish directly to car controller
        # self.pub = rospy.Publisher('')

        # Call a replan
        self.replan = rospy.ServiceProxy('replan', RequestReplan)

        while not rospy.is_shutdown():
            rospy.spin()

    def handle_backup(self, msg):
        if msg.data == True:
            # make controller message
            # spin for a set amount of time
                # self.pub
            # call a replan event
            self.replan()


if __name__ == '__main__':
    try:
        Backup_Controller()
    except rospy.ROSInterruptException:
        rospy.loginfo("Backup Controller finished")
