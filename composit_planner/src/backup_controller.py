#!/usr/bin/python

'''
Copyright 2018 Massachusetts Institute of Technology
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


        self.sub = rospy.Subscriber('call_backup', Bool, self.handle_backup, queue_size=1)
        # Publish directly to car controller
        self.control_pub = rospy.Publisher('trajectory/current', PolygonStamped, queue_size=1)

        self.active = False 

        # Call a replan
        self.replan = rospy.ServiceProxy('replan', RequestReplan)

        while not rospy.is_shutdown():
            rospy.spin()

    def handle_backup(self, msg):
        if self.active == False:
            print 'Handling Backup!'
            self.active = True
            recieve_time = rospy.get_rostime().secs
            print recieve_time
            if msg.data == True:
                self.active = True
                abort_mission = PolygonStamped()
                abort_mission.header.frame_id = 'world'
                abort_mission.header.stamp = rospy.Time(0)
                abort_mission.polygon.points = [Point32(-1,0,0)]
                self.control_pub.publish(abort_mission)
                while rospy.get_rostime().secs < recieve_time + 1.0:
                    pass
                self.active = False
                self.replan()


if __name__ == '__main__':
    try:
        Backup_Controller()
    except rospy.ROSInterruptException:
        rospy.loginfo("Backup Controller finished")
