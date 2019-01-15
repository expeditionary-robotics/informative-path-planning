#!/usr/bin/python

'''
This node listens for MAVROS GPS information in order to publish a local
odom coordinate

Copyright 2018 Massachusetts Institute of Technology
'''

import navpy
from obstacles import *
import rospy
from tf.transformations import quaternion_from_euler
from std_msgs.msg import *
from composit_planner.srv import *
from composit_planner.msg import *
from geometry_msgs.msg import *
from trajectory_msgs.msg import *
from nav_msgs.msg import *
from map_msgs.msg import *
from sensor_msgs.msg import *


class Converter(object):
    '''
    class to listen for MAVROS messages and convert them to useable interfaces for the
    COMPOSIT planning algorithm
    '''

    def __init__(self):
        '''
        Initialize the conversion scheme
        '''
        # get the lat, lon origin coordinate
        self.origin = rospy.get_param('origin', None)
       
        # Subscribed topics
        global_pos_topic = rospy.get_param('global_position_topic', 'mavros/global_position/global')
        hdg_topic = rospy.get_param('heading_topic', 'mavros/global_position/compass_hdg')
        altimeter_topic = rospy.get_param('altimeter_topic', '/slicklizard/sensors/micron_echo/data')
        goal_reached_topic = rospy.get_param('goal_reached_topic', 'mavros/rc/goal_reached') # custom

        # subscribe to the mavros state message
        rospy.Subscriber(global_pos_topic, NavSatFix, self.pos_glbl_cb)
        rospy.Subscriber(hdg_topic, Float64, self.hdg_cb)
        rospy.Subscriber(altimeter_topic, Range, self.altitude_cb)

        # publish to the odom topic
        self.odom_pub = rospy.Publisher('/pose', PoseStamped, queue_size=1)
        self.data_pub = rospy.Publisher('/chem_data', ChemicalSample, queue_size = 1)

        # initialize the variables of interest
        self.glbl_pose = [0, 0, 0]
        self.lcl_pose = [0, 0, 0]
        self.hdg = 0.

        # run the node at a certain rate to check things
        r = rospy.Rate(100)
        while not rospy.is_shutdown():
            # Pubish current belief map
            self.send_odom()
            r.sleep()

    def pos_glbl_cb(self, msg):
        '''Gets the navSatFix message from mavros'''
        self.glbl_pose = [msg.latitude, msg.longitude, msg.altitude]
        self.lcl_pose = navpy.lla2ned(msg.latitude, msg.longitude, 0.0,
                                      self.origin[0], self.origin[1], 0.0)

    def hdg_cb(self, msg):
        '''Gets our compass heading'''
        self.hdg = msg.data
    
    def altitude_cb(self, range_msg):
        ''' Listen for the altitude topic, potentially smooth, and republish
            to the chem_data topic '''

        alt = range_msg.range
        is_valid = range_msg.min_range < alt < range_msg.max_range

        if not is_valid:
            # rospy.logerr('Altitude measurement is out of range. Ignoring.')
            return

        self.data_pub.publish(data = float(alt))

    def send_odom(self):
        ''' Publishes a ROS Odom topic message for use in the planning system'''
        p = PoseStamped()
        p.header.frame_id = '/world'
        p.header.stamp = rospy.Time(0)

        p.pose.position.y = self.lcl_pose[0]
        p.pose.position.x = self.lcl_pose[1]
        p.pose.position.z = self.lcl_pose[2]

        q = quaternion_from_euler(0., 0., self.hdg)
        p.pose.orientation.w = q[3]
        p.pose.orientation.x = q[0]
        p.pose.orientation.y = q[1]
        p.pose.orientation.z = q[2]

        self.odom_pub.publish(p)

if __name__ == '__main__':
        # initialize node
    rospy.init_node('converter')
    try:
        Converter()
    except rospy.ROSInterruptException:
        pass
