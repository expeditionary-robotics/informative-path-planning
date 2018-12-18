#!/usr/bin/env python

'''Copyright 2018 Massachusetts Institute of Technology'''

import rospy
from composit_planner.srv import *
from composit_planner.msg import *
from mavros_msgs.msg import *
from geometry_msgs.msg import *
from std_msgs.msg import *

# TODO this node can be used to trigger replanning in the midst of executing a trajectory already


class TrajMonitor(object):
    '''
    Node that arbitrates replanning system by parsing polygon object into mavros waypoint
    and listening for controller callback.
    '''

    def __init__(self):
        '''initialize node and callback'''
        rospy.init_node('execute_dubin')

        # subscribers
        rospy.Subscriber("/trajectory/current", PolygonStamped, self.handle_trajectory, queue_size=1)
        goal_reached_topic = rospy.get_param('goal_reached_topic', 'mavros/rc/goal_reached')
        rospy.Subscriber(goal_reached_topic, Bool, self.handle_reached, queue_size=1)

        # publishers
        lcl_waypt_topic = rospy.get_param('lcl_waypt_topic', 'mavros/setpoint_raw/local')
        self.waypt_pub = rospy.Publisher(lcl_waypt_topic, PositionTarget, queue_size=1)

        # access replan service to trigger when finished a trajectory
        self.replan = rospy.ServiceProxy('replan', RequestReplan)
        self.can_replan = False

        # spin until shutdown
        while not rospy.is_shutdown():
            rospy.spin()

    def handle_trajectory(self, traj):
        '''
        The trajectory comes in as a series of poses. It is assumed that the desired angle
        has already been determined
        '''
        goal = traj.polygon.points[-1]
        g = PositionTarget()
        g.header.frame_id = ''
        g.header.stamp = rospy.Time(0)
        g.coordinate_frame = g.FRAME_LOCAL_OFFSET_NED
        # g.type_mask = g.IGNORE_PZ
        g.position.x = goal[0]
        g.position.y = goal[1]
        g.position.z = 0.
        self.waypt_pub.publish(g)
        self.can_replan = True


    def handle_reached(self, msg):
        '''Sends replanning message when we reach goal'''
        if msg.data is True and self.can_replan is True:
            self.replan()
            self.can_replan = False


if __name__ == '__main__':
    try:
        TrajMonitor()
    except rospy.ROSInterruptException:
        rospy.loginfo("Trajectory Monitor terminated.")
