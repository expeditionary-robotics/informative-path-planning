#!/usr/bin/env python

'''Copyright 2018 Massachusetts Institute of Technology'''

import rospy
import numpy as np
from composit_planner.srv import *
from composit_planner.msg import *
from geometry_msgs.msg import *

class TrajMonitor(object):
    '''This class handles the replan trigger event based on the location of the vehicle relative to
    the last valid goal point.
    '''
    def __init__(self):
        '''initialize node and callback'''
        rospy.init_node('execute_dubin')

        self.allowed_error = rospy.get_param('trajectory_endpoint_precision', 0.2)
        self.last_viable = None
        self.new_goals = None

        #subscribe to trajectory topic
        self.sub = rospy.Subscriber("/trajectory/current", PolygonStamped, self.handle_trajectory, queue_size=1)
        self.pose_sub = rospy.Subscriber("/pose", PoseStamped, self.handle_pose, queue_size=1)
        
        self.traj_pub = rospy.Publisher("/trajectory/current", PolygonStamped, queue_size=1)

        #access replan service to trigger when finished a trajectory
        self.replan = rospy.ServiceProxy('replan', RequestReplan)

        #spin until shutdown
        while not rospy.is_shutdown():
            rospy.spin()

    def handle_trajectory(self, traj):
        '''
        The trajectory comes in as a series of poses. It is assumed that the desired angle has
        already been determined
        '''
        check_points = traj.polygon.points
        if self.last_viable != -1:
            if self.last_viable is not None and len(check_points) > 0 and np.isclose(check_points[-1].x, self.last_viable.x) is True:
                pass
            elif len(check_points) > 0:
                self.new_goals = traj.polygon.points
                self.last_viable = self.new_goals[-1]
        else:
            pass


    def handle_pose(self, msg):
        '''
        Listens for the robot's current pose and triggers replan event
        '''
        last = self.last_viable
        if last is not None:
            if last != -1:
                if (msg.pose.position.x-last.x)**2 + (msg.pose.position.y-last.y)**2 < self.allowed_error**2:
                    # print "~~~~~~~~TRIGGERING REPLAN~~~~~~~~~~~~"
                    pte = PolygonStamped()
                    pte.header.frame_id = 'world'
                    pte.header.stamp = rospy.Time(0)
                    self.traj_pub.publish(pte)
                    self.replan()
            else:
                pass
        else:
            pass



if __name__ == '__main__':
    try:
        TrajMonitor()
    except rospy.ROSInterruptException:
        rospy.loginfo("Trajectory Monitor terminated.")
