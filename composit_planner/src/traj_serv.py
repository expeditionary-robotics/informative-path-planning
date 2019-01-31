#!/usr/bin/env python

# Copyright 2018 Massachusetts Institute of Technology

import rospy
import actionlib
from composit_planner.srv import *
from composit_planner.msg import *
from nav_msgs.msg import Path
from geometry_msgs.msg import *
from tf.transformations import quaternion_from_euler

#TODO this node can be used to trigger replanning in the midst of executing a trajectory already

class TrajMonitor():
    def __init__(self):
        #initialize node and callback
        rospy.init_node('execute_dubin')

        self.allowed_error = rospy.get_param('trajectory_endpoint_precision',0.2)
        self.replan_distance = None #TODO use this to trigger replan
        self.last_viable = None

        #subscribe to trajectory topic
        self.sub = rospy.Subscriber("/trajectory/current", PolygonStamped, self.handle_trajectory, queue_size=1)
        self.pose_sub = rospy.Subscriber("/pose", PoseStamped, self.handle_pose, queue_size=1)
        #access replan service to trigger when finished a trajectory
        self.replan = rospy.ServiceProxy('replan', RequestReplan)

        #spin until shutdown
        while not rospy.is_shutdown():
            rospy.spin()

    def handle_trajectory(self, traj):
        '''
        The trajectory comes in as a series of poses. It is assumed that the desired angle has already been determined
        '''
        # print 'Getting planning point'

        check_points = traj.polygon.points
        if self.last_viable is not None:
            if check_points[-1] == self.last_viable:
                self.last_viable = -1
            else:
                self.new_goals = traj.polygon.points
                self.last_viable = self.new_goals[-1]
        else:
            self.last_viable = None


    def handle_pose(self, msg):
        last = self.last_viable
        if last is not None:
            if last != -1:
                if (msg.pose.position.x-last.x)**2 + (msg.pose.position.y-last.y)**2 < self.allowed_error**2:
                    self.last_viable = -1
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
