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
from nav_msgs.msg import *
import dubins
from tf.transformations import quaternion_from_euler, euler_from_quaternion

'''
This node spoofs a trajectory callback from the planner
'''
class TrajectorySpoofer():
    def __init__(self):
        self.pub = rospy.Publisher('selected_trajectory', Path, queue_size = 1)
        self.sub = rospy.Subscriber('odom',Odometry,self.handle_pose)

        self.pose = None

        self.check_traj = rospy.ServiceProxy('query_obstacles', TrajectoryCheck)

        rate = float(0.1)
        r = rospy.Rate(rate)

        while not rospy.is_shutdown():
            if self.pose is not None:
                set_goal = (self.pose[0]+1, self.pose[1]+2, self.pose[2]-1.2)
                path = dubins.shortest_path(self.pose, set_goal, 0.7)
                true_path, _ = path.sample_many(0.1)

                pub_path = []
                for coord in true_path:
                    c = PoseStamped()
                    c.header.frame_id = 'odom'
                    #c.header.stamp = rospy.Time.now()
                    c.header.stamp = rospy.Time(0)
                    c.pose.position.x = coord[0]
                    c.pose.position.y = coord[1]
                    c.pose.position.z = 0.
                    q = quaternion_from_euler(0, 0, coord[2])
                    c.pose.orientation.x = q[0]
                    c.pose.orientation.y = q[1]
                    c.pose.orientation.z = q[2]
                    c.pose.orientation.w = q[3]
                    pub_path.append(c)

                pte = Path()
                pte.header.frame_id = 'odom'
                #pte.header.stamp = rospy.Time.now()
                pte.header.stamp = rospy.Time(0)
                pte.poses = pub_path
                pte = self.check_traj(TrajectoryCheckRequest(pte))
                pte.safe_path.header.frame_id = 'odom'
                #pte.safe_path.header.stamp = rospy.Time.now() 
                pte.safe_path.header.stamp = rospy.Time(0) 
                self.pub.publish(pte.safe_path)
            r.sleep()


    def handle_pose(self,msg):
        q = msg.pose.pose.orientation
        angle = euler_from_quaternion((q.x, q.y, q.z, q.w))
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, angle[2])



if __name__ == '__main__':
    rospy.init_node('traj_spoofer')
    try:
        TrajectorySpoofer()

    except rospy.ROSInterruptException:
        pass
