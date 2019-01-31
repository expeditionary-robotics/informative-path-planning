#!/usr/bin/env python

'''
This library can be used to access the multiple ways in which path sets can be generated for the simulated vehicle in the PLUMES framework.

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''

import numpy as np
import math
import dubins
import rospy
from std_msgs.msg import *
from composit_planner.srv import *
from geometry_msgs.msg import *
from trajectory_msgs.msg import *
from nav_msgs.msg import *
from tf.transformations import quaternion_from_euler, euler_from_quaternion


class ROS_Path_Generator():
    def __init__(self):
        ''' Initialize a path generator
        Input:
            frontier_size (int) the number of points on the frontier we should consider for navigation
            horizon_length (float) distance between the vehicle and the horizon to consider
            turning_radius (float) the feasible turning radius for the vehicle
            sample_step (float) the unit length along the path from which to draw a sample
        '''
        rospy.init_node("path_generator")

        # the parameters for the dubin trajectory
        self.fs = rospy.get_param('frontier_size',15)
        self.hl = rospy.get_param('horizon_length',1.5)
        self.tr = rospy.get_param('turning_radius',0.05)
        self.ss = rospy.get_param('sample_step',0.5)

        # Global variables
        self.goals = [] #The frontier coordinates
        self.samples = {} #The sample points which form the paths
        self.cp = (0,0,0) #The current pose of the vehicle

        self.srv_path = rospy.Service('get_paths', PathFromPose, self.get_path_set)
        self.check_traj = rospy.ServiceProxy('query_obstacles', TrajectoryCheck)

        while not rospy.is_shutdown():
            rospy.spin()

    def generate_frontier_points(self):
        '''From the frontier_size and horizon_length, generate the frontier points to goal'''
        angle = np.linspace(-2.35,2.35,self.fs) #fix the possibilities to 75% of the unit circle, ignoring points directly behind the vehicle
        goals = []
        for a in angle:
            x = self.hl*np.cos(self.cp[2]+a)+self.cp[0]
            y = self.hl*np.sin(self.cp[2]+a)+self.cp[1]
            p = self.cp[2]+a
            goals.append((x,y,p))
        goals.append(self.cp)
        self.goals = goals
        return self.goals

    def make_sample_paths(self):
        '''Connect the current_pose to the goal places'''
        all_paths = []
        for goal in self.goals:
            path = dubins.shortest_path(self.cp, goal, self.tr)
            true_path, _ = path.sample_many(self.ss)
            all_paths.append(true_path)
        return all_paths

    def rosify_safe_path(self, paths):
        clear_paths = []
        for path in paths:
            pub_path = []
            for coord in path:
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
            pte.header.stamp = rospy.Time(0)
            pte.poses = pub_path
            clear_paths.append(pte)
        clear_paths = self.check_traj(TrajectoryCheckRequest(clear_paths))
        return clear_paths.safe_path

    def get_path_set(self, req):
        '''Primary interface for getting list of path sample points for evaluation
        Input:
            current_pose (tuple of x, y, z, a which are floats) current location of the robot in world coordinates
        Output:
            paths (dictionary of frontier keys and sample points)
        '''
        self.cp = self.handle_pose(req.query)
        self.generate_frontier_points()
        paths = self.make_sample_paths()
        safe_paths = self.rosify_safe_path(paths)
        return PathFromPoseResponse(safe_paths)

    def get_frontier_points(self):
        ''' Method to access the goal points'''
        return self.goals

    def handle_pose(self, msg):
        q = msg.orientation
        angle = euler_from_quaternion((q.x, q.y, q.z, q.w))
        return (msg.position.x, msg.position.y, angle[2])


if __name__ == '__main__':
    try:
        ROS_Path_Generator()
    except rospy.ROSInterruptException:
        rospy.loginfo("Path generator finished")
