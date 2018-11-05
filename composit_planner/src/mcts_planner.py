#!/usr/bin/python

# Copyright 2018 Massachusetts Institute of Technology

import numpy as np
import math
import os
import rospy
from std_msgs.msg import *
from geometry_msgs.msg import *
from nav_msgs.msg import * 
from composit_planner.srv import *
from composit_planner.msg import *
import paths_library as paths_lib
from tf.transformations import quaternion_from_euler, euler_from_quaternion


'''
This node runs the MCTS system in order to select trajectories
'''

class Planner:
    def __init__(self):
        #Wait for (simulated) environment to initialize
        rospy.wait_for_service('query_obstacles')

        #Access all of the services
        self.srv_traj = rospy.ServiceProxy('query_obstacles', TrajectoryCheck)
        self.srv_chem = rospy.ServiceProxy('query_chemical', SimMeasurement)
        self.srv_replan = rospy.ServiceProxy('replan', RequestReplan)
        self.srv_maxima= rospy.ServiceProxy('pred_value', GetValue)

        #Subsribe to relevant nodes
        self.pose_sub = rospy.Subscriber("/odom", Odometry, self.update_pose)

        #We'll be publishing a path plan to execute
        self.plan_pub = rospy.Publisher("/selected_trajectory", Path, queue_size=1)

        #Get planning parameters like number of replanning steps
        self.replan_budget = rospy.get_param('replan_budget',150)
        self.fs = rospy.get_param('frontier_size',15)
        self.hl = rospy.get_param('horizon_length',1.5)
        self.tr = rospy.get_param('turning_radius',0.05)
        self.ss = rospy.get_param('sample_step',0.5)
        self.type_planner = rospy.get_param('type_planner', 'myopic')
        self.rl = rospy.get_param('rollout_length', 5)
        self.tt = rospy.get_param('tree_type','dpw')
        self.rate = rospy.get_param('replan_rate',0.1)
        self.rew = rospy.get_param('reward_func','mes')
        r = rospy.Rate(self.rate)

        self.path_generator = paths_lib.ROS_Path_Generator(self.fs, self.hl, self.tr, self.ss)
        self.cp = None

        while not rospy.is_shutdown():
            replan = self.srv_replan()
            if replan.success == True:
                r.sleep()
            self.get_plan()
            r.sleep()

    def update_pose(self,msg):
        self.pose = msg.pose.pose
        q = self.pose.orientation
        angle = euler_from_quaternion((q.x,q.y,q.z,q.w))
        self.cp = (self.pose.position.x, self.pose.position.y, angle[2])

    def choose_myopic_trajectory(self):
        # Generate options
        options = self.path_generator.get_path_set(self.cp)

        clear_paths = []
        # Check options against the current map
        for path in options:
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
            #pte.header.stamp = rospy.Time.now()
            pte.header.stamp = rospy.Time(0)
            pte.poses = pub_path
            pte = self.srv_traj(TrajectoryCheckRequest(pte))
            clear_paths.append(pte)

        #Now, select the path with the highest potential reward
        path_selector = {}
        for i,path in enumerate(clear_paths):
            value = 0
            for sample in path.safe_path.poses:
                s = Pose()
                s.position.x = sample.pose.position.x
                s.position.y = sample.pose.position.y
                val = self.srv_maxima(GetValueRequest(self.rew,s,1))
                value += val.value
            path_selector[i] = value
        best_key = np.random.choice([key for key in path_selector.keys() if path_selector[key] == max(path_selector.values())])
        return clear_paths[best_key], path_selector[best_key]


    def get_plan(self):
        if self.type_planner == 'myopic':
            if self.cp is not None:
                best_path, value = self.choose_myopic_trajectory()
                self.plan_pub.publish(best_path.safe_path)
            else:
                pass
        else:
            #TODO rewrite the MCTS library in order to support the type of planning we want to do
            # belief_snapshot = None #placeholder
            # mcts = mctslib.cMCTS(self.cb, self.GP, self.cp, self.rl, self.path_generator, self.aquisition_function, self.f_rew, None, tree_type = self.tt)
            # sampling_path, best_path, best_val, all_paths, all_values, self.max_locs, self.max_val = mcts.choose_trajectory(t=None)
            # self.plan_pub.publish(best_path)
            pass


if __name__ == '__main__':
    rospy.init_node('mcts_planner')
    try:
        Planner()
    except rospy.ROSInterruptException:
        pass
