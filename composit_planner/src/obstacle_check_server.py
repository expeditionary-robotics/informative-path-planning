#!/usr/bin/python

# Copyright 2018 Massachusetts Institute of Technology

import numpy as np
import math
import os
import GPy as GPy
from obstacles import *
import rospy
from std_msgs.msg import *
from composit_planner.srv import *
from geometry_msgs.msg import *
from trajectory_msgs.msg import *
from nav_msgs.msg import *
from scipy.ndimage import gaussian_filter, convolve
from tf import TransformListener


'''
This is a node which keeps a physical map of obstacles in the world, and provides a ROSSERVICE when queried to report whether or not a trajectory is in an obstacle
'''

class ObstacleCheck:
    ''' The ObstacleCheck class, which represents the physical location of barriers and obstacles in a 3D world. Can optionally take in a predetermined map, or can generate a map live from a LIDAR subscription. On query, returns safe trajectories based on current status of the world. 
    '''
    def __init__(self):
        ''' Initialize the environment either from a parameter input or by subscribing to a topic of interest. 
        Input:
        - cost_limit (float) number from 0.-100. to curtail a trajectory
        '''

        # get params
        self.safe_threshold = rospy.get_param('cost_limit', 85.)

        # subscribe to transforms
        self.tf = TransformListener()
        # create service to query cost map from server
        self.world = rospy.ServiceProxy('obstacle_map', GetCostMap)
        # create service to create adjusted safe trajectories
        self.srv = rospy.Service('query_obstacles', TrajectoryCheck, self.check_obstacles)
        # spin until interrupt
        rospy.spin()

    def check_obstacles(self, req):
        #parse the poses in a trajectory and check if any are in/near obstacles
        #TODO think of a clever matrix-transform way of doing this for speed up
        updated_trajectory = []
        map_resp = self.world(GetCostMapRequest())
        current_map = map_resp.map
        # reshape the array to be the matrix of interest
        data = self.make_array(current_map.data, current_map.info.height, current_map.info.width)
        np.save('../cost_map', data)
        # transform trajectory coordinates into indices of the matrix to query
        true_coords = req.query_path.poses
        for c in true_coords:
            self.tf.getLatestCommonTime("/odom", "/base_link")
            p = self.tf.transformPose('/base_link', c)
            self.tf.getLatestCommonTime("/map", "/base_link")
            p = self.tf.transformPose('/map', p)
            idx = (p.pose.position.x-current_map.info.origin.position.x)/current_map.info.resolution
            idy = (p.pose.position.y-current_map.info.origin.position.y)/current_map.info.resolution
            # check that each index is less than some threshold
            queryx, queryy = self.traj_index(idx,idy)
            safe_flag = True
            for x,y in zip(queryx,queryy):
                if data[x,y] > self.safe_threshold:
                    safe_flag = False
                    break
            if safe_flag:
                updated_trajectory.append(c)
            else:
                break
        #create the response message
        resp = Path()
        #resp.header.stamp = rospy.Time.now()
        resp.header.stamp = rospy.Time(0)
        resp.header.frame_id = 'odom'
        resp.poses = updated_trajectory
        return TrajectoryCheckResponse(resp)

    def traj_index(self, idx, idy, buff=3):
        x = []
        y = []
        for i in range(-buff,buff):
            for j in range(-buff,buff):
                x.append(int(round(idx)+i))
                y.append(int(round(idy)+j))
        return x,y

    def make_array(self,data,height,width):
        output = np.zeros((height,width))
        for i in range(width):
            for j in range(height):
                output[i,j] = data[i+j*width]
        return output



def main():
	#initialize node
	rospy.init_node('obstacle_check')
	try:
		ObstacleCheck()
	except rospy.ROSInterruptException:
		pass


if __name__ == '__main__':
	main()
