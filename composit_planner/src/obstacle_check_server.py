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
        self.safe_threshold = rospy.get_param('cost_limit', 80.)
        self.turn_threshold = rospy.get_param('turn_limit', 50.)

        # subscribe to transforms
        self.tf = TransformListener()
        # create service to query cost map from server
        self.world = rospy.ServiceProxy('obstacle_map', GetCostMap)
        # create service to create adjusted safe trajectories
        self.srv = rospy.Service('query_obstacles', TrajectoryCheck, self.check_obstacles)
        # spin until interrupt
        rospy.spin()

    def check_obstacles(self, req):
        ''' Given a list of Paths, check agains the current map for truncating
        Input: list of nav_msg/Path 
        Output: list of nav_msg/Path
        '''
        safe_paths = []
        # get the costmap of interest
	req = GetCostMapRequest()
	req.type = 'inflated'
        map_resp = self.world(req)
        current_map = map_resp.map
        # reshape the array to be a matrix for querying
        data = self.make_array(current_map.data, current_map.info.height, current_map.info.width)
        # np.save('../cost_map', data)
        # get the most recent transforms
        self.tf.getLatestCommonTime("/world", "/map")
        # self.tf.getLatestCommonTime("/world", "/body")
        self.tf.getLatestCommonTime("/world", "/body_flat")
        # walk through the paths to check
        for path in req.query_path:
            updated_trajectory = []
            last_safe = 0
            # transform trajectory coordinates into indices of the matrix to query
            true_coords = path.poses
            for i,c in enumerate(true_coords):
                p = self.tf.transformPose('/map', c) # c is in world coordinates(?)
                #p = self.tf.transformPose('/map', p)
                idx = int(round((p.pose.position.x-current_map.info.origin.position.x)/current_map.info.resolution))
                idy = int(round((p.pose.position.y-current_map.info.origin.position.y)/current_map.info.resolution))

                if data[idx,idy] <= self.turn_threshold:
                    updated_trajectory.append(c)
                    last_safe = i
                elif data[idx,idy] < self.safe_threshold:
                    updated_trajectory.append(c)
                elif data[idx,idy] >= self.safe_threshold:
                    break
            updated_trajectory = updated_trajectory[0:last_safe]
            if len(updated_trajectory) != 0:
                #create the response message
                resp = Path()
                resp.header.stamp = rospy.Time(0)
                resp.header.frame_id = 'world'
                resp.poses = updated_trajectory
                safe_paths.append(resp)
        return TrajectoryCheckResponse(safe_paths)

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
