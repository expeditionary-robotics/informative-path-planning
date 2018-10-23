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

'''
This is a node which keeps a physical map of obstacles in the world, and provides a ROSSERVICE when queried to report whether or not a trajectory is in an obstacle
'''

class Obstacle_Map:
    ''' The Obstacle_Map class, which represents the physical location of barriers and obstacles in a 3D world. Can optionally take in a predetermined map, or can generate a map live from a LIDAR subscription. On query, returns safe trajectories based on current status of the world. 
    '''
    def __init__(self):
        ''' Initialize the environment either from a parameter input or by subscribing to a topic of interest. 
        Input:
        - world_type (string): indicates whether to create a shapely system or an occupancy grid system. Options are: sim_default (shapely FreeWorld), sim_block, sim_bugtrap, sim_channel, live_map
        - simulation world specific domains
        '''

        self.world_type = rospy.get_params('world_type','sim_default')

        if self.world_type == 'sim_default':
            self.world = FreeWorld()
        else if self.world_type == 'sim_block':
            # for sim_block
            self.extent = rospy.get_params('extent',[10.,0.,10.,0.])
            self.num_blocks = rospy.get_params('num_blocks',1)
            self.dim_blocks = rospy.get_params('dim_blocks',[2.,2.])
            self.centers = rospy.get_params('centers',[(5.,5)])
            self.world = BlockWorld(extent, num_blocks, dim_blocks, centers)
        else if self.world_type == 'sim_bug':
            # for sim_bug
            self.extent = rospy.get_params('extent',[10.,0.,10.,0.])
            self.opening_location = rospy.get_params('opening_location', [5.,5.])
            self.opening_size = rospy.get_params('opening_size', 3.)
            self.channel_size = rospy.get_params('channel_size', 0.5)
            self.width = rospy.get_params('width', 3.)
            self.orientation = rospy.get_params('orientation', 'left')
        else if self.world_type == 'sim_channel':
            # for sim_chnnel
            self.extent = rospy.get_params('extent',[10.,0.,10.,0.])
            self.opening_location = rospy.get_params('opening_location', [5.,5.])
            self.opening_size = rospy.get_params('opening_size', 3.)
            self.wall_thickness = rospy.get_params('wall_thickness', 0.5)
        else if self.world_type == 'live_map':
            # subscribe to the map topic output by gmapping
            self.world = rospy.ServiceProxy('dynamic_map', GetMap)
        
        self.srv = rospy.Service('query_obstacles', TrajectoryCheck, self.check_obstacles)
        rospy.spin()

    def check_obstacles(self, req):
        #parse the poses in a trajectory and check if any are in/near obstacles
        #TODO think of a clever matrix-transform way of doing this for speed up
        updated_trajectory = []
        if self.world_type != 'live_map':
            for pose in req.poses:
                coord = [pose.position.x, pose.position.y]
                if self.world_type != 'live_map':
                    # use the obslib stuff to check the trajectory
                    if self.world.in_obstacle(coord):
                        pass
                    else:
                        updated_trajectory.append(pose)
        else:
            current_map = self.world()
            data = np.asarray(current_map.data, dtype=np.int8).reshape(current_map.info.height, current_map.info.width)
            # get points that are obstacles or near obstacles
            # check if the desired trajectory pose is near that pose
            # if so, truncate trajectory to last good point and return
            # if not, the whole thing is safe and can be returned in full

        resp = Path()
        resp.header.stamp = rospy.Time.now()
        resp.poses = updated_trajectory
        return resp



def main():
	#initialize node
	rospy.init_node('physical_map')
	try:
		Obstacle_Map()
	except rospy.ROSInterruptException:
		pass


if __name__ == '__main__':
	main()