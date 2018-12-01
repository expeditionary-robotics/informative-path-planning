#!/usr/bin/python

# Copyright 2018 Massachusetts Institute of Technology

import numpy as np
import math
import os
import threading
import GPy as GPy
from obstacles import *
import rospy
from std_msgs.msg import *
from composit_planner.srv import *
from geometry_msgs.msg import *
from trajectory_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from scipy.ndimage import gaussian_filter, convolve
from tf import TransformListener


'''
This is a node which prevents crashes into nearby obstacles that reveal themselves
'''

class Hindbrain:
    ''' The ObstacleCheck class, which represents the physical location of barriers and obstacles in a 3D world. Can optionally take in a predetermined map, or can generate a map live from a LIDAR subscription. On query, returns safe trajectories based on current status of the world. 
    '''
    def __init__(self):
        ''' Initialize the environment either from a parameter input or by subscribing to a topic of interest. 
        '''

        self.safe_threshold = rospy.get_param('cost_limit', 50.)

        self.map = None
        self.map_data = None
        self.path = None

        # subscribe to costmap and trajectory
        # self.map_sub = rospy.Subscriber('/costmap',OccupancyGrid, self.handle_map)
        self.traj_sub = rospy.Subscriber('/selected_trajectory', PolygonStamped, self.handle_trajectory)
        self.cost_srv = rospy.ServiceProxy('obstacle_map', GetCostMap)

        # publish to generate new plan
        self.replan = rospy.ServiceProxy('replan', RequestReplan)

        #create polygon object to kill current trajectory
        self.path_pub = rospy.Publisher('/trajectory/current', PolygonStamped,
                                        queue_size=1)

        self.data_lock = threading.Lock()

        #run the node at a certain rate to check things
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            # Pubish current belief map
            self.check_trajectory()
            r.sleep()

    def handle_map(self, req):
        pass
        # self.map = self.make_array(req.data, req.info.height, req.info.width)
        # self.map_data = req

    def handle_trajectory(self, msg):
        coordinates = []
        for coord in msg.polygon.points:
            coordinates.append([coord.x,coord.y])
        self.path = coordinates

    def check_trajectory(self):
        map_resp = self.cost_srv(GetCostMapRequest())
        current_map = map_resp.map
        data = self.make_array(current_map.data, current_map.info.height, current_map.info.width)

        if self.path is not None:
            idx = [int(round((x[0]-current_map.info.origin.position.x)/current_map.info.resolution)) for x in self.path]
            idy = [int(round((x[1]-current_map.info.origin.position.y)/current_map.info.resolution)) for x in self.path]
            try:
                cost = np.sum(self.map[idx,idy])
            except:
                cost = 0
                for m, n in zip(idx, idy):
                    try:
                        cost += data[m, n]
                    except:
                        break
            if cost > self.safe_threshold:
                print 'Replanning!'
                abort_mission = PolygonStamped()
                abort_mission.header.frame_id = 'world'
                abort_mission.header.stamp = rospy.Time(0)
                self.path_pub.publish(abort_mission)
                self.replan()
                self.path = None

    def make_array(self,data,height,width):
        return np.array(data).reshape((height,width),order='C')#self.make_array(msg.data, msg.info.height, msg.info.width)

        # output = np.zeros((height,width))
        # for i in range(width):
        #     for j in range(height):
        #         output[i,j] = data[i+j*width]
        # return output



def main():
	#initialize node
	rospy.init_node('hindbrain')
	try:
		Hindbrain()
	except rospy.ROSInterruptException:
		pass


if __name__ == '__main__':
	main()
