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
from tf.transformations import euler_from_quaternion


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
        self.allow_backup = rospy.get_param('allow_to_backup', True)


        self.map = None
        self.map_data = None
        self.path = None
        self.pose = None

        # subscribe to costmap and trajectory
        # self.map_sub = rospy.Subscriber('/costmap',OccupancyGrid, self.handle_map)
        self.traj_sub = rospy.Subscriber('/trajectory/current', PolygonStamped, self.handle_trajectory, queue_size=1)
        self.cost_srv = rospy.ServiceProxy('obstacle_map', GetCostMap)
        self.pose_sub = rospy.Subscriber("/pose", PoseStamped, self.handle_pose, queue_size=1)


        # publish to generate new plan
        # self.replan = rospy.ServiceProxy('replan', RequestReplan)

        #create polygon object to kill current trajectory
        self.path_pub = rospy.Publisher('/trajectory/current', PolygonStamped,
                                        queue_size=1)

        #run the node at a certain rate to check things
        r = rospy.Rate(30)
        while not rospy.is_shutdown():
            # Pubish current belief map
            self.check_trajectory()
            r.sleep()

    def handle_trajectory(self, msg):
        coordinates = []
        for coord in msg.polygon.points:
            coordinates.append([coord.x,coord.y])
        self.path = coordinates

    def check_trajectory(self):
	req = GetCostMapRequest()
	req.type = 'inflated'
        map_resp = self.cost_srv(req)
        current_map = map_resp.map
        data = self.make_array(current_map.data, current_map.info.height, current_map.info.width)

        if self.path is not None and len(self.path) > 0:
            idx = [int(round((x[0]-current_map.info.origin.position.x)/current_map.info.resolution)) for x in self.path]
            idy = [int(round((x[1]-current_map.info.origin.position.y)/current_map.info.resolution)) for x in self.path]
            try:
                # print data[idy,idx]
                cost_vals = [k for k in data[idy,idx] if k>=0.]
                cost = np.sum(cost_vals)
            except:
                cost = 0
                for m, n in zip(idx, idy):
                    try:
                        if data[n,m] >= 0:
                            cost += data[n,m]
                    except:
                        break
            if cost > self.safe_threshold:
                abort_mission = PolygonStamped()
                abort_mission.header.frame_id = 'world'
                abort_mission.header.stamp = rospy.Time(0)
                abort_mission.polygon.points = [self.pose, self.pose, self.pose]
                self.path_pub.publish(abort_mission)
                

    def make_array(self,data,height,width):
        return np.array(data).reshape((height,width),order='C')

    def handle_pose(self, msg):
        current = Point32()
        current.x = msg.pose.position.x
        current.y = msg.pose.position.y
        q = msg.pose.orientation
        current.z = euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
        self.pose = current


def main():
	#initialize node
	rospy.init_node('hindbrain')
	try:
		Hindbrain()
	except rospy.ROSInterruptException:
		pass


if __name__ == '__main__':
	main()