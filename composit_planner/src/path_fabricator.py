#!/usr/bin/python

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
from sensor_msgs.msg import *
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
        # self.check_traj = rospy.ServiceProxy('query_obstacles', TrajectoryCheck)

        self.cost_srv = rospy.ServiceProxy('obstacle_map', GetCostMap)
        self.path_pub = rospy.Publisher('/path_options', PointCloud, queue_size=1)
        self.tf_listener = TransformListener()

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

        # Get the costmap
        map_resp = self.cost_srv(GetCostMapRequest())
        current_map = map_resp.map
        data = self.make_array(current_map.data, current_map.info.height, current_map.info.width)

        # Check the poses
        for path in paths:
        	idx = int(round(([x[0] for x in path] - current_map.origin.x)/current_map.info.resolution))
        	idy = int(round(([x[1] for x in path] - current_map.origin.y)/current_map.info.resolution))

        	cost = data[idx,idy]
        	if cost < self.safe_threshold:
        		clear_paths.append(self.make_rosmsg(path))
    	m = PointCloud()
    	m.header.frame_if = 'world'
		m.header.stamp = rospy.Time(0)
    	for poly in clear_paths:
    		m.points.append(poly.polygon.points)
    		values = np.ones(np.size(poly.polygon.points))
    		m.channels.append(values)
		self.path_pub.publish(m)
        return clear_paths

    def make_rosmsg(self,path):
    	pub_path = []
    	self.tf.getLatestCommonTime("/world", "/map")
        for coord in path:
        	p = Pose()
        	p.position.x = coord[0]
        	p.position.y = coord[1]
        	p.orientation = quaternion_from_euler(0,0,coord[2])
        	coord = self.tf_listener.transformPose('/world', p)
            c = Point32()
            c.x = coord.position.x[0]
            c.y = coord.position.y[1]
            q = coord.orientation
        	angle = euler_from_quaternion((q.x, q.y, q.z, q.w))
            c.z = angle[2] # keep heading information
            pub_path.append(c)
        pte = PolygonStamped()
        pte.header.frame_id = 'world'
        pte.header.stamp = rospy.Time(0)
        pte.polygon.points = pub_path
        return pte

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
    	''' Gets the pose of the robot, in body frame and converts to pose in the map frame'''
    	self.tf.getLatestCommonTime("/world", "/map")
    	t = PoseStamped()
    	t.header.frame_id = '/world'
    	t.header.stamp = rospy.Time(0)
    	t.pose.position.x = msg.x
    	t.pose.position.y = msg.y
    	t.pose.orientation = quaternion_from_euler(0,0,msg.z)

        p = self.tf_listener.transformPose('/map', t)
        q = p.orientation
        angle = euler_from_quaternion((q.x, q.y, q.z, q.w))
        self.map_frame = (p.position.x, p.position.y, angle[2])

        return self.map_frame #for checking obstacles


if __name__ == '__main__':
    try:
        ROS_Path_Generator()
    except rospy.ROSInterruptException:
        rospy.loginfo("Path generator finished")