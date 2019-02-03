#!/usr/bin/python

'''
This library can be used to access the multiple ways in which path sets can be generated for
the simulated vehicle in the PLUMES framework.

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''

import copy
import dubins
import numpy as np
import rospy
from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from std_msgs.msg import *
from composit_planner.srv import *
from composit_planner.msg import *
from tf import TransformListener


class ROS_Path_Generator(object):
    ''' Uses dubins library to generate valid, safe paths for the vehicle.
    '''
    def __init__(self):
        ''' Initialize a path generator
        Input:
            frontier_size (int) the number of points on the frontier to consider for navigation
            horizon_length (float) distance between the vehicle and the horizon to consider
            turning_radius (float) the feasible turning radius for the vehicle
            sample_step (float) the unit length along the path from which to draw a sample
        '''
        rospy.init_node("path_generator")

        # the parameters for the dubin trajectory
        self.fs = rospy.get_param('frontier_size', 15)
        self.hl = rospy.get_param('horizon_length', 1.5)
        self.tr = rospy.get_param('turning_radius', 0.05)
        self.ss = rospy.get_param('sample_step', 0.5)
        self.ang = rospy.get_param('frontier_angle_range', np.pi/4)
        self.safe_threshold = rospy.get_param('cost_limit', 50.)
        self.unknown_threshold = rospy.get_param('unknown_limit', -2.0)
        self.make_paths_behind = rospy.get_param('make_paths_behind', False)
        self.make_stay_path = rospy.get_param('allow_to_stay', True)
        self.visualize_paths = rospy.get_param('visualize_path_generation', True)

        # Global variables
        self.goals = [] #The frontier coordinates
        self.samples = {} #The sample points which form the paths
        self.cp = (0, 0, 0) #The current pose of the vehicle

        # Services and Publishers
        self.srv_path = rospy.Service('get_paths', PathFromPose, self.get_path_set)
        self.cost_srv = rospy.ServiceProxy('obstacle_map', GetCostMap)
        self.path_pub = rospy.Publisher('/path_options', PointCloud, queue_size=1)
        self.tf_listener = TransformListener()

        while not rospy.is_shutdown():
            rospy.spin()

    def generate_frontier_points(self):
        '''From the frontier_size and horizon_length, generate the frontier points to goal'''
        angle = np.linspace(-self.ang, self.ang, self.fs)
        goals = []
        for a in angle:
            goals.append(trig_projection(self.cp, self.hl, a))

        #also append a path directly in front of the vehicle
        goals.append(trig_projection(self.cp, self.hl, 0))

        self.goals = goals
        return self.goals

    def make_sample_paths(self):
        '''Connect the current_pose to the goal places'''
        all_paths = []
        for goal in self.goals:
            paths = dubins.shortest_path(self.cp, goal, self.tr)
            true_path, _ = paths.sample_many(self.ss)
            all_paths.append(true_path)

        if self.make_paths_behind is True:
            # make a single path directly behind the vehicle
            behind_goal = trig_projection(self.cp, self.hl*0.80, np.pi)
            paths = dubins.shortest_path((self.cp[0], self.cp[1], self.cp[2]+np.pi),
                                         behind_goal,
                                         self.tr)
            true_path, _ = paths.sample_many(self.ss)
            true_path = [(-100000.,-100000.,-100000.)] + true_path #indicator for the controller
            all_paths.append(true_path)

        if self.make_stay_path is True:
            all_paths.append([self.cp for i in range(0, int(self.hl/self.ss)+1)])

        return all_paths

    def rosify_safe_path(self, paths):
        '''
        Transforms relative coordinates to rosmessage types and checks that they are safe
        '''
        clear_paths = []

        # Get the costmap
        map_resp = self.cost_srv(GetCostMapRequest())
        current_map = map_resp.map
        data = make_array(current_map.data, current_map.info.height, current_map.info.width)

        # Check the poses
        self.viz = []
        for path in paths:
            if path[0][0] > -100000:
                idy = [int(round((x[0]-current_map.info.origin.position.x)/current_map.info.resolution)) for x in path]
                idx = [int(round((x[1]-current_map.info.origin.position.y)/current_map.info.resolution)) for x in path]
            else:
                idy = [int(round((x[0]-current_map.info.origin.position.x)/current_map.info.resolution)) for x in path[1:]]
                idx = [int(round((x[1]-current_map.info.origin.position.y)/current_map.info.resolution)) for x in path[1:]]
            try:
                cost_vals = data[idx, idy]
                cost = np.sum([k for k in cost_vals if k >= 0.])
                unknown_cost = np.sum([k for k in cost_vals if k < 0.])
            except:
                cost = 0
                unknown_cost = 0
                for m, n in zip(idx, idy):
                    try:
                        if data[m, n] >= 0:
                            cost += data[m, n]
                        else:
                            unknown_cost += data[m, n]

                    except:
                        break


            if (cost < self.safe_threshold and unknown_cost > self.unknown_threshold) and len(path) > 0:
                clear_paths.append(self.make_rosmsg(path))

        # Make a debugging message
        if self.visualize_paths is True:
            m = PointCloud()
            m.header.frame_id = 'world'
            m.header.stamp = rospy.Time(0)
            m.points = self.viz
            val = ChannelFloat32()
            val.name = 'path_options'
            val.values = np.ones(np.size(self.viz))
            m.channels.append(val)
            self.path_pub.publish(m)
        # Return all of the polygons to assess, includes header information
        return clear_paths

    def make_rosmsg(self, path):
        ''' Does the ROS message conversionros
        '''
        pub_path = []
        for coord in path:
            pc = Point32()
            pc.x = coord[0]
            pc.y = coord[1]
            if pc.x > -100000: #don't visualize helper indicators
                self.viz.append(pc)
            c = copy.copy(pc)
            c.z = coord[2] # keep heading information
            pub_path.append(c)
        pte = PolygonStamped()
        pte.header.frame_id = 'world'
        pte.header.stamp = rospy.Time(0)
        pte.polygon.points = pub_path
        return pte

    def get_path_set(self, req):
        '''Primary interface for getting list of path sample points for evaluation
        Input:
            current_pose (tuple of x, y, z, a which are floats) current location of the robot in
            world coordinates
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
        self.map_frame = (msg.x, msg.y, msg.z)
        return self.map_frame #for checking obstacles

def make_array(data, height, width):
    ''' Converts occupancy grid vector to useable array'''
    return np.array(data).reshape((height, width), order='C')

def trig_projection(point, step_size, angle):
    ''' Function to perform a path projection'''
    x = step_size*np.cos(point[2]+angle)+point[0]
    y = step_size*np.sin(point[2]+angle)+point[1]
    p = point[2]+angle
    return (x, y, p)


if __name__ == '__main__':
    try:
        ROS_Path_Generator()
    except rospy.ROSInterruptException:
        rospy.loginfo("Path generator finished")
