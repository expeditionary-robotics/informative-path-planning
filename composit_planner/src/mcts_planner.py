#!/usr/bin/python

# Copyright 2018 Massachusetts Institute of Technology
# Sytem includes
import numpy as np
import scipy as sp
import math
import os
import threading

# ROS includes
import rospy
from geometry_msgs.msg import *
from nav_msgs.msg import * 
from sensor_msgs.msg import *
from std_msgs.msg import *
from composit_planner.srv import *
from composit_planner.msg import *
from tf.transformations import quaternion_from_euler, euler_from_quaternion

# GPy and GP includes
import GPy as GPy
from gpmodel_library import GPModel, OnlineGPModel
from scipy.stats import multivariate_normal
from scipy.stats import norm

# Libraries includes
import paths_library as paths_lib
import aq_library as aq_lib
import mcts_library as mcts_lib

'''
This node runs the MCTS system in order to select trajectories
'''

class Planner:
    def __init__(self):
        ''' Get ROS parameters '''
        # The kernel hyperparmeters for the robot's GP model 
        self.variance = float(rospy.get_param('model_variance','100'))
        self.lengthscale = float(rospy.get_param('model_lengthscale','0.1'))
        self.noise = float(rospy.get_param('model_noise', '0.0001'))
        self.x1min = float(rospy.get_param('xmin', '0'))
        self.x1max = float(rospy.get_param('xmax', '10'))
        self.x2min = float(rospy.get_param('ymin', '0'))
        self.x2max = float(rospy.get_param('ymax', '10'))

        # Get planning parameters like number of replanning steps
        self.visualize_rate = rospy.get_param('visualize_rate', 0.1)
        self.reward = rospy.get_param('reward_func','mes')
        self.replan_budget = rospy.get_param('replan_budget' ,150)
        self.frontier_size = rospy.get_param('frontier_size',15)
        self.horizon_len = rospy.get_param('horizon_length',1.5)
        self.turn_radius  = rospy.get_param('turning_radius',0.05)
        self.sample_step = rospy.get_param('sample_step',0.5)
        self.rollout_len = rospy.get_param('rollout_length', 5)
        self.tree_type = rospy.get_param('tree_type','dpw_tree')
        self.planner_type = rospy.get_param('planner_type', 'myopic')
        
        # Initialize member variables
        self.current_max = -float("inf")
        self.data_queue = list()
        self.pose_queue = list()
        self.pose = Point32() 
        
        # Initialize the robot's GP model with the initial kernel parameters
        self.GP = OnlineGPModel(ranges = [self.x1min, self.x1max, self.x2min, self.x2max], lengthscale = self.lengthscale, variance = self.variance, noise = self.noise)
       
        # Initialize path generator
        # self.path_generator = paths_lib.ROS_Path_Generator(self.fs, self.hl, self.tr, self.ss)

        # Create mutex for the data queue
        self.data_lock = threading.Lock()

        # Subscriptions to topics and services 
        rospy.wait_for_service('query_obstacles')
        rospy.wait_for_service('query_chemical')
        self.srv_traj = rospy.ServiceProxy('query_obstacles', TrajectoryCheck)
        self.srv_paths = rospy.ServiceProxy('get_paths', PathFromPose)
        self.srv_chem = rospy.ServiceProxy('query_chemical', SimMeasurement)
        self.pose_sub = rospy.Subscriber("/pose", PoseStamped, self.update_pose)
        self.data = rospy.Subscriber("/chem_data", ChemicalSample, self.get_sensordata)
        
        # Publications and service offering 
        self.srv_replan = rospy.Service('replan', RequestReplan, self.replan)
        self.pub = rospy.Publisher('/chem_map', PointCloud, queue_size = 100)
        self.plan_pub = rospy.Publisher("/selected_trajectory", PolygonStamped, queue_size=1)
        
        r = rospy.Rate(self.visualize_rate)
        while not rospy.is_shutdown():
            # Pubish current belief map
            status = self.update_model()  #Updating the model this frequently is costly, but helps visualization 
            self.publish_gpbelief()
            r.sleep()

    def update_model(self):
        ''' Adds all data currently in the data queue into the GP model and clears the data queue. Threadsafe. 
        Input: None
        Output: Boolean success. ''' 
        try:
            # Cannot update model if no data has been collected
            if len(self.data_queue) == 0:
                return True 

            # Aquire the data lock
            self.data_lock.acquire()

            # Add all current observations in the data queue to the current model
            NUM_PTS = len(self.data_queue)
            zobs = np.array([msg.data for msg in self.data_queue]).reshape(NUM_PTS, 1)
            xobs = np.array([[msg.x, msg.y] for msg in self.pose_queue]).reshape(NUM_PTS, 2)

            self.GP.add_data(xobs, zobs)
            rospy.loginfo("Number of sample points in belief model %d", self.GP.zvals.shape[0])
        
            # Delete data from the data_queue
            del self.data_queue[:] 
            del self.pose_queue[:] 

            # Release the data lock
            self.data_lock.release()
            return True

        except ValueError as e:
            print e 
            return False

    def replan(self, _):
        ''' Updates the GP model and generates the next best path. 
        Input: None
        Output: Boolean success service response. ''' 
        status = self.update_model()
        print "Update model status:", status
        # Publish the best plan
        if status is True:
            self.get_plan()
            return RequestReplanResponse(True)
        else:
            return RequestReplanResponse(False)

    def update_pose(self, msg):
        ''' Update the current pose of the robot.
        Input: msg (geometry_msgs/PoseStamped)
        Output: None ''' 
        odom_pose = msg.pose # of type odometry messages
        trans_pose = Point32()
        trans_pose.x = odom_pose.position.x
        trans_pose.y = odom_pose.position.y
        q = odom_pose.orientation
        trans_pose.z = euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
        self.pose = trans_pose

    
    def get_sensordata(self, msg):
        ''' Creates a queue of incoming sample points on the /chem_data topic 
        Input: msg (flat64) checmical data at current pose
        '''
        self.data_lock.acquire()
        self.data_queue.append(msg)
        self.pose_queue.append(self.pose)
        self.data_lock.release()    
    
    def publish_gpbelief(self):
        ''' Publishes the current GP belief as a point cloud for visualization. 
        Input: None
        Output: msg (sensor_msgs/PointCloud) point cloud centered at current pose '''

    	rospy.loginfo("Publishing GP belief with pose:")
    	print self.pose.x
    	print self.pose.y
        # Aquire the data lock
        self.data_lock.acquire()

        # Generate a set of observations from robot model with which to make contour plots
        grid_size = 8.0 # grid size in meters
        num_pts = 100 # number of points to visaulzie in grid (num_pts x num_pts)
        x1max = self.pose.x + grid_size / 2.0
        x1min = self.pose.x - grid_size / 2.0
        x2max = self.pose.y + grid_size / 2.0
        x2min = self.pose.y - grid_size / 2.0

        x1 = np.linspace(x1min, x1max, num_pts)
        x2 = np.linspace(x2min, x2max, num_pts)
        x1, x2 = np.meshgrid(x1, x2, sparse = False, indexing = 'xy') # dimension: NUM_PTS x NUM_PTS       
        data = np.vstack([x1.ravel(), x2.ravel()]).T

        # Get GP predictions across grid
        if self.GP.xvals is not None:
            observations, var = self.GP.predict_value(data)

            # Scale obesrvations between the 10th and 90th percentile value
            max_val = norm.ppf(q = 0.90, loc = 0.0, scale = np.sqrt(self.GP.variance))
            min_val = norm.ppf(q = 0.10, loc = 0.0, scale = np.sqrt(self.GP.variance))

            # Define lambda for transforming from observation to 0-255 range
            if max_val == min_val and max_val == 0.00: 
                topixel = lambda val: 0.0
            else:
                topixel = lambda val: int((val - min_val) / (max_val - min_val) * 255.0)

        # Create the point cloud message
        msg = PointCloud()
        msg.header.frame_id = 'map' # Global frame
        val = ChannelFloat32()
        val.name = 'intensity'
        for i, d in enumerate(data):
            pt = geometry_msgs.msg.Point32()
            pt.x = data[i, 0]
            pt.y = data[i, 1]
            pt.z = 1.0
            msg.points.append(pt)

            # If no data, just publish the average value
            if self.GP.xvals is None:
                val.values.append(255./2.)
            else:
                val.values.append(topixel(observations[i, :]))
        msg.channels.append(val)
        self.pub.publish(msg)

        # Release the data lock
        self.data_lock.release()
    
    def choose_myopic_trajectory(self, eval_value):
        # Generate paths (will be obstacle checked against current map)
        clear_paths = self.srv_paths(PathFromPoseRequest(self.pose))
        clear_paths = clear_paths.safe_paths
        #Now, select the path with the highest potential reward
        path_selector = {}
        for i, path in enumerate(clear_paths):
            if len(path.polygon.points) != 0:
                # TODO: need to keep an updated discrete time for the UCB reward
                path_selector[i] = eval_value.predict_value(self.GP, path.polygon.points)
            else:
                path_selector[i] = -float("inf")

        best_key = np.random.choice([key for key in path_selector.keys() if path_selector[key] == max(path_selector.values())])
        return clear_paths[best_key], path_selector[best_key]


    def get_plan(self):
        # Aquire the data lock (no new data can be added to the GP model during planning)
        self.data_lock.acquire()

        # Generate object to calculate reward from list of geometry_msgs/Pose
        eval_value = aq_lib.GetValue(self.reward)

        if self.planner_type == 'myopic':
            if self.pose is not None:
                try:
                    best_path, value = self.choose_myopic_trajectory(eval_value)
                    controller_path = self.strip_angle(best_path)
                    self.plan_pub.publish(controller_path) #send the trajectory to move base
                except:
                    print 'ATTENTION HUMAN! I AM STUCK. SAVE MEEEE'
            else:
                pass
        else:
            if self.pose is not None:
                try:
                    # TODO: set time
                    mcts = mcts_lib.cMCTS(self.GP, self.pose, self.replan_budget, self.rollout_len, self.srv_paths, eval_value, time = 0, tree_type = self.tree_type)
                    # TODO: set time
                    best_path, value = mcts.choose_trajectory(t = 0)
                    controller_path = self.strip_angle(best_path)
                    self.plan_pub.publish(controller_path) #send the trajectory to move base
                except:
                    print 'PLANNER FAILED'
            else:
                pass

            #TODO rewrite the MCTS library in order to support the type of planning we want to do
            # belief_snapshot = None #placeholder
            # Get planning parameters like number of replanning steps
        
            # sampling_path, best_path, best_val, all_paths, all_values, self.max_locs, self.max_val = mcts.choose_trajectory(t=None)
            # self.plan_pub.publish(best_path)

        # Release the data lock
        self.data_lock.release()

    def strip_angle(self, l):
        clear = []
        for elem in l.polygon.points:
            elem.z = 0
            clear.append(elem)
        l.polygon.points = clear
        return l



if __name__ == '__main__':
    rospy.init_node('mcts_planner')
    try:
        Planner()
    except rospy.ROSInterruptException:
        exit(0)
