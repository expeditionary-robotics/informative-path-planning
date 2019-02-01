#!/usr/bin/env python

import rospy
import actionlib
# from composit_planner.srv import *
# from composit_planner.msg import *
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import *
from tf.transformations import quaternion_from_euler
import time

# create a class that would send waypoints to the car
clss navigation_leader:

	#initialize class
	def __init__(self):
		# class variables
		self.flag = False

		# variables for checking if reached end of navigation
		self.current_x = 0.0 # in world frame
		self.current_y = 0.0
		self.goal_x = 0.0
		self.goal_y = 0.0
		self.reached_goal = True

		# self.current_path_index = -1;
		# paths lists, only for testing
		self.paths = [[Point32(1,0,0),Point32(2, 0, 0)], [Point32(2.5, 0.5, 0.0), Point32(3.0, 0.5, 0.0)]]
		
		# variables for map
		self.map_origin_x = 0.0 # in the world frame
		self.map_origin_y = 0.0
		self.map_resolution = 0.1
		self.map_grid = [] # holds a list of occupancy probabilities. 
					  # inverted row major, starting at bottom left
					  # when looked at from world frame
		self.map_width = 0
		self.map_height = 0


		#init ros
		rospy.init_node('car_nav')

		#publish to trajectory topics
		self.path_pub = rospy.Publisher("/trajectory/current", 
			PolygonStamped,
			queue_size=1)

		#subscribe to position topics.
		# move one meter forward when we get new pose. do it once ####
		self.pose_sub = rospy.Subscriber("/pose_body", 
			PoseStamped, 
			self.pose_cb,
			queue_size=1)

		# subscribe to the map topic - to get the map itself and the information about its origin
		self.map_sub = rospy.Subscriber("/projected_map",
			OccupancyGrid,
			self.map_cb,
			queue_size = 1)

		#while not shut down, keep this node spinning
		while not rospy.is_shutdown():
			rospy.spin()

	def follow_path(self, path_list):
	    # path list is a list of Point32 objects -- which are the waypoints to be followed
	    print('#####sending a sequence')
	    
	    # store the last goal as the goal we are going to 
	    self.goal_x = path_list[-1].x
	    self.goal_y = path_list[-1].y

	    # send the sequence
	    #create a message
	    msg = PolygonStamped()
	    msg.header.stamp = rospy.Time(0)
	    msg.header.frame_id = 'world'
	    msg.polygon.points = path_list

	    #publish this path
	    self.path_pub.publish(msg)


    

	def pose_cb(self, pose_msg):
	    # this callback will handle the following:
            # * calling functions that publish paths
            # * checking if navigation has been completed
            # (will toggle the goal_reached bool variable)

            # update current location in the world frame
			self.current_x = pose_msg.pose.position.x
			self.current_y = pose_msg.pose.position.y

            # if we have rerached the goal, go to the next path
			if self.reached_goal:
				# check if there are remaining paths
				if len(self.paths) > 0:
					# got to the first path, and set the reached variable to false
					self.reached_goal = False;

					# get the path we want to follow
					path = self.paths[0]
					#remove it from the paths list
					self.paths.pop(0)
					# follow the path
					self.follow_path(path)
				
				else:
					# no more goals and reached goal
					print("navigation test finished")

			else:
				# not reached goal yet, check if there now`
				if abs(self.current_x - self.goal_x) < 0.15 and abs(self.current_y - self.goal_y) < 0.15:
					# then we are close enough to the goal and can toggle the reach variable
					self.reached_goal = True
					print('reached goal')


	def map_cb(self, map_msg):
    	# callback to be fired every time a map update is being published
    	# updates the class variables
		self.map_resolution = map_msg.info.resolution
		self.map_width = map_msg.info.width
		self.map_height = map_msg.info.height
		self.map_origin_x = map_msg.info.origin.position.x
		self.map_origin_y = map_msg.info.origin.position.y
		self.map_grid = map_msg.data

		# print("=====\n map origin at x=", self.map_origin_x, " y=", self.map_origin_y)
		# robot_pixel = self.world_to_pixel(self.current_x, self.current_y);
		# print("=====\n robot at pixel x=", robot_pixel[0], " y=",robot_pixel[1])
		# print(" the value of the pixel=", self.get_pixel_value(robot_pixel[0], robot_pixel[1], self.map_grid))

		# robot_world= self.pixel_to_world(robot_pixel[0], robot_pixel[1])
		# print("=====\n should be at x =", self.current_x, "y=", self.current_y)
		# print(" calculated at x=", robot_world[0], "y=", robot_world[1])



	def world_to_pixel(self, robot_x, robot_y):
		# returns a tuple with the (row, col) position of the robot on the map grid
		pixel_x= round((robot_x - self.map_origin_x)/self.map_resolution)
		pixel_y = round((robot_y - self.map_origin_y)/self.map_resolution)
		return (pixel_x, pixel_y)

	def pixel_to_world(self, pixel_x, pixel_y):
		# returns a tuple with the robot (x,y) in the world frame
		robot_x = pixel_x*self.map_resolution + self.map_origin_x
		robot_y = pixel_y*self.map_resolution + self.map_origin_y
		return (robot_x, robot_y)

	def get_pixel_value(self, pixel_x, pixel_y, grid):
		# returns the value stored in the map_grid at that pixel
		index = int(pixel_x + self.map_width*pixel_y) # int because index cannot be float
		return grid[index]


	def get_possible_pixels(self, pixel_x, pixel_y, grid = None):
		# arguments are a pixel in the map_grid
		# returns a list of tuples of possible pixels the robot can progress to
		# these pixels have to be in the map_grid and be vacant (value = 0)

	    if grid == None:
	        grid = self.grid_data

	    # the allowed moves are up down left right
	    # return a list of tuples that only include coordinates of cells that are empty
	    tries = [(pixel_x, pixel_y+1), (pixel_x, pixel_y-1), (pixel_x+1, pixel_y), (pixel_x-1, pixel_y)]
	    finals = []
	    for coord in tries:
	        if coord[0] < self.map_width and coord[0] >= 0 and coord[1] < self.map_height and coord[1] >= 0:
	            if self.get_pixel_value(coord[0], coord[1], grid) in [0,'G']:
	                finals.append(coord)
		return finals


	def directed_graph(self):
		# function that creates a wighted graph
		# this object takes the form of {coord_tuple: {neighbor0_tuple: weight, neighbor1_tuple: weight1}}
		weighted_graph =  {};
		for pixel_x in range(self.map_width):
			for pixel_y in range (self.map_height):
				pixel = (pixel_x,pixel_y)
				# get the surrounding coords
				neighbors = self.get_possible_pixels(pixel_x, pixel_y, self.map_grid)
				# create the dictionary mapping adjacent nodes to their weight
				neighbors_dict = dict()
				for nighbor in neighbors:
					neighbors_dict[neighbor] = 1 # all the weights are 1 in this example
				# add the pixel we are working with, with its adjacent nodes to the weighted_graph
				weighted_graph[pixel] = neighbors_dict
		return weighted_graph

	def dijkstra_to_goal(self, weighted_graph, start, end):
		# arguments:
		'''
		weighted_graph : see function above
		start : a tuple of pixel to start navigating from, of form (x,y) in the map_grid
		end : a tuple of pixel to end navigating at, of form (x,y) in the map_grid

		returns:
		a list of pixels of form [start, ... , pixels in between, ..., end]

		'''
		# set up initial values
		# always need to visit the start
		nodes_to_visit = {start}
		visited_nodes = set()
		distance_from_start = defaultdict(lambda: float("inf"))
		# distance from start to start is 0
		distance_from_start[start] = 0
		tentative_parents = {}

		while nodes_to_visit:
			# the next node should be the one with the smalles weight
			current = min(
				(distance_from_start[node],node) for node in nodes_to_visit)[1]

			#check if end was reached
			if current == end:
				break

			# otherwise
			nodes_to_visit.discard(current)
			visited_nodes.add(current)

			for neighbor, distance in weighted_graph[current].items():
				if neighbor in visited_nodes:
					continue
				neighbor_distance = distance_from_start[current] + distance
				if neighbor_distance < distance_from_start[neighbor]:
					distance_from_start[neighbor] = neighbor_distance
					tentative_parents[neighbor] = current
					nodes_to_visit.add(nighbor)
		
		return self.deconstruct_path(tentative_parents, end)


	def decontruct_path(self, tentative_parents, end):
		if end not in tentative_parents:
			return None

		cursor = end
		path = []
		while cursor:
			path.append(cursor)
			cursor = tentative_parents.get(cursor)

		# reverse the path
		pixel_list = list(reversed(path))
		return pixel_list


	def path_from_pixels_list(self, pixels_list):
		# returns a list of Point32 objects that can be followed by racecar

		path_list = []
		for pixel in pixels_list:
			# convert to world frame
			pass












		# convert to robot frame ('world', no longer pixels)



if __name__ == '__main__':
	#print a simple thing just to show that this script is alive
	print("car_nav_script is ALIVE")

	try:
		print("##### trying the navigation leaader class")
		navigation_leader()
	except rospy.ROSInterruptException:
		rospy.loginfo("#############Navigation test finished.")
