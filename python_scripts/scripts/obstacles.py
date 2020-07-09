# !/usr/bin/python

'''
This allows for access to the obstacle class which can populate a given environment. Generally is a wrapper for the shapely library.

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''

import numpy as np
import shapely
from shapely.geometry import Polygon, Point
import matplotlib
import matplotlib.pyplot as plt

class FreeWorld():
	def __init__(self):
		self.obstacles = []

	def in_obstacle(self, point, buff=0.1):
		return False

	def get_obstacles(self):
		return []

class BlockWorld():
	'''
	This class allows for the initialization of n blocks of mxp dimensions in the environment of interest either placed randomly or at provided locations.
	'''

	def __init__(self, extent, num_blocks=1, dim_blocks=(2.,2.), centers=None):
		'''
		Input: extent (list of floats, range of the environment); num_blocks (int, number of blocks to generate in the world), dim_blocks (tuple of floats, size of the obstacle to be created), centers (tuple of list of tuples of floats, locations to place bocks; if none, they will be placed randomly)
		'''

		self.extent = extent
		self.num_blocks = num_blocks
		self.dim_blocks = dim_blocks

		if centers is None:
			self.centers = []
			for i in range(0,num_blocks):
				self.centers.append(((extent[1]-extent[0])*np.random.random_sample()+extent[0], (extent[3]-extent[2])*np.random.random_sample()+extent[2]))
		else:
			self.centers = centers

		self.obstacles = []
		self.points = []
		for center in self.centers:
			points = [(center[0]+dim_blocks[0]/2, center[1]+dim_blocks[1]/2),
			          (center[0]+dim_blocks[0]/2, center[1]-dim_blocks[1]/2),
			          (center[0]-dim_blocks[0]/2, center[1]-dim_blocks[1]/2),
			          (center[0]-dim_blocks[0]/2, center[1]+dim_blocks[1]/2)]
			self.obstacles.append(Polygon(points))
			self.points.append(points)


	def in_obstacle(self, point, buff=0.1):
		'''
		Checks through the obstacles to determine if a point of interest is in a shape, and returns True or False depending on the status
		'''
		# point = Point(point)
		# for poly in self.obstacles:
		# 	temp = poly.buffer(buff, cap_style=3)
		# 	if point.within(temp):
		# 		return True
		# 	elif temp.contains(point):
		# 		return True
		# return False

		for obs in self.points:
			if point[0] > obs[2][0]-buff and point[0] < obs[0][0]+buff:
				if point[1] > obs[1][1]-buff and point[1] < obs[0][1]+buff:
					return True
		return False




	def draw_obstacles(self):
		'''
		Visualizer for the obstacles
		'''
		plt.figure()
		plt.axis(self.extent)
		for obs in self.obstacles:
			x,y = obs.exterior.xy
			plt.plot(x,y)
		plt.show()

	def get_obstacles(self):
		return self.obstacles

	def get_centers(self):
		return self.centers

	def get_coordinates(self):
		coords = []
		for obs in self.obstacles:
			x,y = obs.exterior.xy
			coords.append((x,y))
		return coords

class BugTrap(BlockWorld):
	'''
	Class to generate a typical bugtrap environment
	'''

	def __init__(self, extent, opening_location, opening_size, channel_size = 0.5, width = 3., orientation='left'):
		self.extent = extent
		self.opening_location = opening_location
		self.opening_size = opening_size
		self.orientation = orientation

		if orientation == 'left':
			points = [(opening_location[0], opening_location[1]+opening_size/2),
			          (opening_location[0], opening_location[1]+opening_size/2+channel_size),
			          (opening_location[0]+width+channel_size, opening_location[1]+opening_size/2+channel_size),
			          (opening_location[0]+width+channel_size, opening_location[1]-opening_size/2-channel_size),
			          (opening_location[0], opening_location[1]-opening_size/2-channel_size),
			          (opening_location[0], opening_location[1]-opening_size/2),
			          (opening_location[0]+width, opening_location[1]-opening_size/2),
			          (opening_location[0]+width, opening_location[1]+opening_size/2),
			          (opening_location[0], opening_location[1]+opening_size/2)]
			self.inner_xlim = [opening_location[0], opening_location[0]+width]
			self.inner_ylim = [opening_location[1]+opening_size/2, opening_location[1]-opening_size/2]
			self.outer_xlim = [opening_location[0], opening_location[0]+width+channel_size]
			self.outer_ylim = [opening_location[1]+opening_size/2+channel_size, opening_location[1]-opening_size/2-channel_size]
		elif orientation == 'right':
			points = [(opening_location[0], opening_location[1]+opening_size/2),
			          (opening_location[0], opening_location[1]+opening_size/2+channel_size),
			          (opening_location[0]-width-channel_size, opening_location[1]+opening_size/2+channel_size),
			          (opening_location[0]-width-channel_size, opening_location[1]-opening_size/2-channel_size),
			          (opening_location[0], opening_location[1]-opening_size/2-channel_size),
			          (opening_location[0], opening_location[1]-opening_size/2),
			          (opening_location[0]-width, opening_location[1]-opening_size/2),
			          (opening_location[0]-width, opening_location[1]+opening_size/2),
			          (opening_location[0], opening_location[1]+opening_size/2)]
			self.inner_xlim = [opening_location[0]-width, opening_location[0]]
			self.inner_ylim = [opening_location[1]+opening_size/2, opening_location[1]-opening_size/2]
			self.outer_xlim = [opening_location[0]-width-channel_size, opening_location[0]]
			self.outer_ylim = [opening_location[1]+opening_size/2+channel_size, opening_location[1]-opening_size/2-channel_size]
		self.obstacles = [Polygon(points)]

	def in_obstacle(self, point, buff=0.1):

		if self.orientation == 'left':
			if point[0] > self.outer_xlim[0]-buff and point[0] < self.outer_xlim[1]+buff:
				if point[1] < self.outer_ylim[0]+buff and point[1] > self.outer_ylim[1]-buff:
					if point[0] >= self.inner_xlim[0]-buff and point[0] <= self.inner_xlim[1]-buff:
						if point[1] <= self.inner_ylim[0]-buff and point[1] >= self.inner_ylim[1]+buff:
							return False
					return True
			return False
		elif self.orientation == 'right':
			if point[0] > self.outer_xlim[0]-buff and point[0] < self.outer_xlim[1]+buff:
				if point[1] < self.outer_ylim[0]+buff and point[1] > self.outer_ylim[1]-buff:
					if point[0] >= self.inner_xlim[0]+buff and point[0] <= self.inner_xlim[1]+buff:
						if point[1] <= self.inner_ylim[0]-buff and point[1] >= self.inner_ylim[1]+buff:
							return False
					return True
			return False

class ChannelWorld(BlockWorld):
	'''
	Class to generate an environment that is divided with a single hallway for entering
	'''

	def __init__(self, extent, opening_location, opening_size, wall_thickness):
		self.extent = extent
		self.opening_location = opening_location
		self.opening_size = opening_size
		self.wall_thickness = wall_thickness

		top_wall = [(opening_location[0]-wall_thickness/2, opening_location[1]+opening_size/2),
		            (opening_location[0]-wall_thickness/2, extent[3]),
		            (opening_location[0] + wall_thickness/2, extent[3]),
		            (opening_location[0] + wall_thickness/2, opening_location[1]+opening_size/2)]

		bottom_wall = [(opening_location[0]-wall_thickness/2, opening_location[1]-opening_size/2),
					   (opening_location[0]-wall_thickness/2, extent[2]),
		               (opening_location[0] + wall_thickness/2, extent[2]),
		               (opening_location[0] + wall_thickness/2, opening_location[1]-opening_size/2)]

		points = [top_wall, bottom_wall]

		self.obstacles = [Polygon(ob) for ob in points]
		self.xlim = [opening_location[0]-wall_thickness/2, opening_location[0]+wall_thickness/2]
		self.ylim = [opening_location[1]-opening_size/2, opening_location[1]+opening_size/2]

	def in_obstacle(self, point, buff=0.1):
		if point[0] > self.xlim[0]-buff and point[0] < self.xlim[1]+buff:
			if point[1] < self.ylim[0]+buff or point[1] > self.ylim[1]-buff:
				return True
		return False


if __name__ == '__main__':
	# bw = BlockWorld([0.,10.,0.,10.], 1, (8.,3.), [(5,5)])
	# bw.draw_obstacles()
	# print bw.in_obstacle((4,5))

	bt = BugTrap([0., 10., 0., 10.], (3,3), 2., 0.5, 2., 'left')
	bt.draw_obstacles()
	print bt.in_obstacle((3,2.4))

	# cw = ChannelWorld([0., 10., 0., 10.], (3,3), 3, 2)
	# cw.draw_obstacles()
	# print cw.in_obstacle((3,2))