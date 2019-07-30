# !/usr/bin/python

'''
This allows for access to the obstacle class which can populate a given shapely environment.
Generally is a wrapper for the shapely library.

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''

from shapely.geometry import LineString
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from plumes.src.generate_metric_environment import *
import numpy as np

def test_trajectory():
	''' This function will create a known static world and set of trajectories
	with various ray casting relationships to check '''
	free_world = World([0, 10, 0, 10])
	free_world.add_blocks(num=3, dim=(2, 2), centers=[(3, 3), (5, 5), (7, 7)])

	a = [(1, 1), (9, 9)]
	b = [(1, 1), (9, 1)]
	c = [(1, 1), (1, 9)]
	d = [(1, 9), (9, 1)]
	e = [(0, 0), (0, 0)]

	assert free_world.safe_trajectory(a) is False, 'False Safety Dectected'
	assert free_world.safe_trajectory(b) is True, 'False Collision Dectected'
	assert free_world.safe_trajectory(c) is True, 'False Collision Dectected'
	assert free_world.safe_trajectory(d) is False, 'False Safety Dectected'
	assert free_world.safe_trajectory(e) is False, 'False Safety Detected'

if __name__ == '__main__':
	test_trajectory()
