#!/usr/bin/python

# Copyright 2018 Massachusetts Institute of Technology

# System includes
import numpy as np
import os
import sys
import matplotlib.pyplot as plt 

'''
This script contains useful analysis scripts for stored data from a simulation or real trial.
In particular, there is a function to visualize a .npz file of the true chemical distribution
and scripts to process rosbag data.

TODO:
	* Add scripts for processing rosbag data
	* Improve user interface for this system
'''

def visualize_true_world(world_file, MIN_VALUE=-25.0, MAX_VALUE=25.0):
	'''
	A function to create a plot of the chemical belief object from saved data
	Inputs:
		world_file (string) filepath and name of data to load
		MIN_COLOR (float) plotting value to make the minimum of the colorbar
		MAX_COLOR (float) plotting value to make the maximum of the colorbar 
	'''
	data = np.load(world_file)
	plt.contourf(data['x1'], data['x2'], data['z'], cmap='viridis', vmin=MIN_VALUE, vmax=MAX_VALUE, levels=np.linspace(MIN_VALUE,MAX_VALUE,15))
	plt.show()

if __name__ == '__main__':
	file_path = sys.argv[1]
	visualize_true_world(file_path)
