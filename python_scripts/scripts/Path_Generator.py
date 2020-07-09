from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib import cm
from sklearn import mixture
# from IPython.display import display
from scipy.stats import multivariate_normal
import numpy as np
import math
import os
import GPy as GPy
import dubins
import time
from itertools import chain
# import glog as log
import logging as log
# import gpmodel_library as gp_lib
# from continuous_traj import continuous_traj

from Environment import *
from Evaluation import *
from GPModel import *
from MCTS import *
from Robot import *

class Path_Generator:
    '''The Path_Generator class which creates naive point-to-point straightline paths'''
    
    def __init__(self, frontier_size, horizon_length, turning_radius, sample_step, extent):
        '''
        frontier_size (int) the number of points on the frontier we should consider for navigation
        horizon_length (float) distance between the vehicle and the horizon to consider
        turning_radius (float) the feasible turning radius for the vehicle
        sample_step (float) the unit length along the path from which to draw a sample
        '''

        self.fs = frontier_size
        self.hl = horizon_length
        self.tr = turning_radius
        self.ss = sample_step
        self.extent = extent

        # Global variables
        self.goals = [] #The frontier coordinates
        self.samples = {} #The sample points which form the paths
        self.cp = (0,0,0) #The current pose of the vehicle
        
    def generate_frontier_points(self):
        '''From the frontier_size and horizon_length, generate the frontier points to goal'''
        angle = np.linspace(-2.35,2.35,self.fs) #fix the possibilities to 75% of the unit circle, ignoring points directly behind the vehicle
        goals = [(self.hl*np.cos(self.cp[2]+a)+self.cp[0], self.hl*np.sin(self.cp[2]+a)+self.cp[1], self.cp[2]+a) for a in angle]
        self.goals = goals#[coordinate for coordinate in goals if coordinate[0] < self.extent[1] and coordinate[0] > self.extent[0] and coordinate[1] < self.extent[3] and coordinate[1] > self.extent[2]]
        return self.goals
        
    def make_sample_paths(self):
        '''Connect the current_pose to the goal places'''
        cp = np.array(self.cp)
        coords = {}
        for i,goal in enumerate(self.goals):
            g = np.array(goal)
            distance = np.sqrt((cp[0]-g[0])**2 + (cp[1]-g[1])**2)
            samples = int(round(distance/self.ss))
            
            # Don't include the start location but do include the end point
            for j in range(0,samples):
                x = cp[0]+((j+1)*self.ss)*np.cos(g[2])
                y = cp[1]+((j+1)*self.ss)*np.sin(g[2])
                a = g[2]
                try: 
                    coords[i].append((x,y,a))
                except:
                    coords[i] = []
                    coords[i].append((x,y,a))
        self.samples = coords
        return coords
    
    def get_path_set(self, current_pose):
        '''Primary interface for getting list of path sample points for evaluation'''
        self.cp = current_pose
        self.generate_frontier_points()
        paths = self.make_sample_paths()
        return paths
    
    def get_frontier_points(self):
        return self.goals
    
    def get_sample_points(self):
        return self.samples            

class Dubins_Path_Generator(Path_Generator):
    '''
    The Dubins_Path_Generator class, which inherits from the Path_Generator class. Replaces the make_sample_paths
    method with paths generated using the dubins library
    '''
        
    def make_sample_paths(self):
        '''Connect the current_pose to the goal places'''
        coords = {}
        for i,goal in enumerate(self.goals):
            g = (goal[0],goal[1],self.cp[2])
            path = dubins.shortest_path(self.cp, goal, self.tr)
            configurations, _ = path.sample_many(self.ss)
            coords[i] = [config for config in configurations if config[0] > self.extent[0] and config[0] < self.extent[1] and config[1] > self.extent[2] and config[1] < self.extent[3]]
        
        self.samples = coords
        return coords

class Dubins_EqualPath_Generator(Path_Generator):
    '''
    The Dubins_EqualPath_Generator class which inherits from Path_Generator. Modifies Dubin Curve paths so that all
    options have an equal number of sampling points
    '''
    def make_sample_paths(self):
        '''Connect the current_pose to the goal places'''
        coords = {}
        true_coords = {}
        for i,goal in enumerate(self.goals):
            g = (goal[0],goal[1],self.cp[2])
            path = dubins.shortest_path(self.cp, goal, self.tr)
            configurations, _ = path.sample_many(self.ss)
            true_coords[i], _ = path.sample_many(self.ss/5)
            coords[i] = [config for config in configurations if config[0] > self.extent[0] and config[0] < self.extent[1] and config[1] > self.extent[2] and config[1] < self.extent[3] ]
        
        # find the "shortest" path in sample space
        current_min = 1000
        for key,path in coords.items():
            if len(path) < current_min and len(path) > 1:
                current_min = len(path)
        
        # limit all paths to the shortest path in sample space
        # NOTE! for edge cases nar borders, this limits the paths significantly
        for key,path in coords.items():
            if len(path) > current_min:
                path = path[0:current_min]
                coords[key]=path

        for key,path in true_coords.items():
            ftemp = []
            for c in path:
                if(len(coords[key])>0):
                    if c[0] == coords[key][-1][0] and c[1] == coords[key][-1][1]:
                        ftemp.append(c)
                        break
                    else:
                        ftemp.append(c)
            true_coords[key] = ftemp
        return coords, true_coords

    # def make_sample_paths(self):
    #     '''Connect the current_pose to the goal places'''
    #     coords = {}
    #     true_coords = {}
    #     for i, goal in enumerate(self.goals):
    #         g = (goal[0],goal[1],self.cp[2])
    #         path = dubins.shortest_path(self.cp, goal, self.tr)
    #         configurations, _ = path.sample_many(self.ss)
    #         true_coords[i], _ = path.sample_many(self.ss/5)
    #         coords[i] = [config for config in configurations if config[0] > self.extent[0] and config[0] < self.extent[1] and config[1] > self.extent[2] and config[1] < self.extent[3]]
        
    #     # find the "shortest" path in sample space
    #     current_min = 1000
    #     for key,path in coords.items():
    #         if len(path) < current_min and len(path) > 1:
    #             current_min = len(path)
        
    #     # limit all paths to the shortest path in sample space
    #     # NOTE! for edge cases nar borders, this limits the paths significantly
    #     for key,path in coords.items():
    #         if len(path) > current_min:
    #             path = path[0:current_min]
    #             coords[key]=path

    #     for key,path in true_coords.items():
    #         ftemp = []
    #         for c in path:
    #             if c[0] == coords[key][-1][0] and c[1] == coords[key][-1][1]:
    #                 ftemp.append(c)
    #                 break
    #             else:
    #                 ftemp.append(c)
    #         true_path[key] = ftemp
    #     return coords , true_coords
    #     # , true_coords

    def get_path_set(self, current_pose):
        '''Primary interface for getting list of path sample points for evaluation
        Input:
            current_pose (tuple of x, y, z, a which are floats) current location of the robot in world coordinates
        Output:
            paths (dictionary of frontier keys and sample points)
        '''
        self.cp = current_pose
        self.generate_frontier_points()
        paths, true_paths = self.make_sample_paths()
        return paths, true_paths