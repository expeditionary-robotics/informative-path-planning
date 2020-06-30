# !/usr/bin/python

from IPython.display import display
import numpy as np
import math
import dubins
import obstacles as obs
import pdb
import matplotlib.pyplot as plt
from random import *


class continuous_traj_sampler:
    def __init__(self, input_limit, sample_number, frontier_size,  horizon_length, total_time, sample_step, extent, obstacle_world=obs.FreeWorld()):
        '''
        Sample number: Number of action samples will generate. 
        '''
        self.limit = input_limit
        self.fs = frontier_size
        self.hl = horizon_length
        self.ss = sample_step
        self.extent = extent
        self.num_action = sample_number
        self.time = total_time #Time horizon for single local trajectory 

        self.goals = []
        self.samples = {}
        self.cp = (0,0,0) #Current pose of the vehicle 

        self.obstacle_world = obstacle_world
        

    def input_sampler(self):
        vmin = self.limit[0]
        vmax = self.limit[1]
        yaw_min = self.limit[2]
        yaw_max = self.limit[3]
        input_dict = {}
        for i in list(range(self.num_action)):
            v_sample = uniform(vmin, vmax)
            yaw_sample = uniform(yaw_min, yaw_max)
            input_dict[i]=(v_sample, yaw_sample)
        return input_dict

    def generate_frontier_points(self):
        '''From the frontier_size and horizon_length, generate the frontier points to goal'''
        angle = np.linspace(-2.35,2.35,self.fs) #fix the possibilities to 75% of the unit circle, ignoring points directly behind the vehicle
        goals = []
        for a in angle:
            x = self.hl*np.cos(self.cp[2]+a)+self.cp[0]
            y = self.hl*np.sin(self.cp[2]+a)+self.cp[1]
            p = self.cp[2]+a
            # if np.linalg.norm([self.cp[0]-x, self.cp[1]-y]) <= self.tr:
            #     pass
            # # elif x > self.extent[1]-3*self.tr or x < self.extent[0]+3*self.tr:
            # #     pass
            # # elif y > self.extent[3]-3*self.tr or y < self.extent[2]+3*self.tr:
            # #     pass
            # else:
            goals.append((x,y,p))
        goals.append(self.cp)
        self.goals = goals
        return self.goals

    def make_single_traj(self, current_pose, vel, ang_vel, sample_period):
        '''
        Single Trajectory(np matrix) is generated with given 
        '''
        coords = {}
        cur_time = 0.0
        iteration = 0

        x_traj = np.array([current_pose[0]])
        y_traj = np.array([current_pose[1]])
        yaw_traj = np.array([current_pose[2]])
        # local_traj = []
        
        while cur_time < self.time:
            iteration = iteration + 1
            x = x_traj[-1]
            y = y_traj[-1]
            yaw = yaw_traj[-1]
            x_traj = np.append(x_traj, x + vel * np.cos(np.deg2rad(yaw)) * sample_period)            
            y_traj = np.append(y_traj, y + vel * np.sin(np.deg2rad(yaw)) * sample_period)
            yaw_traj = np.append( yaw_traj, yaw + ang_vel * sample_period)
            try:
                coords[iteration].append((x,y,yaw))
            except:
                coords[iteration] = []
                coords[iteration].append((x,y,yaw))
            cur_time = cur_time + sample_period

        return coords

    def get_path_set(self, pose):
        self.cp = pose
        cp = np.array(self.cp)
        MPL = {} #Motion Primitive Library
        input_sample = self.input_sampler()
        for i in input_sample:
            # for v and yaw in input_sample:
            v = input_sample[i][0]
            yaw = input_sample[i][1]
            local_traj = self.make_single_traj(cp, v, yaw, self.ss)
            MPL[i]= local_traj
            MPL[i]['v'] = v
            MPL[i]['yaw'] = yaw
        return MPL

    def path_cost(self, path, loc=None):
        ''' Calculate the cost of a path sequence either with respect to path length, or distance from some element in the world (loc)'''
        dist = 0
        if loc is None:
            # cost will be path length
            for i in range(len(path)-1):
                dist += np.sqrt((path[i][0]-path[i+1][0])**2 + (path[i][1]-path[i+1][1])**2)
            return dist
        else:
            # cost will be average distance from element of interest
            for coord in path:
                dist += np.sqrt((coord[0]-loc[0])**2 + (coord[1]-loc[1])**2)
            dist = dist/len(path)
            return dist

    def visualize_path(self, MPL):
        fig, ax = plt.subplots(figsize=(8, 6))
        xmin = self.extent[0]
        xmax = self.extent[1]
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([self.extent[2], self.extent[3]])

        color = iter(plt.cm.cool(np.linspace(0,1,len(MPL))))

        for i in MPL:
            c = next(color)
            for j in MPL[i]:
                plt.plot(MPL[i][0][j][0][0], MPL[i][0][j][0][1],c=c, marker='*')
        # plt.plot(MPL[:,0], MPL[:,1], marker='*')

        plt.show()

# if __name__=='__main__':

#     bw = obs.FreeWorld()
#     pose = (0.0, 0.0, 0.0)
#     extent = [-10.0, 10.0, -10.0, 10.0]
#     input_limit = [0.0, 10.0, -30.0, 30.0]
#     sample_number = 10
#     horizon_length = 4
#     sample_step = 1.0
#     frontier_size = 1.0
#     total_time = 5.0
#     traj_sampler = continuous_traj_sampler(input_limit,frontier_size, sample_number,total_time, horizon_length, sample_step, extent, bw )
#     sampled_input = traj_sampler.input_sampler()
#     MPL = traj_sampler.get_path_set(pose)

#     for i in MPL:        
#         print(MPL[i])

#     # traj_sampler.visualize_path(MPL)
