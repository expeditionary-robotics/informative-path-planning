# !/usr/bin/python

'''
This library can be used to access the multiple ways in which path sets can be generated for the simulated vehicle in the PLUMES framework.

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''

import dubins
import numpy as np
import generate_metric_environment as gme
import matplotlib.pyplot as plt


class Trajectory(object):
    ''' Creates a trajectory object '''
    def __init__(self, length):
        pass

class ActionSet(object):
    ''' Creates a variety of trajectory options '''
    def __init__(self, num_actions, length, turning_radius, num_pts, metric_environment):
        pass




class Path_Generator:    
    def __init__(self, frontier_size, horizon_length, turning_radius, sample_step, extent, obstacle_world=obs.FreeWorld()):
        ''' Initialize a path generator
        Input:
            frontier_size (int) the number of points on the frontier we should consider for navigation
            horizon_length (float) distance between the vehicle and the horizon to consider
            turning_radius (float) the feasible turning radius for the vehicle
            sample_step (float) the unit length along the path from which to draw a sample
            extent (list of floats) the world boundaries
        '''

        # the parameters for the dubin trajectory
        self.fs = frontier_size
        self.hl = horizon_length
        self.tr = turning_radius
        self.ss = sample_step
        self.extent = extent

        # Global variables
        self.goals = [] #The frontier coordinates
        self.samples = {} #The sample points which form the paths
        self.cp = (0,0,0) #The current pose of the vehicle

        # Determining whether to consider obstacles
        self.obstacle_world = obstacle_world

    def generate_frontier_points(self):
        '''From the frontier_size and horizon_length, generate the frontier points to goal'''
        angle = np.linspace(-2.35,2.35,self.fs) #fix the possibilities to 75% of the unit circle, ignoring points directly behind the vehicle
        goals = []
        for a in angle:
            x = self.hl*np.cos(self.cp[2]+a)+self.cp[0]
            # if x >= self.extent[1]-3*self.tr:
            #     pass
                # x = self.extent[1]-3*self.tr
                # y = (x-self.cp[0])*np.sin(self.cp[2]+a)+self.cp[1]
            # elif x <= self.extent[0]+3*self.tr:
            #     pass
                # x = self.extent[0]+3*self.tr
                # y = (x-self.cp[0])*np.sin(self.cp[2]+a)+self.cp[1]
            # else:
            y = self.hl*np.sin(self.cp[2]+a)+self.cp[1]
                # if y >= self.extent[3]-3*self.tr:
                #     pass
                #     # y = self.extent[3]-3*self.tr
                #     # x = (y-self.cp[1])*-np.cos(self.cp[2]+a)+self.cp[0]
                # elif y <= self.extent[2]+3*self.tr:
                #     pass
                    # y = self.extent[2]+3*self.tr
                    # x = (y-self.cp[1])*-np.cos(self.cp[2]+a)+self.cp[0]
            p = self.cp[2]+a
            if np.linalg.norm([self.cp[0]-x, self.cp[1]-y]) <= self.tr:
                pass
            elif x > self.extent[1]-3*self.tr or x < self.extent[0]+3*self.tr:
                pass
            elif y > self.extent[3]-3*self.tr or y < self.extent[2]+3*self.tr:
                pass
            # elif self.obstacle_world.in_obstacle((x,y), buff=3*self.tr):
            #     pass
            else:
                goals.append((x,y,p))
        goals.append(self.cp)
        self.goals = goals
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
        return self.samples, self.samples

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

    def path_cost(self, path, loc=None):
        ''' Calculate the cost of a path sequence either with respect to path length, or distance from some element in the world (loc)'''
        dist = 0
        if loc is None:
            # cost will be path length
            for i in xrange(len(path)-1):
                dist += np.sqrt((path[i][0]-path[i+1][0])**2 + (path[i][1]-path[i+1][1])**2)
            return dist
        else:
            # cost will be average distance from element of interest
            for coord in path:
                dist += np.sqrt((coord[0]-loc[0])**2 + (coord[1]-loc[1])**2)
            dist = dist/len(path)
            return dist

    def get_frontier_points(self):
        ''' Method to access the goal points'''
        return self.goals

    def get_sample_points(self):
        return self.samples

class Dubins_Path_Generator(Path_Generator):
    '''
    The Dubins_Path_Generator class, which inherits from the Path_Generator class. Replaces the make_sample_paths
    method with paths generated using the dubins library
    '''
    
    def buffered_paths(self):
        sampling_path = {}
        true_path = {}
        for i,goal in enumerate(self.goals):            
            path = dubins.shortest_path(self.cp, goal, self.tr)
            fconfig, _ = path.sample_many(self.ss/10)

            ftemp = []
            for c in fconfig:
                if c[0] > self.extent[0] and c[0] < self.extent[1] and c[1] > self.extent[2] and c[1] < self.extent[3] and not self.obstacle_world.in_obstacle((c[0], c[1]), buff = 0.0):
                    ftemp.append(c)
                else:
                    break

            try:
                ttemp = ftemp[0::10]
                for m,c in enumerate(ttemp):
                    if c[0] <= self.extent[0]+3*self.tr or c[0] >= self.extent[1]-3*self.tr or c[1] <= self.extent[2]+3*self.tr or c[1] >= self.extent[3]-3*self.tr or self.obstacle_world.in_obstacle((c[0], c[1]), buff = 3*self.tr):
                        ttemp = ttemp[0:m-1]

                if len(ttemp) < 2:
                    pass
                else:
                    sampling_path[i] = ttemp
                    true_path[i] = ftemp[0:ftemp.index(ttemp[-1])+1]
            except:
                pass

        return sampling_path, true_path

        
    def make_sample_paths(self):
        '''Connect the current_pose to the goal places'''
        coords, true_coords = self.buffered_paths()

        
        if len(coords) == 0:
            print 'no viable path'
            #pdb.set_trace()
            #coords, true_coords = self.buffered_paths()
            
        self.samples = coords
        return coords, true_coords

        
if __name__ == '__main__':
    # bw = obs.BlockWorld( [0., 10., 0., 10.], num_blocks=1, dim_blocks=(2.,2.), centers=[(6.1,5)])
    # bw = obs.BugTrap([0., 10., 0., 10.], (5,5), 3, channel_size = 0.5, width = 3., orientation='left')
    # bw = obs.ChannelWorld([0., 10., 0., 10.], (6,5), 3, 0.4)
    bw = obs.FreeWorld()

    # extent, discretization, sample_step, turning_radius, step_size,obstacle_world=obs.FreeWorld()
    gen = Reachable_Frontier_Generator([0., 10., 0., 10.], (20,20), 0.5, 0.1, 1.5, bw)
    # gen = Dubins_Path_Generator(15., 1.5, 0.05, 0.5, [0., 10., 0., 10.], bw)
    
    plt.figure()

    trajectory = []
    samples = []
    coord = (5.2,5.2,0)
    for m in range(1):
        paths, true_paths = gen.get_path_set(coord)
        print len(paths)
        action = np.random.choice(paths.keys())
        for i, path in paths.items():
            f = np.array(path)
            plt.plot(f[:,0], f[:,1], 'k*')
        for i, path in true_paths.items():
            f = np.array(path)
            plt.plot(f[:,0], f[:,1], 'r')
        # samples.append(paths[action])
        # trajectory.append(true_paths[action])
        coord = paths[action][-1]
        print m

    # for e, k in zip(samples, trajectory):
    #     f = np.array(e)
    #     l = np.array(k)
    #     plt.plot(f[:,0], f[:,1], 'r*')
    #     plt.plot(l[:,0], l[:,1])
    
    obstacles = bw.get_obstacles()
    for o in obstacles:
        x,y = o.exterior.xy
        plt.plot(x,y)
    plt.axis([0., 10., 0., 10.])
    plt.show()
