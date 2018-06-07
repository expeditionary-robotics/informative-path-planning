# !/usr/bin/python

'''
This library can be used to access the multiple ways in which path sets can be generated for the simulated vehicle in the PLUMES framework.

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''

from IPython.display import display
import numpy as np
import math
import dubins

class Path_Generator:    
    def __init__(self, frontier_size, horizon_length, turning_radius, sample_step, extent):
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

    def generate_frontier_points(self):
        '''From the frontier_size and horizon_length, generate the frontier points to goal'''
        angle = np.linspace(-2.35,2.35,self.fs) #fix the possibilities to 75% of the unit circle, ignoring points directly behind the vehicle
        goals = []
        for a in angle:
            x = self.hl*np.cos(self.cp[2]+a)+self.cp[0]
            if x >= self.extent[1]-3*self.tr:
                x = self.extent[1]-3*self.tr
                y = (x-self.cp[0])*np.sin(self.cp[2]+a)+self.cp[1]
            elif x <= self.extent[0]+3*self.tr:
                x = self.extent[0]+3*self.tr
                y = (x-self.cp[0])*np.sin(self.cp[2]+a)+self.cp[1]
            else:
                y = self.hl*np.sin(self.cp[2]+a)+self.cp[1]
                if y >= self.extent[3]-3*self.tr:
                    y = self.extent[3]-3*self.tr
                    x = (y-self.cp[1])*-np.cos(self.cp[2]+a)+self.cp[0]
                elif y <= self.extent[2]+3*self.tr:
                    y = self.extent[2]+3*self.tr
                    x = (y-self.cp[1])*-np.cos(self.cp[2]+a)+self.cp[0]
            p = self.cp[2]+a
            if np.linalg.norm([self.cp[0]-x, self.cp[1]-y]) <= self.tr:
                pass
            elif x > self.extent[1]-3*self.tr or x < self.extent[0]+3*self.tr:
                pass
            elif y > self.extent[3]-3*self.tr or y < self.extent[2]+3*self.tr:
                pass
            else:
                goals.append((x,y,p))

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
        return self.samples

    def get_path_set(self, current_pose):
        '''Primary interface for getting list of path sample points for evaluation
        Input:
            current_pose (tuple of x, y, z, a which are floats) current location of the robot in world coordinates
        Output:
            paths (dictionary of frontier keys and sample points)
        '''
        self.cp = current_pose
        self.generate_frontier_points()
        paths = self.make_sample_paths()
        return paths

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
        coords = {}
        for i,goal in enumerate(self.goals):            
            path = dubins.shortest_path(self.cp, goal, self.tr)
            configurations, _ = path.sample_many(self.ss)
            configurations.append(goal)

            temp = []
            for config in configurations:
                if config[0] > self.extent[0] and config[0] < self.extent[1] and config[1] > self.extent[2] and config[1] < self.extent[3]:
                    temp.append(config)
                else:
                    temp = []
                    break

            if len(temp) < 2:
                pass
            else:
                coords[i] = temp

        if len(coords) == 0:
            pdb.set_trace()
        return coords    
        
    def make_sample_paths(self):
        '''Connect the current_pose to the goal places'''
        coords = self.buffered_paths()
        
        if len(coords) == 0:
            print 'no viable path'
            
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
        for i,goal in enumerate(self.goals):
            g = (goal[0],goal[1],self.cp[2])
            path = dubins.shortest_path(self.cp, goal, self.tr)
            configurations, _ = path.sample_many(self.ss)
            coords[i] = [config for config in configurations if config[0] > self.extent[0] and config[0] < self.extent[1] and config[1] > self.extent[2] and config[1] < self.extent[3]]
        
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
        
class Reachable_Frontier_Generator():
    '''
    Generates a list of reachable goals within a world, and develops Dubins curve style trajectories and sample sets to reach these goals
    '''
    def __init__(self, extent, discretization, sample_step, turning_radius, step_size):
        self.ranges = extent
        self.discretization = discretization
        self.sample_step = sample_step
        self.turning_radius = turning_radius
        self.step_size = step_size

        x1vals = np.linspace(extent[0], extent[1], discretization[0])
        x2vals = np.linspace(extent[2], extent[3], discretization[1])
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy')
        self.goals = np.vstack([x1.ravel(), x2.ravel()]).T

    def take_step(self, loc):
        ''' Given a current location and a goal, determine the dubins curve sampling path'''
        sampling_path = {}

        for i,goal in enumerate(self.goals):
            sampling_path[i] = []
            dist = np.sqrt((loc[0]-goal[0])**2 + (loc[1]-goal[1])**2)
            angle_to_goal = np.arctan2([goal[1]-loc[1]], [goal[0]-loc[0]])[0]
            new_goal = (goal[0], goal[1], angle_to_goal)

            path = dubins.shortest_path(loc, new_goal, self.turning_radius)
            configurations, _ = path.sample_many(self.sample_step)
            configurations.append(new_goal)

            for config in configurations:
                if config[0] > self.ranges[0] and config[0] < self.ranges[1] and config[1] > self.ranges[2] and config[1] < self.ranges[3]:
                    sampling_path[i].append(config)
                else:
                    break
        return sampling_path 

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

    def get_path_set(self, cp):
        return self.take_step(cp)


class Reachable_Step_Generator(Reachable_Frontier_Generator):
    '''
    Generates a list of reachable goals within a world, and develops Dubins curve style trajectories which take a step toward the goal
    '''
    def take_step(self, loc):
        ''' Given a current location and a goal, determine the dubins curve sampling path'''
        sampling_path = {}

        for i,goal in enumerate(self.goals):
            sampling_path[i] = []
            dist = np.sqrt((loc[0]-goal[0])**2 + (loc[1]-goal[1])**2)
            angle_to_goal = np.arctan2([goal[1]-loc[1]], [goal[0]-loc[0]])[0]

            if dist > self.step_size:
                new_goal = (loc[0]+self.step_size*np.sin(np.pi/2-angle_to_goal), loc[1]+self.step_size*np.sin(angle_to_goal), angle_to_goal)
            else:
                new_goal = (goal[0], goal[1], angle_to_goal)

            path = dubins.shortest_path(loc, new_goal, self.turning_radius)
            configurations, _ = path.sample_many(self.sample_step)
            configurations.append(new_goal)

            for config in configurations:
                if config[0] > self.ranges[0] and config[0] < self.ranges[1] and config[1] > self.ranges[2] and config[1] < self.ranges[3]:
                    sampling_path[i].append(config)
                else:
                    break
        return sampling_path 

    