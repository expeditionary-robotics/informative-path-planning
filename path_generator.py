import dubins
import numpy as np
import matplotlib.pyplot as plt

class Path_Generator:
    '''The Path_Generator class which creates naive point-to-point straightline paths'''
    
    def __init__(self, frontier_size, horizon_length, turning_radius, sample_step):
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
        
        # Global variables
        self.goals = [] #The frontier coordinates
        self.samples = {} #The sample points which form the paths
        self.cp = (0,0,0) #The current pose of the vehicle
        
    def generate_frontier_points(self):
        '''From the frontier_size and horizon_length, generate the frontier points to goal'''
        angle = np.linspace(-2.35,2.35,self.fs) #fix the possibilities to 75% of the unit circle, ignoring points directly behind the vehicle
        self.goals = [(self.hl*np.cos(self.cp[2]+a)+self.cp[0], self.hl*np.sin(self.cp[2]+a)+self.cp[1], self.cp[2]+a) for a in angle]
        return self.goals
        
    def make_sample_paths(self):
        '''Connect the current_pose to the goal places'''
        cp = np.array(self.cp)
        coords = {}
        for i,goal in enumerate(self.goals):
            g = np.array(goal)
            distance = np.sqrt((cp[0]-g[0])**2 + (cp[1]-g[1])**2)
            samples = int(round(distance/self.ss))
            
            for i in range(0,samples):
                x = cp[0]+(i*self.ss)*np.cos(g[2])
                y = cp[1]+(i*self.ss)*np.sin(g[2])
                try: 
                    coords[i].append((x,y))
                except:
                    coords[i] = []
                    coords[i].append((x,y))
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
            coords[i] = configurations
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
            coords[i] = configurations
        
        # find the "shortest" path in sample space
        current_min = 1000
        for key,path in coords.items():
            if len(path) < current_min:
                current_min = len(path)
        
        # limit all paths to the shortest path in sample space
        for key,path in coords.items():
            if len(path) > current_min:
                path = path[0:current_min]
                coords[key]=path
        
        self.samples = coords
        return coords