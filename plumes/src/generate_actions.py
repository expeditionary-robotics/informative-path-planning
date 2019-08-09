# !/usr/bin/python

'''
This library can be used to access the multiple ways in which path sets can be generated for the simulated vehicle in the PLUMES framework.

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''

import dubins
import numpy as np
import copy
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from geometry_msgs.msg import *
import generate_metric_environment as gme

class ActionSet(object):
    ''' Creates a variety of trajectory options '''
    def __init__(self, **kwargs):
        self.num_actions = kwargs['num_actions']
        self.length = kwargs['length']
        self.turning_radius = kwargs['turning_radius']
        self.radius_angle = kwargs['radius_angle']
        self.num_samples = kwargs['num_samples']
        self.safe_threshold = kwargs['safe_threshold']
        self.unknown_threshold = kwargs['unknown_threshold']
        self.allow_reverse = kwargs['allow_reverse']
        self.allow_stay = kwargs['allow_stay']

        self.actions = None

    def generate_trajectories(self, robot_pose, time, world, using_sim_world=True):
        ''' Based upon the parameters, create an action set '''
        actions = []
        path_swath = np.linspace(-self.radius_angle, self.radius_angle, self.num_actions)
        angles = list(path_swath).append(0)
        for a in path_swath:
            frontier_goal = trig_projection(robot_pose, self.length, a)
            path = dubins.shortest_path(robot_pose, frontier_goal, self.turning_radius)
            samples, _ = path.sample_many(np.round(self.length/self.num_samples, 2))
            actions.append(samples)

        if self.allow_reverse:
            reverse_goal = trig_projection(robot_pose, self.length*1.2, np.pi)
            path = dubins.shortest_path((robot_pose[0], robot_pose[1], robot_pose[2]+np.pi),
                                        reverse_goal, self.turning_radius)
            samples, _ = path.sample_many(self.length/self.num_samples)
            samples = [(s[0], s[1], s[2]-np.pi) for s in samples]
            samples = [(-np.inf, -np.inf, -np.inf)] + samples #flag for the controller
            actions.append(samples)

        if self.allow_stay:
            actions.append([robot_pose for i in range(0, self.num_samples)])

        actions = self.prune_trajectories(actions, time, world, using_sim_world)

        if len(actions) == 0:
            print len(actions)
            relaxed_world = copy.copy(world)
            relaxed_world.safety_buffer=world.safety_buffer*0.5
            relaxed_world.refresh_world()
            print relaxed_world.safety_buffer
            self.generate_trajectories(robot_pose, time, relaxed_world, using_sim_world)

        return actions

    def prune_trajectories(self, actions, time, world, using_sim_world):
        ''' Based upon the world input, prune possible trajectories to be safe '''
        safe_actions = []
        if using_sim_world:
            # want to use the built in functionality to test
            for action in actions:
                if action[0][0] > -np.inf:
                    if world.safe_trajectory(action):
                        safe_actions.append(action)
                else:
                    if world.safe_trajectory(action[1:]):
                        safe_actions.append(action[1:])
        else:
            # want to use the occupancy grid from ROS
            data = make_array(world.data, world.info.height, world.info.width)
            for action in actions:
                if action[0][0] > -np.inf:
                    idy = [int(round((x[0]-current_map.info.origin.position.x)/current_map.info.resolution)) for x in action]
                    idx = [int(round((x[1]-current_map.info.origin.position.y)/current_map.info.resolution)) for x in action]
                else:
                    idy = [int(round((x[0]-current_map.info.origin.position.x)/current_map.info.resolution)) for x in action[1:]]
                    idx = [int(round((x[1]-current_map.info.origin.position.y)/current_map.info.resolution)) for x in action[1:]]
                
                try: #catch if project outside of array
                    cost_vals = data[idx, idy]
                    cost = np.sum([k for k in cost_vals if k >= 0.])
                    unknown_cost = np.sum([k for k in cost_vals if k < 0.])
                except:
                    cost = 0
                    unknown_cost = 0
                    for m, n in zip(idx, idy):
                        try: #get everything in the array
                            if data[m, n] >= 0:
                                cost += data[m, n]
                            else:
                                unknown_cost += data[m, n]
                        except:
                            break
                if (cost < self.safe_threshold and unknown_cost > self.unknown_threshold) and len(action) > 0:
                    safe_actions.append(make_path_object(action, time))
        return safe_actions

def trig_projection(point, step_size, angle):
    ''' Function to perform a path projection'''
    x = step_size*np.cos(point[2]+angle)+point[0]
    y = step_size*np.sin(point[2]+angle)+point[1]
    p = point[2]+angle
    return (x, y, p)

def make_path_object(path, time):
    ''' Does the ROS message conversionros'''
    pub_path = []
    for coord in path:
        pc = Point32()
        pc.x = coord[0]
        pc.y = coord[1]
        pc.z = coord[2] # keep heading information
        pub_path.append(pc)
    pte = PolygonStamped()
    pte.header.frame_id = 'world'
    pte.header.stamp = time
    pte.polygon.points = pub_path
    return pte
def make_array(data, height, width):
    ''' Converts occupancy grid vector to useable array'''
    return np.array(data).reshape((height, width), order='C')

if __name__ == '__main__':
    free_world = gme.World([0, 10, 0, 10])

    action_set = ActionSet({'num_actions':15,
                           'length': 3.5,
                           'turning_radius': 0.005,
                           'radius_angle': np.pi/4.,
                           'num_samples': 10,
                           'safe_threshold': 50.,
                           'unknown_threshold': -2.,
                           'allow_reverse': True,
                           'allow_stay': True})

    safe_actions = action_set.generate_trajectories(robot_pose=(9.7, 2.8, 0),
                                                    time=0,
                                                    world=free_world,
                                                    using_sim_world=True)

    # Visualize the world and pathsets
    bounds = PolygonPatch(free_world.world, alpha=0.5, fc='k', ec='k')
    plt.gca().add_patch(bounds)

    # let's visualize the obstacles
    for obs in free_world.obstacles:
        plt.gca().add_patch(PolygonPatch(obs.geom, alpha=0.5))

    # let's visualize the trajectory
    for action in safe_actions:
        plt.plot([c.x for c in action.polygon.points], [c.y for c in action.polygon.points], c='r')

    plt.gca().axis('square')
    plt.show()
    plt.close()

