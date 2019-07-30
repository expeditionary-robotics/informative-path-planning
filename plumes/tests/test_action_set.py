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

def test_action_set():
    ''' This function will create a known static world and set of trajectories
    and will check pruning'''
    free_world = World([0, 10, 0, 10])
    free_world.add_blocks(num=3, dim=(2, 2), centers=[(3, 3), (5, 5), (7, 7)])

    args = {'num_actions':15,
            'length': 1.5,
            'turning_radius': 0.005,
            'radius_angle': np.pi/4.,
            'num_samples': 10,
            'safe_threshold': 50.,
            'unknown_threshold': -2.,
            'allow_reverse': True,
            'allow_stay': True}

    action_set = ActionSet(args)

    a = action_set.generate_trajectories(robot_pose=(1, 1, 0),
                                         time=0,
                                         world=free_world,
                                         using_sim_world=True)
    b = action_set.generate_trajectories(robot_pose=(5, 5, 0),
                                         time=0,
                                         world=free_world,
                                         using_sim_world=True)
    c = action_set.generate_trajectories(robot_pose=(3, 8, 0),
                                         time=0,
                                         world=free_world,
                                         using_sim_world=True)


    assert len(a) > 1, 'Bad Pruning near Boundaries'
    assert len(b) == 0, 'Does Not Handle Teleportation'
    assert len(c) == 17, 'Conservative Pruning'

if __name__ == '__main__':
    test_action_set()
