# !/usr/bin/python

'''
This allows for access to the obstacle class which can populate a given shapely environment.
Generally is a wrapper for the shapely library.

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''

from shapely.geometry import LineString, Point, Polygon
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import numpy as np

class BlockObstacle(object):
    ''' An obstacle object'''
    def __init__(self, dim=(2., 2.), center=(0., 0.), safety_buffer=0.1):
        ''' Creates a block object '''
        points = [(center[0]+dim[0]/2, center[1]+dim[1]/2),
                  (center[0]+dim[0]/2, center[1]-dim[1]/2),
                  (center[0]-dim[0]/2, center[1]-dim[1]/2),
                  (center[0]-dim[0]/2, center[1]+dim[1]/2),
                  (center[0]+dim[0]/2, center[1]+dim[1]/2)]
        self.points = points
        self.geom = Polygon(points).buffer(safety_buffer)

    def get_corners(self):
        ''' returns the point that define the polygon '''
        return self.points

    def get_polygon(self):
        '''  returns the polygon '''
        return self.geom

    def crosses(self, points, safety_buffer=0.1):
        ''' returns whether a set of points crosses the obstacle '''
        pts = LineString(points).buffer(safety_buffer)
        return pts.intersects(self.geom)

    def contains(self, point):
        return self.geom.contains(Point(point))

class World(object):
    ''' creates a world polygon with some extent '''
    def __init__(self, extent=None, safety_buffer=0.1):
        self.extent = extent
        if self.extent is None:
            self.world = None
        else:
            self.world = LineString([(extent[0], extent[2]),
                                     (extent[0], extent[3]),
                                     (extent[1], extent[3]),
                                     (extent[1], extent[2]),
                                     (extent[0], extent[2])]).buffer(safety_buffer)
        self.obstacles = []

    def contains(self, trajectory, safety_buffer=0.1):
        ''' checks that a trajectory stays within the world bounds '''
        if self.extent is None:
            return True
        else:
            traj = LineString(trajectory).buffer(safety_buffer)
            crosses = traj.intersects(self.world)
            outside = traj.bounds[0] >= self.extent[1] or \
                      traj.bounds[1] >= self.extent[3] or \
                      traj.bounds[0] <= self.extent[0] or \
                      traj.bounds[1] <= self.extent[2]
            return not crosses and not outside

    def contains_point(self, point):
        '''checks that a specific point is in the world bounds '''
        if self.extent is None:
            if self.obstacles is None:
                return True
            else:
                for obs in self.obstacles:
                    if obs.contains(Point(point)):
                        return False
                    else:
                        pass
                return True
        else:
            outside = point[0] >= self.extent[1] or \
                      point[1] >= self.extent[3] or \
                      point[0] <= self.extent[0] or \
                      point[1] <= self.extent[2]
            if outside:
                return False
            else:
                print outside
                if self.obstacles is None:
                    return True
                else:
                    for obs in self.obstacles:
                        if obs.contains(Point(point)):
                            return False
                        else:
                            pass
                    return True

    def add_blocks(self, num, dim, centers=None):
        ''' adds block obstacles to the world '''
        if centers is None:
            centers = []
            for i in range(0, num):
                centers.append(((self.extent[1]-dim[0]/2-(self.extent[0]+dim[0]/2))*np.random.random()+self.extent[0]+dim[0]/2,
                                (self.extent[3]-dim[1]/2-(self.extent[2]+dim[1]/2))*np.random.random()+self.extent[2]+dim[1]/2))
        self.obstacles = []
        for i in range(0, num):
            self.obstacles.append(BlockObstacle(dim, centers[i]))

    def safe_trajectory(self, trajectory):
        ''' Takes a trajectory and determines whether it is safe '''
        if self.contains(trajectory):
            if self.obstacles is None:
                return True
            else:
                for obs in self.obstacles:
                    if obs.crosses(trajectory):
                        return False
                return True
        else:
            return False


if __name__ == '__main__':
    free_world = World([0, 10, 0, 10])
    free_world.add_blocks(3, (2, 5))# [(3, 3), (5, 5), (7, 7)])

    trajectory = [(5, 5), (5, 5)]

    print free_world.safe_trajectory(trajectory)

    #let's visualize the world
    bounds = PolygonPatch(free_world.world, alpha=0.5, fc='k', ec='k')
    plt.gca().add_patch(bounds)

    # let's visualize the obstacles
    for obs in free_world.obstacles:
        plt.gca().add_patch(PolygonPatch(obs.geom, alpha=0.5))

    # let's visualize the trajectory
    plt.plot([x[0] for x in trajectory], [x[1] for x in trajectory], c='r')
    traj = LineString(trajectory).buffer(0.1)
    buff = PolygonPatch(traj, alpha=0.5, fc='r', ec='r')
    plt.gca().add_patch(buff)

    plt.gca().axis('square')
    plt.show()
    plt.close()
