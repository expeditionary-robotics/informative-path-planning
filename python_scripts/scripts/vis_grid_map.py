import grid_map_ipp_module as grid 
import obstacles as obs 
import numpy as np
import itertools as iter 
import math 
import os 

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll

class visualization():
    def __init__(self, mapsize, resol, lidar_belief, reward_function, save):
        '''
        - mapsize : Axis length of the map (m)
        - resol : Resolution of grid 
        - Lidar class(C++ binding) which contains belief map
        '''
        self.mapsize = mapsize
        self.resol = resol
        self.lidar = lidar_belief 
        self.save = save #Bool value
        self.reward_function = reward_function

    def visualization(self, t):
        data = self.iterator()
        fig = self.show(data)

        if self.save:
            if not os.path.exists('../figures/nonmyopic/'+str(self.reward_function)+'/GridMap/'):
                os.makedirs('../figures/nonmyopic/'+str(self.reward_function)+'/GridMap/')
            fig.savefig('../figures/nonmyopic/'+str(self.reward_function)+'/GridMap/' + str(t) + '.png')


    def iterator(self):
        num_idx = math.floor(self.mapsize/ self.resol) 
        data = np.random.rand(100, 100) * 2 - 0.5
        # print(type(num_idx))
        for i in range(int(num_idx)):
            for j in range(int(num_idx)):
                x = self.resol/2.0 + i * self.resol 
                y = self.resol/2.0 + j * self.resol

                cur_val = self.lidar.get_occ_value(x, y)
                if cur_val < 0.15:
                    data[j,i] = 1.0
                elif cur_val > 0.85:
                    data[j,i] = 0.0
                else:
                    data[j,i] = 0.5
                
                # print(data[i,j])
        return data 

    def show(self, data):
        fig, ax = plt.subplots()
        cmap = mcolors.ListedColormap(['white', 'gray', 'black'])
        # bounds = [0.0, 1.0, 0.5]
        # norm = mcolors.BoundaryNorm(bounds, cmap.N)
        im = ax.imshow(data, cmap="gray", vmin=0, vmax=1, origin="lower")
        grid = np.arange(-self.resol/2.0, self.mapsize+1, self.resol)
        xmin, xmax, ymin, ymax = -self.resol/2.0, self.mapsize + self.resol/2.0, -self.resol/2.0, self.mapsize + self.resol/2.0

        # plt.show()
        return fig
        



        

# fig, ax = plt.subplots()
# cmap = mcolors.ListedColormap(['white', 'black'])
# bounds = [-0.5, 0.5, 1.5]
# norm = mcolors.BoundaryNorm(bounds, cmap.N)

# data = np.random.rand(100, 100) * 2 - 0.5
# im = ax.imshow(data, cmap=cmap, norm=norm)

# grid = np.arange(-0.5, 101, 1)
# print(grid)
# xmin, xmax, ymin, ymax = -0.5, 100.5, -0.5, 100.5
# lines = ([[(x, y) for y in (ymin, ymax)] for x in grid]
#          + [[(x, y) for x in (xmin, xmax)] for y in grid])
# grid = mcoll.LineCollection(lines, linestyles='solid', linewidths=2,
#                             color='teal')
# # ax.add_collection(grid)

# def animate(i):
#     data = np.random.rand(100, 100) * 2 - 0.5
#     im.set_data(data)
#     # return a list of the artists that need to be redrawn
#     return [im, grid]

# anim = animation.FuncAnimation(
#     fig, animate, frames=200, interval=0, blit=True, repeat=False)
# plt.show()

