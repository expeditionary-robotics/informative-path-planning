import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib import cm
from sklearn import mixture
from scipy.stats import multivariate_normal
import math
import os
import sys

''' Must get the world max from the relevent robot log file '''
# TODO: could automate
#world_max = np.reshape(np.array([6.84210526,  0.52631579]), (1,2)) #0
world_max = np.reshape(np.array([ 9.47368421,  3.15789474]), (1,2)) #500
#world_max = np.reshape(np.array([ 9.47368421,  7.89473684]), (1,2)) #10000

''' Read in the relevent sample locations '''
data_mes = np.loadtxt('./experiments/nonmyopic_01_seed500/figures/mes/robot_model.csv')
data_ucb = np.loadtxt('./experiments/nonmyopic_01_seed500/figures/mean/robot_model.csv')

x1_mes = data_mes[0, :]
x2_mes = data_mes[1, :]
x1_ucb = data_ucb[0, :]
x2_ucb = data_ucb[1, :]
x_mes = np.vstack([x1_mes, x2_mes]).T
x_ucb = np.vstack([x1_ucb, x2_ucb]).T

''' Compute average distances '''
dist_mes = sp.spatial.distance.cdist(world_max, x_mes).T
dist_ucb = sp.spatial.distance.cdist(world_max, x_ucb).T

''' Plot Histogram '''
#hist, bin_edges = np.histogram(dist_mes, linspace(max(dist_mes), min(dist_mes), floor(max(dist_mes)-min(dist_mes)))
fig, axes = plt.subplots(1, 2, sharey = True)
axes[0].hist(dist_mes, bins = np.linspace(min(dist_mes), max(dist_mes), np.floor(max(dist_mes)-min(dist_mes))), color = 'b')
axes[0].set_title('MES')
axes[1].hist(dist_ucb, bins = np.linspace(min(dist_mes), max(dist_mes), np.floor(max(dist_mes)-min(dist_mes))), color = 'r')
axes[1].set_title('UCB')
plt.show()

