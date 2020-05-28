# !/usr/bin/python

'''
Based on the library of PLUMES, extend Monte Carlo Tree Search class to continuous-action case. 

This library allows access to the Monte Carlo Tree Search class used in the PLUMES framework.
A MCTS allows for performing many forward simulation of multiple-chained actions in order to 
select the single most promising action to take at some time t. We have presented a variation
of the MCTS by forward simulating within an incrementally updated GP belief world.

'''

import numpy as np
import scipy as sp
import math
import os
import GPy as GPy
import time
from itertools import chain
import pdb
import logging
logger = logging.getLogger('robot')
from aq_library import *
import copy
import random
import mcts_library as mcts_lib
import continuous_traj as traj

class conti_action_MCTS(mcts_lib.MCTS):
    '''
    Class inherited from MCTS class. 
    '''
    def __init__(self, time):
        self.t = time

    def update_action(self):
        
        grad = self.get_value_gradient

    def get_value_gradient(self):
        val_gradient = 0.0

        return val_gradient


if __name__ == "__main__":

    