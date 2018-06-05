# !/usr/bin/python

'''
This script can be used as a library file for a simulated robot which selects either myopically or nonmyopically the most "rewarding" place to visit in a discretized belief model, and takes a step toward that location, using a Dubins curve path generator.

This script draws heavily from elements in the ipp_library.py file.
''' 

from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib import cm
from sklearn import mixture
from IPython.display import display
from scipy.stats import multivariate_normal
import numpy as np
import scipy as sp
import math
import os
import sys
import GPy as GPy
import dubins
import time
from itertools import chain
import pdb
import logging
logger = logging.getLogger('robot')
import ipp_library as il


# globals for plotting
# MIN_COLOR = 3.0
# MAX_COLOR = 7.5
MIN_COLOR = -25.
MAX_COLOR = 25.


class MCTS_Reachable:
    '''Class that establishes a MCTS for nonmyopic planning'''

    def __init__(self, computation_budget, belief, initial_pose, rollout_length, goal_selections, f_rew, time, ranges, aq_param = None, sample_step=0.5, step_size=1.5, turning_radius = 0.001):
        '''Initialize with constraints for the planning, including whether there is a budget or planning horizon
        Inputs:
            computation_budget (float) number of seconds to run the tree building procedure
            belief (GP model) current belief of the vehicle
            initial_pose (tuple of floats) location of the vehicle in world coordinates
            rollout_length (int) number of actions to rollout after selecting a child (tree depth)
            frontier_size (int) number of options for each action in the tree (tree breadth)
            path_generator (string) how action sets should be developed
            aquisition_function (function) the criteria to make decisions
            f_rew (string) the name of the function used to make decisions
            time (float) time in the global world used for aquisition weighting
        '''

        # Parameterization for the search
        self.comp_budget = computation_budget
        self.GP = belief
        self.cp = initial_pose
        self.rl = rollout_length
        self.goals = goal_selections
        self.ranges = ranges
        if f_rew == 'mes':
            self.aquisition_function = il.mves
        elif f_rew == 'mean':
            self.aquisition_function = il.mean_UCB
        elif f_rew == 'exp_improve':
            self.aquisition_function = il.exp_improvement
        else:
            print 'ERROR!!!!'
        self.f_rew = f_rew
        self.t = time
        self.turning_radius = turning_radius
        self.step_size = step_size
        self.sample_step = sample_step

        # The tree
        self.tree = None
        
        # Elements which are relevant for some acquisition functions
        self.params = None
        self.max_val = None
        self.max_locs = None
        self.current_max = aq_param

    def take_step(self, loc, goal):
        '''
        Create an intermediary goal towards the point of interest such that the robot only translates the step size specified
        Input: Goal
        Output: Navigable points to the intermediary goal
        '''
        coords = {}

        dist = np.sqrt((loc[0]-goal[0])**2 + (loc[1]-goal[1])**2)
        angle_to_goal = np.arctan2([goal[1]-loc[1]], [goal[0]-loc[0]])[0]
        if dist > self.step_size:
            new_goal = (loc[0]+self.step_size*np.sin(np.pi/2-angle_to_goal), loc[1]+self.step_size*np.sin(angle_to_goal), angle_to_goal)
        else:
            new_goal = (goal[0], goal[1], angle_to_goal)

        path = dubins.shortest_path(loc, new_goal, self.turning_radius)
        configurations, _ = path.sample_many(self.sample_step)
        configurations.append(new_goal)

        temp = []
        for i,config in enumerate(configurations):
            if config[0] > self.ranges[0] and config[0] < self.ranges[1] and config[1] > self.ranges[2] and config[1] < self.ranges[3]:
                temp.append(config)
            else:
                pass

        return temp 

    def choose_trajectory(self, t):
        ''' Main function loop which makes the tree and selects the best child
        Output:
            path to take, cost of that path
        '''
        # initialize tree
        self.tree = self.initialize_tree() 
        i = 0 #iteration count

        # randonly sample the world for entropy search function
        if self.f_rew == 'mes':
            self.max_val, self.max_locs = il.sample_max_vals(self.GP, t = t)
            
        time_start = time.clock()            
            
        # while we still have time to compute, generate the tree
        while time.clock() - time_start < self.comp_budget:
            i += 1
            current_node = self.tree_policy()
            sequence = self.rollout_policy(current_node)
            reward = self.get_reward(sequence)
            self.update_tree(reward, sequence)

        # get the best action to take with most promising futures
        best_sequence, best_val, all_vals = self.get_best_child()
        print "Number of rollouts:", i, "\t Size of tree:", len(self.tree)
        logger.info("Number of rollouts: {} \t Size of tree: {}".format(i, len(self.tree)))

        paths = {}
        for i in xrange(len(self.goals)):
            paths[i] = self.take_step(self.cp, self.goals[i])
        return self.tree[best_sequence][0], best_val, paths, all_vals, self.max_locs, self.max_val

    def initialize_tree(self):
        '''Creates a tree instance, which is a dictionary, that keeps track of the nodes in the world
        Output:
            tree (dictionary) an initial tree
        '''
        tree = {}
        # root of the tree is current location of the vehicle
        tree['root'] = (self.cp, 0) #(pose, number of queries)
        for i,goal in enumerate(self.goals):
            samples = self.take_step(self.cp, goal)
            cost = np.sqrt((self.cp[0]-samples[-1][0])**2 + (self.cp[1]-samples[-1][1])**2)
            tree['child '+ str(i)] = (samples, cost, 0, 0) #(samples, cost, reward, number of times queried)
        return tree

    def tree_policy(self):
        '''Implements the UCB policy to select the child to expand and forward simulate. From Arora paper, the following is defined:
            avg_r - average reward of all rollouts that have passed through node n
            c_p - some arbitrary constant, they use 0.1
            N - number of times parent has been evaluated
            n - number of times that node has been evaluated
            the formula: avg_r + c_p * np.sqrt(2*np.log(N)/n)
        '''
        leaf_eval = {}
        # TODO: check initialization, when everything is zero. appears to be throwing error
        actions = self.goals
        for i, val in enumerate(actions):
            try:
                node = 'child '+ str(i)
                leaf_eval[node] = self.tree[node][2] + 0.1*np.sqrt(2*(np.log(self.tree['root'][1]))/self.tree[node][3])
            except:
                pass
        return max(leaf_eval, key=leaf_eval.get)

    def rollout_policy(self, node):
        '''Select random actions to expand the child node
        Input:
            node (the name of the child node that is to be expanded)
        Output:
            sequence (list of names of nodes that make the sequence in the tree)
        '''

        sequence = [node] #include the child node
        #TODO use the cost metric to signal action termination, for now using horizon
        for i in xrange(self.rl):
            actions = self.goals #self.path_generator.get_path_set(self.tree[node][0][-1]) #plan from the last point in the sample
            try:
                a = np.random.randint(0,len(actions)-1) #choose a random path
            except:
                if len(actions) != 0:
                    a = 0

            samples = self.take_step(self.tree[node][0][-1], actions[a])
            self.tree[node + ' child ' + str(a)] = (samples, 0, 0, 0) #add random path to the tree
            node = node + ' child ' + str(a)
            sequence.append(node)

        return sequence

    def get_reward(self, sequence):
        '''Evaluate the sequence to get the reward, defined by the percentage of entropy reduction.
        Input:
            sequence (list of strings) names of the nodes in the tree
        Outut:
            reward value from the aquisition function of choice
        '''
        sim_world = self.GP
        samples = []
        obs = []
        for seq in sequence:
            samples.append(self.tree[seq][0])
            # samples.append((self.goals[int(seq[-1])][0], self.goals[int(seq[-1])][1]))
        obs = list(chain.from_iterable(samples))

        if self.f_rew == 'mes':
            return self.aquisition_function(time = self.t, xvals = obs, robot_model = sim_world, param = self.max_val)
        elif self.f_rew == 'exp_improve':
            return self.aquisition_function(time=self.t, xvals = obs, robot_model = sim_world, param = [self.current_max])
        else:
            return self.aquisition_function(time=self.t, xvals = obs, robot_model = sim_world)

    
    def update_tree(self, reward, sequence):
        '''Propogate the reward for the sequence
        Input:
            reward (float) the reward or utility value of the sequence
            sequence (list of strings) the names of nodes that form the sequence
        '''
        #TODO update costs as well
        self.tree['root'] = (self.tree['root'][0], self.tree['root'][1]+1)
        for seq in sequence:
            samples, cost, rew, queries = self.tree[seq]
            queries += 1
            n = queries
            rew = ((n-1)*rew+reward)/n
            self.tree[seq] = (samples, cost, rew, queries)

    def get_best_child(self):
        '''Query the tree for the best child in the actions
        Output:
            (string, float) node name of the best child, the cost of that child
        '''
        best = -float('inf')
        best_child = None
        value = {}
        for i in xrange(len(self.goals)):
            try:
                r = self.tree['child '+ str(i)][2]
                value[i] = r
                #if r > best and len(self.tree['child '+ str(i)][0]) > 1: 
                if r > best: 
                    best = r
                    best_child = 'child '+ str(i)
            except:
                pass
        return best_child, best, value


class Reachable_Robot():
    '''This robot inherits from the Robot class, but uses a MCTS in order to perform global horizon planning'''
    
    def __init__(self, sample_world, start_loc = (0.0, 0.0, 0.0), extent = (-10., 10., -10., 10.), 
            kernel_file = None, kernel_dataset = None, prior_dataset = None, init_lengthscale = 10.0, 
            init_variance = 100.0, noise = 0.05, step_size = 1.5, turning_radius = 1, sample_step = 0.5, discretization = (10,10), evaluation = None, 
            f_rew = 'mean', create_animation = False, use_mcts = False, learn_params = False, computation_budget = 60, rollout_length = 6):
        ''' Initialize the robot class with a GP model, initial location, path sets, and prior dataset'''
                   
        # General params
        self.ranges = extent
        self.create_animation = create_animation
        self.eval = evaluation
        self.loc = start_loc
        self.sample_world = sample_world
        self.f_rew = f_rew
        self.maxes = []
        self.current_max = -1000
        self.current_max_loc = [-1,-1]
        self.max_locs = None
        self.max_vals = None
        self.learn_params = learn_params

        # Handle the type of reward function
        if f_rew == 'hotspot_info':
            self.aquisition_function = il.hotspot_info_UCB
        elif f_rew == 'mean':
            self.aquisition_function = il.mean_UCB 
        elif f_rew == 'info_gain':
            self.aquisition_function = il.info_gain
        elif f_rew == 'mes':
            self.aquisition_function = il.mves
        elif f_rew == 'exp_improve':
            self.aquisition_function = il.exp_improvement
        else:
            raise ValueError('Only \'hotspot_info\' and \'mean\' and \'info_gain\' and \'mes\' and \'exp_improve\' reward fucntions supported.')
        

        # Initialize the belief model
        self.GP = il.GPModel(ranges = extent, lengthscale = init_lengthscale, variance = init_variance)
                
        # If both a kernel training dataset and a prior dataset are provided, train the kernel using both
        if  kernel_dataset is not None and prior_dataset is not None:
            data = np.vstack([prior_dataset[0], kernel_dataset[0]])
            observations = np.vstack([prior_dataset[1], kernel_dataset[1]])
            self.GP.train_kernel(data, observations, kernel_file) 
        # Train the kernel using the provided kernel dataset
        elif kernel_dataset is not None:
            self.GP.train_kernel(kernel_dataset[0], kernel_dataset[1], kernel_file)
        # If a kernel file is provided, load the kernel parameters
        elif kernel_file is not None:
            self.GP.load_kernel()
        # No kernel information was provided, so the kernel will be initialized with provided values
        else:
            pass
        
        # Incorporate the prior dataset into the model
        if prior_dataset is not None:
            self.GP.add_data(prior_dataset[0], prior_dataset[1]) 

        # Params for nonmyopic planning
        self.use_mcts = use_mcts
        self.comp_budget = computation_budget
        self.roll_length = rollout_length

        # Params for navigation
        self.step_size = step_size
        self.sample_step = sample_step
        self.turning_radius = turning_radius

        x1vals = np.linspace(extent[0], extent[1], discretization[0])
        x2vals = np.linspace(extent[2], extent[3], discretization[1])
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy')
        self.goals = np.vstack([x1.ravel(), x2.ravel()]).T



    def choose_destination(self, t):
        '''
        Select the best location in the discretized world to navigate to
        Input: t, step number
        Output: None or location index
        '''
        value = {}
        paths = {}
        param = None

        max_locs = max_vals = None
        if self.f_rew == 'mes':
            self.max_val, self.max_locs = il.sample_max_vals(self.GP, t = t)

        for i,goal in enumerate(self.goals):
            if self.f_rew == 'mes':
                param = self.max_val
            elif self.f_rew == 'exp_improve':
                param = [self.current_max]
            xvals = self.take_step(goal)#[(goal[0], goal[1])]
            paths[i] = xvals
            value[i] = self.aquisition_function(time=t,
                                                xvals=xvals,
                                                robot_model=self.GP,
                                                param=param)
        try:
            return self.goals[max(value, key=value.get)], value[max(value, key=value.get)], paths, value, self.max_locs
        except:
            return None

    def take_step(self, goal):
        '''
        Create an intermediary goal towards the point of interest such that the robot only translates the step size specified
        Input: Goal
        Output: Navigable points to the intermediary goal
        '''
        coords = {}

        dist = np.sqrt((self.loc[0]-goal[0])**2 + (self.loc[1]-goal[1])**2)
        angle_to_goal = np.arctan2([goal[1]-self.loc[1]], [goal[0]-self.loc[0]])[0]
        if dist > self.step_size:
            new_goal = (self.loc[0]+self.step_size*np.sin(np.pi/2-angle_to_goal), self.loc[1]+self.step_size*np.sin(angle_to_goal), angle_to_goal)
        else:
            new_goal = (goal[0], goal[1], angle_to_goal)

        path = dubins.shortest_path(self.loc, new_goal, self.turning_radius)
        configurations, _ = path.sample_many(self.sample_step)
        configurations.append(new_goal)

        temp = []
        for i,config in enumerate(configurations):
            if config[0] > self.ranges[0] and config[0] < self.ranges[1] and config[1] > self.ranges[2] and config[1] < self.ranges[3]:
                temp.append(config)
            else:
                pass

        return temp 

    def collect_observations(self, xobs):
        ''' Gather noisy samples of the environment and updates the robot's GP model.
        Input: 
            xobs (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2 '''
        zobs = self.sample_world(xobs)       
        self.GP.add_data(xobs, zobs)

        for z, x in zip (zobs, xobs):
            if z[0] > self.current_max:
                self.current_max = z[0]
                self.current_max_loc = [x[0],x[1]]

    def predict_max(self):
        # If no observations have been collected, return default value
        if self.GP.xvals is None:
            return np.array([0., 0.]).reshape(1,2), 0.

        ''' First option, return the max value observed so far '''
        #return self.GP.xvals[np.argmax(self.GP.zvals), :], np.max(self.GP.zvals)

        ''' Second option: generate a set of predictions from model and return max '''
        # Generate a set of observations from robot model with which to predict mean
        x1vals = np.linspace(self.ranges[0], self.ranges[1], 30)
        x2vals = np.linspace(self.ranges[2], self.ranges[3], 30)
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy') 
        data = np.vstack([x1.ravel(), x2.ravel()]).T
        observations, var = self.GP.predict_value(data)        

        return data[np.argmax(observations), :], np.max(observations)

    def planner(self, T):
        '''
        Select the best point to navigate to myopically or nonmyopically; and perform the simulated navigation
        '''
        self.trajectory = []

        for t in xrange(T):
            print "[", t, "] Current Location:  ", self.loc
            logger.info("[{}] Current Location: {}".format(t, self.loc))
            if self.use_mcts == False:
                best_location, best_val, all_paths, all_values, max_locs = self.choose_destination(t = t)
                sampling_path = self.take_step(best_location)
            else:
                if self.f_rew == "exp_improve":
                    param = self.current_max
                else:
                    param = None

                mcts = MCTS_Reachable(self.comp_budget, self.GP, self.loc, self.roll_length, self.goals, self.f_rew, t, aq_param = param, turning_radius=self.turning_radius, sample_step=self.sample_step, step_size = self.step_size, ranges=self.ranges)
                best_path, best_val, all_paths, all_values, max_locs, max_val = mcts.choose_trajectory(t = t)
                sampling_path = best_path


            # Given this choice, update the evaluation metrics 
            pred_loc, pred_val = self.predict_max()
            print "Current predicted max and value: \t", pred_loc, "\t", pred_val
            logger.info("Current predicted max and value: {} \t {}".format(pred_loc, pred_val))
            try:
                self.eval.update_metrics(len(self.trajectory), self.GP, all_paths, sampling_path, value = best_val, max_loc = pred_loc, max_val = pred_val, params = [self.current_max, self.current_max_loc, self.max_val, self.max_locs]) 
            except:
                max_locs = [[-1, -1], [-1, -1]]
                max_val = [-1,-1]
                self.eval.update_metrics(len(self.trajectory), self.GP, all_paths, sampling_path, value = best_val, max_loc = pred_loc, max_val = pred_val, params = [self.current_max, self.current_max_loc, max_val, max_locs]) 


            # Given this choice, take a step in the right direction, obeying to the dynamics of the vehicle
            print sampling_path
            if len(sampling_path) == 0:
                break
            data = np.array(sampling_path)
            x1 = data[:,0]
            x2 = data[:,1]
            xlocs = np.vstack([x1, x2]).T

            self.collect_observations(xlocs)

            if t < T/3 and self.learn_params == True:
                self.GP.train_kernel()

            self.trajectory.append(sampling_path)

            if self.create_animation:
                self.visualize_trajectory(screen = False, filename = t, best_path = sampling_path, maxes = max_locs, all_paths = all_paths, all_vals = all_values)            

            self.loc = sampling_path[-1]
        np.savetxt('./figures/' + self.f_rew+ '/robot_model.csv', (self.GP.xvals[:, 0], self.GP.xvals[:, 1], self.GP.zvals[:, 0]))

    def visualize_trajectory(self, screen = True, filename = 'SUMMARY', best_path = None, 
        maxes = None, all_paths = None, all_vals = None):      
        ''' Visualize the set of paths chosen by the robot 
        Inputs:
            screen (boolean): determines whether the figure is plotted to the screen or saved to file
            filename (string): substring for the last part of the filename i.e. '0', '1', ...
            best_path (path object)
            maxes (list of locations)
            all_paths (list of path objects)
            all_vals (list of all path rewards) 
            T (string or int): string append to the figure filename
        '''
        
        # Generate a set of observations from robot model with which to make contour plots
        x1vals = np.linspace(self.ranges[0], self.ranges[1], 100)
        x2vals = np.linspace(self.ranges[2], self.ranges[3], 100)
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy') 
        data = np.vstack([x1.ravel(), x2.ravel()]).T
        observations, var = self.GP.predict_value(data)        
        
       
        # Plot the current robot model of the world
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(self.ranges[0:2])
        ax.set_ylim(self.ranges[2:])
        plot = ax.contourf(x1, x2, observations.reshape(x1.shape), cmap = 'viridis', vmin = MIN_COLOR, vmax = MAX_COLOR, levels=np.linspace(MIN_COLOR, MAX_COLOR, 15))
        if self.GP.xvals is not None:
            scatter = ax.scatter(self.GP.xvals[:, 0], self.GP.xvals[:, 1], c='k', s = 20.0, cmap = 'viridis')                
        color = iter(plt.cm.cool(np.linspace(0,1,len(self.trajectory))))
       
        # Plot the current trajectory
        for i, path in enumerate(self.trajectory):
            c = next(color)
            f = np.array(path)
            plt.plot(f[:,0], f[:,1], c=c, marker='*')

        # If available, plot the current set of options available to robot, colored
        # by their value (red: low, yellow: high)
        # if all_paths is not None:
        #     all_vals = [x for x in all_vals.values()]   
        #     path_color = iter(plt.cm.autumn(np.linspace(0, max(all_vals),len(all_vals))/ max(all_vals)))        
        #     path_order = np.argsort(all_vals)
            
        #     for index in path_order:
        #         c = next(path_color)                
        #         points = all_paths[all_paths.keys()[index]]
        #         f = np.array(points)
        #         plt.plot(f[:,0], f[:,1], c = c, marker='.')
               
        # If available, plot the selected path in green
        if best_path is not None:
            f = np.array(best_path)
            plt.plot(f[:,0], f[:,1], c = 'g', marker='*')
           
        # If available, plot the current location of the maxes for mes
        if maxes is not None:
            for coord in maxes:
                plt.scatter(coord[0], coord[1], color = 'r', marker = '*', s = 500.0)
            # plt.scatter(maxes[:, 0], maxes[:, 1], color = 'r', marker = '*', s = 500.0)
           
        # Either plot to screen or save to file
        if screen:
            plt.show()           
        else:
            if not os.path.exists('./figures/' + str(self.f_rew)):
                os.makedirs('./figures/' + str(self.f_rew))
            fig.savefig('./figures/' + str(self.f_rew)+ '/trajectory-N.' + str(filename) + '.png')
            #plt.show()
            plt.close()
            
    def visualize_world_model(self, screen = True, filename = 'SUMMARY'):
        ''' Visaulize the robots current world model by sampling points uniformly in space and 
        plotting the predicted function value at those locations.
        Inputs:
            screen (boolean): determines whether the figure is plotted to the screen or saved to file 
            filename (String): name of the file to be made
            maxes (locations of largest points in the world)
        '''
        # Generate a set of observations from robot model with which to make contour plots
        x1vals = np.linspace(self.ranges[0], self.ranges[1], 100)
        x2vals = np.linspace(self.ranges[2], self.ranges[3], 100)
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy') # dimension: NUM_PTS x NUM_PTS       
        data = np.vstack([x1.ravel(), x2.ravel()]).T
        observations, var = self.GP.predict_value(data)        
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.set_xlim(self.ranges[0:2])
        ax2.set_ylim(self.ranges[2:])        
        ax2.set_title('Countour Plot of the Robot\'s World Model')     
        plot = ax2.contourf(x1, x2, observations.reshape(x1.shape), cmap = 'viridis', vmin = MIN_COLOR, vmax = MAX_COLOR, levels=np.linspace(MIN_COLOR, MAX_COLOR, 15))

        # Plot the samples taken by the robot
        if self.GP.xvals is not None:
            scatter = ax2.scatter(self.GP.xvals[:, 0], self.GP.xvals[:, 1], c=self.GP.zvals.ravel(), s = 10.0, cmap = 'viridis')        
        if screen:
            plt.show()           
        else:
            if not os.path.exists('./figures/' + str(self.f_rew)):
                os.makedirs('./figures/' + str(self.f_rew))
            fig.savefig('./figures/' + str(self.f_rew)+ '/world_model.' + str(filename) + '.png')
            plt.close()
    
    def plot_information(self):
        ''' Visualizes the accumulation of reward and aquisition functions ''' 
        self.eval.plot_metrics()

if __name__ == '__main__':
    seed = 0#int(sys.argv[1])
    reward_function = 'exp_improve'#sys.argv[2]

    if not os.path.exists('./figures/' + str(reward_function)): 
        os.makedirs('./figures/' + str(reward_function))
    logging.basicConfig(filename = './figures/'+ reward_function + '/robot.log', level = logging.INFO)
    logger = logging.getLogger('robot')
    from ipp_library import *

    # Create a random enviroment sampled from a GP with an RBF kernel and specified hyperparameters, mean function 0 
    # The enviorment will be constrained by a set of uniformly distributed  sample points of size NUM_PTS x NUM_PTS
    ''' Options include mean, info_gain, hotspot_info, and mes'''
    ranges = (0., 10., 0., 10.)

    world = il.Environment(ranges = ranges, # x1min, x1max, x2min, x2max constraints
                        NUM_PTS = 20, 
                        variance = 100.0, 
                        lengthscale = 1.0, 
                        visualize = True,
                        seed = seed)

    evaluation = il.Evaluation(world = world, reward_function = reward_function)

    # Create the point robot
    robot = Reachable_Robot(sample_world = world.sample_value, 
                  start_loc = (5.0, 5.0, 0.0), 
                  extent = ranges,
                  kernel_file = None,
                  kernel_dataset = None,
                  prior_dataset =  None, 
                  init_lengthscale = 1.0, 
                  init_variance = 100.0, 
                  noise = 0.0001,
                  step_size = 1.5, 
                  turning_radius = 0.05,
                  sample_step = 0.5,
                  evaluation = evaluation, 
                  f_rew = reward_function, 
                  create_animation = True,
                  discretization=(20,20),
                  use_mcts = True,
                  computation_budget = 3.0,
                  rollout_length = 5) 

    robot.planner(T = 5)
    #robot.visualize_world_model(screen = True)
    # robot.visualize_trajectory(screen = False)
    robot.plot_information()