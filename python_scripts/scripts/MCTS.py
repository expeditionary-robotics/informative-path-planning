from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib import cm
from sklearn import mixture
# from IPython.display import display
from scipy.stats import multivariate_normal
import numpy as np
import math
import os
import GPy as GPy
import dubins
import time
from itertools import chain
import aq_library as aqlib
# import glog as log
import logging as log
# import gpmodel_library as gp_lib
# from continuous_traj import continuous_traj

from Environment import *
from Evaluation import *
from GPModel import *


class MCTS():
    '''Class that establishes a MCTS for nonmyopic planning'''
    def __init__(self, ranges, obstacle_world, computation_budget, belief, initial_pose, planning_limit, frontier_size, path_generator, aquisition_function, time, gradient_on, grad_step):
        '''Initialize with constraints for the planning, including whether there is 
           a budget or planning horizon
           budget - length, time, etc to consider
           belief - GP model of the robot current belief state
           initial_pose - (x,y,rho) for vehicle'''
        self.ranges = ranges
        self.budget = computation_budget
        self.GP = belief
        self.cp = initial_pose
        self.limit = planning_limit
        self.frontier_size = frontier_size
        self.path_generator = path_generator
        self.obstacle_world = obstacle_world
        # self.default_path_generator = Path_Generator(frontier_size, )
        self.spent = 0
        self.tree = None
        self.aquisition_function = aquisition_function
        self.t = time
        self.gradient_on = gradient_on
        self.grad_step = grad_step

    def get_actions(self):
        self.tree = self.initialize_tree()
        time_start = time.clock()
        
        while time.clock() - time_start < self.budget:
            current_node = self.tree_policy() #Find maximum UCT node (which is leaf node)
            
            sequence = self.rollout_policy(current_node, self.budget) #Add node
            
            reward = self.get_reward(sequence)
            # print("cur_reward : " + str(reward))
            value_grad = self.get_value_grad(current_node, sequence, reward)
            # if(len(self.tree[sequence[0]])==4):
            #     self.tree[sequence[0]] = (self.tree[sequence[0]][0],self.tree[sequence[0]][1], self.tree[sequence[0]][2], self.tree[sequence[0]][3], value_grad )

            self.update_tree(reward, sequence, value_grad)
            ###TODO: After Finish Build functions for update
            # if(self.gradient_on):
            #     self.update_action(reward, sequence, None)
            # else:
            #     self.update_tree(reward, sequence, value_grad)
            
        # print(self.tree)
        # self.visualize_tree()
        best_sequence, cost = self.get_best_child()
        # print("best_sequence: ")
        # print(best_sequence)
        # print(self.tree[best_sequence])

        # update_ver = self.update_action(self.tree[best_sequence])
        if(self.gradient_on == True):
            update_ver = self.update_action(self.tree[best_sequence])
            return update_ver[0], cost
        else:
            return self.tree[best_sequence][0], cost

    '''
    Make sure update position does not go out of the ranges
    '''
    def update_action(self,best_sequence):
        grad_val = best_sequence[-1][:]
        grad_x = grad_val[0]
        grad_y = grad_val[1]

        # step_size = 0.05
        step_size = self.grad_step
        last_action = best_sequence[0][-1][:]

        last_action_x = last_action[0] + step_size * grad_x
        last_action_y = last_action[1] + step_size * grad_y

        if(last_action_x < self.ranges[0]):
            last_action_x = self.ranges[0] + 0.1
        elif(last_action_x > self.ranges[1]):
            last_action_x = self.ranges[1] - 0.1
        
        if(last_action_y < self.ranges[2]):
            last_action_y = self.ranges[2] + 0.1
        elif(last_action_y > self.ranges[3]):
            last_action_y = self.ranges[3] - 0.1

        last_action_update = (last_action_x, last_action_y, last_action[2])

        best_sequence[0][-1] = last_action_update

        # print("Modified")
        # print(best_sequence)

        return best_sequence

    def get_value_grad(self,cur_node, cur_seq, cur_reward): #best_seq: tuple (path sequence, path cost, reward, number of queries(called))
        
        init_node = self.tree[cur_seq[0][:]]
        path_seq = []
        for seq in cur_seq:
            for tmp_path_seq in self.tree[seq][0][:]:
                path_seq.append(tmp_path_seq)
        cur_node_reward = init_node[2]
        num_queri = init_node[3]
        
        if(len(path_seq)>=2):
            final_action = path_seq[-1]
            new_x_seq = path_seq[:]
            new_y_seq = path_seq[:]

            x = final_action[0]
            y = final_action[1]
            yaw = final_action[2]
            eps = 0.1
            step = 1.0
            x_dif = x + eps * step 
            y_dif = y + eps * step

            new_x_action = (x_dif, y, yaw)
            new_y_action = (x, y_dif, yaw)

            new_x_seq[-1] = new_x_action
            new_y_seq[-1] = new_y_action
            reward_x = self.get_reward(new_x_seq)
            reward_y = self.get_reward(new_y_seq)

            grad_x = (reward_x- cur_reward)/ eps
            grad_y = (reward_y - cur_reward) / eps
            # print("GRAD_X: "+ str(grad_x))
            # print("GRAD_Y: "+ str(grad_y))

        # print(init_node)
        # print(cur_reward)    
        
        return [grad_x, grad_y]

    def visualize_tree(self):
        ranges = (0.0, 20.0, 0.0, 20.0)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_xlim(ranges[0:2])
        ax.set_ylim(ranges[2:])
        for key, value in self.tree.items():
            cp = value[0][0]
            if(type(cp)==tuple):
                # print(cp)
                x = cp[0]
            # print('x ' + str(type(x))
                y = cp[1]
                plt.plot(x, y,marker='*')
        plt.show()

    def collision_check(self, path_dict):
        free_paths = {}
        for key,path in path_dict.items():
            is_collision = 0
            for pt in path:
                if(self.obstacle_world.in_obstacle(pt, 3.0)):
                    is_collision = 1
                    print("Collision Occured!")
            if(is_collision == 0):
                free_paths[key] = path
        
        return free_paths 
    
    def initialize_tree(self):
        '''Creates a tree instance, which is a dictionary, that keeps track of the nodes in the world'''
        tree = {}
        #(pose, number of queries)
        tree['root'] = (self.cp, 0)
        actions, _ = self.path_generator.get_path_set(self.cp)
        feas_actions = self.collision_check(actions)

        for action, samples in feas_actions.items():
            #(samples, cost, reward, number of times queried)
            cost = np.sqrt((self.cp[0]-samples[-1][0])**2 + (self.cp[1]-samples[-1][1])**2)
            tree['child '+ str(action)] = (samples, cost, 0, 0)
        return tree

    def tree_policy(self):
        '''Implements the UCB policy to select the child to expand and forward simulate'''
        # According to Arora:
        #avg_r average reward of all rollouts that have passed through node n
        #c_p some constant , 0.1 in literature
        #N number of times parent has been evaluated
        #n number of time node n has been evaluated
        #ucb = avg_r + c_p*np.sqrt(2*np.log(N)/n)
        leaf_eval = {}
        for i in xrange(self.frontier_size):
            node = 'child '+ str(i)
            if(node in self.tree): #If 'node' string key value is in current tree. 
                leaf_eval[node] = self.tree[node][2] + 0.1*np.sqrt(2*(np.log(self.tree['root'][1]))/self.tree[node][3])
#         print max(leaf_eval, key=leaf_eval.get)
        
        # print(max(leaf_eval, key=leaf_eval.get))
        return max(leaf_eval, key=leaf_eval.get)

    def rollout_policy(self, node, budget):
        '''Select random actions to expand the child node'''
        sequence = [node] #include the child node
        #TODO use the cost metric to signal action termination, for now using horizon
        for i in xrange(self.limit):
            actions, _ = self.path_generator.get_path_set(self.tree[node][0][-1]) #plan from the last point in the sample
            # feas_actions = self.collision_check(actions)
            a = np.random.randint(0,len(actions)) #choose a random path
            # while(not (a in feas_actions)):
            #     a = np.random.randint(0,len(actions)) #choose a random path
            
            
            #TODO add cost metrics
#             best_path = actions[a]
#             if len(best_path) == 1:
#                 best_path = [(best_path[-1][0],best_path[-1][1],best_path[-1][2]-1.14)]
#             elif best_path[-1][0] < -9.5 or best_path[-1][0] > 9.5:
#                 best_path = (best_path[-1][0],best_path[-1][1],best_path[-1][2]-1.14)
#             elif best_path[-1][1] < -9.5 or best_path[-1][0] >9.5:s
#                 best_path = (best_path[-1][0],best_path[-1][1],best_path[-1][2]-1.14)
#             else:
#                 best_path = best_path[-1]
            self.tree[node + ' child ' + str(a)] = (actions[a], 0, 0, 0) #add random path to the tree
            node = node + ' child ' + str(a)
            sequence.append(node)
        return sequence #return the sequence of nodes that are made

    def update_tree(self, reward, sequence, value_grad):
        '''Propogate the reward for the sequence'''
        #TODO update costs as well
        self.tree['root'] = (self.tree['root'][0], self.tree['root'][1]+1)
        for seq in sequence:
            # value_grad = 0
            if(len(self.tree[seq])>4):
                samples, cost, rew, queries, value_grad = self.tree[seq]
            else:
                samples, cost, rew, queries = self.tree[seq]
            queries += 1
            n = queries
            rew = ((n-1)*rew+reward)/n
            self.tree[seq] = (samples, cost, rew, queries)
            if(value_grad!=None):
                # print("In Here!")
                self.tree[seq] = (samples, cost, rew, queries, value_grad)


    def get_reward(self, sequence):
        '''Evaluate the sequence to get the reward, defined by the percentage of entropy reduction'''
        # The process is iterated until the last node of the rollout sequence is reached 
        # and the total information gain is determined by subtracting the entropies 
        # of the initial and final belief space.
        # reward = infogain / Hinit (joint entropy of current state of the mission)
        sim_world = self.GP
        samples = []
        obs = []
        for seq in sequence:
            # print("Type")
            # print(str(type(seq)))
            if(type(seq)== tuple):
                samples.append([seq])
            else:
                samples.append(self.tree[seq][0])
        obs = list(chain.from_iterable(samples))
        # print("###################################################3")
        # print("Samples: ")
        # print(obs)
        if(self.aquisition_function==aqlib.mves ):
            return self.aquisition_function(time = self.t, xvals = obs, param= [None], robot_model = sim_world)
        else:
            return self.aquisition_function(time = self.t, xvals = obs, robot_model = sim_world)
    
    def info_gain(self, xvals, robot_model):
        ''' Compute the information gain of a set of potential sample locations with respect to the underlying fucntion
            conditioned or previous samples xobs'''        
        data = np.array(xvals)
        x1 = data[:,0]
        x2 = data[:,1]
        queries = np.vstack([x1, x2]).T   
        xobs = robot_model.xvals

        # If the robot hasn't taken any observations yet, simply return the entropy of the potential set
        if xobs is None:
            Sigma_after = robot_model.kern.K(queries)
            entropy_after = 0.5 * np.log(np.linalg.det(np.eye(Sigma_after.shape[0], Sigma_after.shape[1]) \
                                        + robot_model.variance * Sigma_after))
            return (0.5*np.log(entropy_after), 0.5*(np.log(entropy_after)))

        all_data = np.vstack([xobs, queries])

        # The covariance matrices of the previous observations and combined observations respectively
        Sigma_before = robot_model.kern.K(xobs) 
        Sigma_total = robot_model.kern.K(all_data)       

        # The term H(y_a, y_obs)
        entropy_before = 2 * np.pi * np.e * np.linalg.det(np.eye(Sigma_before.shape[0], Sigma_before.shape[1]) \
                                        + robot_model.variance * Sigma_before)

        # The term H(y_a, y_obs)
        entropy_after = 2 * np.pi * np.e * np.linalg.det(np.eye(Sigma_total.shape[0], Sigma_total.shape[1]) \
                                        + robot_model.variance * Sigma_total)

        # The term H(y_a | f)
        entropy_total = 0.5 * np.log(entropy_after / entropy_before)

        ''' TODO: this term seems like it should still be in the equation, but it makes the IG negative'''
        #entropy_const = 0.5 * np.log(2 * np.pi * np.e * robot_model.variance)
        entropy_const = 0.0

        # This assert should be true, but it's not :(
        #assert(entropy_after - entropy_before - entropy_const > 0)
        #return entropy_total - entropsy_const
        return (entropy_total, 0.5*np.log(entropy_before))
    

    def get_best_child(self):
        '''Query the tree for the best child in the actions'''
        best = -1000
        best_child = None
        for i in xrange(self.frontier_size):
            if('child '+ str(i) in self.tree):
                r = self.tree['child '+ str(i)][2]
                if r > best:
                    best = r
                    best_child = 'child '+ str(i)
        return best_child, 0