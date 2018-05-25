import numpy as np
import matplotlib.pyplot as plt

''' This is a toybox for MCTS development for now. This will not run.'''


class MCTS():
	def __init__(self, budget, belief, initial_pose, horizon):
		self.budget = budget
		self.GP = belief
		self.cp = initial_pose
		self.horizon = horizon

		self.spent = 0
		self.tree = None

	def plan(self):
		while self.spent < self.budget:
			pose = self.get_pose()
			best_sequence, cost = self.get_actions(pose, self.budget, self.belief)
			observations = self.execute_actions(best_sequence)
			self.GP.add_data(observations)
			self.spent += cost

	def get_pose(self):
		return self.cp

	def get_actions(self, pose, budget, belief):
		self.tree = self.initialize_tree(pose, budget)
		self.tree[pose] = (budget, 0)

		while True:
			current_node = self.tree_policy()
			sequence = self.rollout_policy(current_node, budget)
			reward = self.get_reward(sequence, belief)
			self.update_tree(sequence, reward)

		best_sequence, cost = self.get_best_child()
		return best_sequence, cost

	def initialize_tree(self, pose, budget):
		tree = {}
		# tuple (pose, budget, number of times queried)
		tree['root'] = (pose, budget, 0)
		actions = self.get_action_set(pose)
		for action, samples in actions.items():
			# tuple (samples, budget, reward, number of times queried)
			tree['child '+ str(action)] = (samples, budget, 0, 0)
		return tree


	def tree_policy(self):
		#avg_r average reward of all rollouts that have passed through node n
		#c_p some constant , 0.1 in literature
		#N number of times parent has been evaluated
		#n number of time node n has been evaluated
		#ucb = avg_r + c_p*np.sqrt(2*np.log(N)/n)
		leaf_eval = {}
		for node, samples in self.tree.items():
			if node == 'root':
				pass
			else:
				leaf_eval[node] = self.tree[node][2] + 0.1*np.sqrt(2*(np.log(self.tree['root'][2]))/self.tree[node][3])
		return self.tree[max(leaf_eval, key=leaf_eval.get())]

	def rollout_policy(self, node, budget):
		sequence = [node]
		for i in xrange(self.horizon):
			actions = self.get_action_set(self.tree[node][0][-1])
			a = np.random.randint(0,self.frontier_size)
			self.tree[node + ' child ' + str(a)] = (actions[a], budget, 0, 0)
			node = node + ' child ' + str(a)
			sequence.append(node)
		return sequence

	def get_reward(self, sequence, belief):
		# The process is iterated until the last node of the rollout sequence is reached 
		# and the total information gain is determined by subtracting the entropies 
		# of the initial and final belief space.
		# reward = infogain / Hinit (joint entropy of current state of the mission)
		sim_world = self.GP
		for seq in sequence:
			observations = sim_world.predict(self.tree[seq][0])
			sim_world.add_data(observations)
		info_gain = (self.GP.entropy() - sim_world.entropy())
		return info_gain / self.GP.entropy()

	def update_tree(reward, sequence):
		for seq in sequence:
			self.tree[seq][3] += 1
			n = self.tree[seq][3]
			self.tree[seq][2] = ((n-1)*(self.tree[seq][2]) + reward)/n

	def get_best_child():
		best = 0
		best_child = None
		for i in xrange(self.frontier_size):
			r = tree['child '+ str(i)][2]
			if r > best:
				best = r
				best_child = 'child '+ str(i)
		return best_child


	def execute_actions(self, best_sequence):
		return observations


if __name__ == '__main__':
	run_search()
