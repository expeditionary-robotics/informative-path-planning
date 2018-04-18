import numpy as np
import matplotlib.pyplot as plt




class MCTS():
	def __init__(self, budget, belief, initial_pose):
		self.budget = budget
		self.GP = belief
		self.cp = initial_pose

		self.spent = 0

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
		# initialize tree
		# 
		return best_sequence, cost

	def execute_actions(self, best_sequence):
		return observations


if __name__ == '__main__':
	run_search()
