import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import gaussian_kde
import seaborn as sns

if __name__ == '__main__':
	accumulated_rewards_dpw = {} #store MSS reward
	accumulated_rewards_belief = {}
	seeds_b = []
	seeds_d = []
	epsilon = 1.5
	root = 'thesis_rbf3d_stat_experiments'

	# want to walk through the directory
	star_paths = []
	sample_paths = []
	for subdir, dirs, files in os.walk(root):
		for file in files:
			path = os.path.join(subdir, file)
			if 'log' in path:
				star_paths.append(path)
			if 'robot_model' in path:
				sample_paths.append(path)

	# let's now read through everything!
	for star_file in star_paths:
		max_loc_x = []
		max_loc_y = []
		#get the seed
		seed = star_file.split('/')[1].split('-')[0].split('_')[1][4:]
		#get the planner
		if 'belief' in star_file:
			planner = 'belief'
		else:
			planner = 'dpw'
		#now extract the maxima for the file
		temp = open(star_file, "r")
		for l in temp.readlines():
			if "Generated with maxima" in l:
				max_loc_x.append(float(l.split(':')[3].split(',')[0]))
				max_loc_y.append(float(l.split(':')[3].split(',')[1]))

		#great! now let's find the matching robot mission
		for robot_file in sample_paths:
			if planner in robot_file and 'sim_seed'+seed in robot_file:
				#it's a match!
				# print robot_file, planner, seed
				robot_poses = pd.read_table(robot_file, sep=' ', dtype=float, header=None)
				robot_poses = robot_poses.T
				r = 0
				for i in range(len(max_loc_x)):
					if i < 150:
						try:
							dx = (np.mean(robot_poses.iloc[3*i:3*i+3,0]) - max_loc_x[i])**2
							dy = (np.mean(robot_poses.iloc[3*i:3*i+3,1]) - max_loc_y[i])**2
							dist = np.sqrt(dx + dy)
							if dist <= epsilon:
								r += 3
						except:
							print 'exception'
							break
					else:
						print 'overflow'
						break
				if planner == 'belief':
					accumulated_rewards_belief[seed] = r
				else:
					accumulated_rewards_dpw[seed] = r


	#now, get the kernel density estimates
	# density_dpw = gaussian_kde(accumulated_rewards_dpw)
	# ds = np.linspace(0, np.nanmax(accumulated_rewards_dpw),200)

	# density_belief = gaussian_kde(accumulated_rewards_belief)
	# bs = np.linspace(0, np.nanmax(accumulated_rewards_belief),200)
	# # density_dpw.covariance_factor = lambda : .25
	# # density_dpw._compute_covariance()
	# plt.plot(ds, density_dpw(ds))
	# plt.plot(bs, density_belief(bs))
	# plt.show()

	# sns.distplot(accumulated_rewards_dpw, hist=True, kde=False, bins=10)
	# sns.distplot(accumulated_rewards_belief, hist=True, kde=False, bins=10)
	# plt.gca().set_xlim([0, 600])
	# plt.show()

	bel = []
	dpw = []
	for k, v in accumulated_rewards_belief.items():
		if float(k) < 1000.:
			bel.append(v)
			dpw.append(accumulated_rewards_dpw[k])


	plt.figure()
	plt.scatter(bel, dpw)
	plt.plot(np.linspace(0,200,10), np.linspace(0,200,10))
	plt.plot(np.linspace(0,200,10), 50*np.ones(len(np.linspace(0,200,10))), c='r')
	plt.gca().set_xlim([0, 200])
	plt.gca().axvline(50, c='r')
	# plt.gca().axis('square')
	plt.show()


	dpw900 = [] #store distance
	belief900 = []
	epsilon = 1.5
	root = 'thesis_rbf3d_stat_experiments'

	# want to walk through the directory
	star_paths = []
	sample_paths = []
	for subdir, dirs, files in os.walk(root):
		for file in files:
			path = os.path.join(subdir, file)
			if 'log' in path and '900' in path:
				star_paths.append(path)
			if 'robot_model' in path and '900' in path:
				sample_paths.append(path)

	# let's now read through everything!
	for star_file in star_paths:
		max_loc_x = []
		max_loc_y = []
		#get the seed
		seed = star_file.split('/')[1].split('-')[0].split('_')[1][4:]
		#get the planner
		if 'belief' in star_file:
			planner = 'belief'
		else:
			planner = 'dpw'
		#now extract the maxima for the file
		temp = open(star_file, "r")
		for l in temp.readlines():
			if "Generated with maxima" in l:
				max_loc_x.append(float(l.split(':')[3].split(',')[0]))
				max_loc_y.append(float(l.split(':')[3].split(',')[1]))

		#great! now let's find the matching robot mission
		for robot_file in sample_paths:
			if planner in robot_file and 'sim_seed'+seed in robot_file:
				#it's a match!
				# print robot_file, planner, seed
				robot_poses = pd.read_table(robot_file, sep=' ', dtype=float, header=None)
				robot_poses = robot_poses.T
				r = 0
				for i in range(len(max_loc_x)):
					if i < 150:
						try:
							dx = (np.mean(robot_poses.iloc[3*i:3*i+3,0]) - max_loc_x[i])**2
							dy = (np.mean(robot_poses.iloc[3*i:3*i+3,1]) - max_loc_y[i])**2
							dist = np.sqrt(dx + dy)
							if planner == 'belief':
								belief900.append(dist)
							else:
								dpw900.append(dist)
							if dist <= epsilon:
								r += 3
						except:
							print 'exception'
							break
					else:
						print 'overflow'
						break

		plt.plot(belief900)
		plt.plot(dpw900)
		plt.show()
