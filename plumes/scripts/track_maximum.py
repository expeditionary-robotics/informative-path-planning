import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
	#seed900 stars
	# ax = 3.8
	# ay = 9.0

	# bx = 6.2
	# by = 1.0

	# stars = ['a' for i in range(15)] + \
	#         ['b' for i in range(3)] + \
	#         ['a' for i in range(5)] + \
	#         ['b' for i in range(7)] + \
	#         ['a' for i in range(1)] + \
	#         ['b' for i in range(2)] + \
	#         ['a' for i in range(3)] + \
	#         ['b' for i in range(4)] + \
	#         ['a' for i in range(10)] + \
	#         ['b' for i in range(1)] + \
	#         ['a' for i in range(2)] + \
	#         ['b' for i in range(1)] + \
	#         ['a' for i in range(2)] + \
	#         ['b' for i in range(5)] + \
	#         ['a' for i in range(19)] + \
	#         ['b' for i in range(20)] + \
	#         ['a' for i in range(1)] + \
	#         ['b' for i in range(9)] + \
	#         ['a' for i in range(1)] + \
	#         ['b' for i in range(14)] + \
	#         ['a' for i in range(1)] + \
	#         ['b' for i in range(3)] + \
	#         ['a' for i in range(1)] + \
	#         ['b' for i in range(2)] + \
	#         ['a' for i in range(5)] + \
	#         ['b' for i in range(1)] + \
	#         ['a' for i in range(62)]

	#seed500 stars
	# ax = 1.9
	# ay = 6.5

	# bx = 6.5
	# by = 5.8

	# stars = ['a' for i in range(19)] + \
	#         ['b' for i in range(1)] + \
	#         ['a' for i in range(30)] + \
	#         ['b' for i in range(10)] + \
	#         ['a' for i in range(7)] + \
	#         ['b' for i in range(93)] + \
	#         ['a' for i in range(1)] + \
	#         ['b' for i in range(6)] + \
	#         ['a' for i in range(1)] + \
	#         ['b' for i in range(17)] + \
	#         ['a' for i in range(13)]

	robot_poses = pd.read_table('thesis_rbf3d_experiments/sim_seed500-nonmyopicTrue-treedpw/figures/gumbel/robot_model.csv', sep=' ', dtype=float, header=None)
	robot_poses = robot_poses.T
	print robot_poses.head(5)
	
	sx = []
	sy = []
	for star in stars:
		if star == 'a':
			sx.append(ax)
			sy.append(ay)
		else:
			sx.append(bx)
			sy.append(by)

	# plt.scatter(sx, sy)
	# plt.scatter(robot_poses.iloc[:,0].values[::3], robot_poses.iloc[:,1].values[::3])
	# plt.show()
	# plt.close()

	# plt.plot(sx)
	# plt.plot(robot_poses.iloc[:,0].values[::3])
	# plt.show()
	# plt.close()

	dist = []
	reward = []
	r = 0
	past_star = 'a'
	toggles = []
	for i, star in enumerate(stars):
		try:
			dx = (np.mean(robot_poses.iloc[3*i:3*i+3,0]) - sx[i])**2
			dy = (np.mean(robot_poses.iloc[3*i:3*i+3,1]) - sy[i])**2
			dist.append(np.sqrt(dx + dy))
			if np.sqrt(dx + dy) < 1.5:
				r += 3
			reward.append(r)
			if past_star != star:
				toggles.append(i)
			past_star = star
		except:
			break


	plt.plot(dist)
	plt.plot(np.zeros(len(dist)))
	for t in toggles:
		plt.gca().axvline(t, c='r')
	plt.show()
	plt.close()

	plt.plot(reward)
	plt.show()
	plt.close()

