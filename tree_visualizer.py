# !/usr/bin/python

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import graphviz


def import_tree(filename):
	return np.load(filename)


if __name__ == '__main__':
	tree = import_tree('figures/mean/tree_1.npy')
	tree = tree.item()

	leaves = []
	for key, value in tree.items():
		temp = key.count('child')
		if temp >= 6:
			leaves.append(key)

	G = nx.DiGraph()
	last_child = 'root'
	G.add_node(last_child)

	nlists = [[] for m in range(3)]
	nlists[0].append('root')

	j = 0
	for leaf in leaves:
		i = 0
		temp = leaf.split(' ')
		for element in temp:
			if element == 'child':
				pass
			else:
				if i == 0:
					element = element
				elif i > 1:
					break
				else:
					element = element + '_' + str(i) + str(j)

				if element in list(G.nodes):
					pass
				else:
					nlists[i+1].append(element)
					G.add_node(element)
				G.add_edge(last_child, element)
				last_child = element
				i+=1
		j += 1
		last_child = 'root'

	coords = {'root': (0,0)}
	for m,l in enumerate(nlists[1:]):
		for i,n in enumerate(l):
			coords[n] = ((i-len(l)/2)*(50-49*m),-((m+1)))

	print nlists
	plt.subplot(111)
	pos = nx.shell_layout(G, nlists)
	pos = nx.kamada_kawai_layout(G, scale=100, center=(0,0))
	# nx.draw_networkx(G, alpha=0.3, with_labels=True)
	nx.draw(G, coords, with_labels=True)
	plt.show()