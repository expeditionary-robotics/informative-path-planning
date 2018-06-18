# !/usr/bin/python

'''
A script for generating MCTS visualizations from a file with saved nonmyopic dataframes.

License: MIT
Maintainers: Genevieve Flaspohler and Victoria Preston
'''

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import graphviz
import os

def import_tree(filename):
    ''' Return the numpy array of the dictionary object saved to file'''
    return np.load(filename)

def get_trees(filepath):
    ''' Read the directory to get all of the relevant files to transform'''
    trees = {}
    for root, dirs, files in os.walk(filepath):
        for name in files:
            if 'tree' in name:
                trees[name.split('_')[1].split('.')[0]]=import_tree(filepath+name).item()
    return trees

def extract_paths(tree):
    ''' Get the last element in a sequence to re-assemble the path'''
    leaves = []
    for key, value in tree.items():
        temp = key.count('child')
        if temp >= 6:
            leaves.append(key)
    return leaves

def make_tree_graph(leaves):
    ''' Make a networkx representation of the paths '''
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

    return G, coords


def plot_trees(trees, path):
    ''' Plot the networkx representation '''
    for key, tree in trees.items():
        paths = extract_paths(tree)
        graph, pos = make_tree_graph(paths)
        plt.figure()
        plt.subplot(111)
        nx.draw(graph, pos, with_labels=True)
        plt.savefig(path+'mct_'+str(key))
        plt.close()
        # plt.show()


if __name__ == '__main__':
    path = 'figures/mean/'
    trees = get_trees(path)
    plot_trees(trees, path)
    
