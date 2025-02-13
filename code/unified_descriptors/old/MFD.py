#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
import networkx as nx
from collections import Counter
import scipy.stats as stats
import pickle
import collections
import powerlaw
import operator
import random
import math
import csv



def network_G(node, edge):
    G = nx.Graph()
    G.add_nodes_from(node)
    G.add_edges_from(edge)
    return G


def node_dimension_all(G,weight=None):
    node_dimension = {}
    for node in G.nodes():
        grow = []
        r_g = []
        num_g = []
        num_nodes = 0
        if weight == None:
            spl = nx.single_source_shortest_path_length(G,node)
        else:
            spl = nx.single_source_dijkstra_path_length(G,node)
        for s in spl.values():
            if s>0:
                grow.append(s)
        grow.sort()
        num = Counter(grow)
        for i,j in num.items():
            num_nodes += j
            if i>0:
                #if np.log(num_nodes) < 0.95*np.log(G.number_of_nodes()):
                r_g.append(i)
                num_g.append(num_nodes)
        x = np.log(r_g)
        y = np.log(num_g)
        if len(r_g) < 3:
            print("local",node)
        if len(r_g) > 1:
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            except:
                print(node)
            node_dimension[node] = slope
        else:
            node_dimension[node] = 0
    return node_dimension


def fractal_dimension(node, edge):

    G = network_G(node, edge)
    max_sub = max(nx.connected_components(G), key=len)
    max_subnet = G.subgraph(max_sub)
    netn = nx.to_numpy_array(max_subnet)
    gn = nx.from_numpy_array(netn)

    node_d_val = list(node_dimension_all(max_subnet,weight=None).values())
    node_d_val_arr = np.array(node_d_val)
    node_d_val_arr = node_d_val_arr.reshape((-1,1))
    node_d_key = list(node_dimension_all(max_subnet,weight=None).keys())
    node_d_key_arr = np.array(node_d_key)
    node_d_key_arr = node_d_key_arr.reshape((-1,1))
    node_max_dimesion = np.hstack((node_d_key_arr, node_d_val_arr))

    if len(node) == nx.number_of_nodes(gn):

        node_d = node_max_dimesion

    else:
#     if len(node) != (nx.number_of_nodes(gn)):
        min_sub = min(nx.connected_components(G), key=len)
        min_subnet = G.subgraph(min_sub)

        node_d_min_val = list(node_dimension_all(min_subnet,weight=None).values())
        node_d_min_val_arr = np.array(node_d_min_val)
        node_d_min_val_arr = node_d_min_val_arr.reshape((-1,1))
        node_d_min_key = list(node_dimension_all(min_subnet,weight=None).keys())
        node_d_min_key_arr = np.array(node_d_min_key)
        node_d_min_key_arr = node_d_min_key_arr.reshape((-1,1))

        node_min_dimension = np.hstack((node_d_min_key_arr, node_d_min_val_arr))
        node_d = np.vstack((node_min_dimension, node_max_dimesion))
        node_d = node_d[np.argsort(node_d[:,0])]

    return node_d

def node_dimension_single(G,node):
    grow = []
    r_g = []
    num_g = []
    #r_g_all = []
    #num_g_all = []
    num_nodes = 0
    spl = nx.single_source_shortest_path_length(G,node)
    for s in spl.values():
        if s>0:
            grow.append(s)
    grow.sort()
    num = Counter(grow)
    for i,j in num.items():
        num_nodes += j
        # adjust!!!
        if i>0:
            r_g.append(i)
            num_g.append(num_nodes)
    x = np.log(r_g)
    y = np.log(num_g)
    slope, intercept, r_value, p_value, std_err  = stats.linregress(x, y)
    dimension = slope
    plt.plot(x,y,'o',label='fitted data')
    plt.plot(x,intercept + slope*x,'r',label='fitted line')
    plt.title("Node Dimension of node:"+str(node)+'='+str(slope))
    return dimension,r_value**2
    #return dimension

#Generate FD: calculate path for the whole network
def FD(G,d_r,weight=None):
    #FD = []
    FD = {}
    for node in G.nodes():
        grow = []
        num_g = []
        num_nodes = 0
        if weight == None:
            spl = nx.single_source_shortest_path_length(G,node)
        else:
            spl = nx.single_source_dijkstra_path_length(G,node)
        for s in spl.values():
            if s>0:
                grow.append(s)
        grow.sort()
        num = Counter(grow)
        for i,j in num.items():
#             Choose if you need to normalize:
#             num_nodes += j
            num_nodes += j/(nx.number_of_nodes(G)-1)
            if i > 0 and i < d_r:
                num_g.append(num_nodes)
        for _ in range(i,d_r-1):
            num_g.append(num_nodes)
        FD[node] = num_g
#         FD.append(num_g)
    return FD

def more_box(node, edge, d_r):

    G = network_G(node, edge)

    max_sub = max(nx.connected_components(G), key=len)
    max_subnet = G.subgraph(max_sub)
    netn = nx.to_numpy_array(max_subnet)
    gn = nx.from_numpy_array(netn)

    max_FD_val = list(FD(max_subnet, d_r).values())
    max_FD_val_arr = np.array(max_FD_val)

    max_FD_key = list(FD(max_subnet, d_r).keys())
    max_FD_key_arr = np.array(max_FD_key)
    max_FD_key_arr = max_FD_key_arr.reshape((-1,1))

    max_FD = np.hstack((max_FD_key_arr, max_FD_val_arr))

    if len(node) == nx.number_of_nodes(gn):

        F_D = max_FD

    else:
#     if len(node) != (nx.number_of_nodes(gn)):
        min_sub = min(nx.connected_components(G), key=len)
        min_subnet = G.subgraph(min_sub)

        min_FD_val = list(FD(min_subnet, d_r).values())
        min_FD_val_arr = np.array(min_FD_val)

        min_FD_key = list(FD(min_subnet, d_r).keys())
        min_FD_key_arr = np.array(min_FD_key)
        min_FD_key_arr = min_FD_key_arr.reshape((-1,1))

        min_FD = np.hstack((min_FD_key_arr, min_FD_val_arr))

        F_D = np.vstack((min_FD, max_FD))
        F_D = F_D[np.argsort(F_D[:,0])]


    return F_D


