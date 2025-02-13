#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
from biopandas.pdb import PandasPdb
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
import rampy
from scipy.spatial.distance import euclidean
import itertools
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
from prody import *
from collections import Counter
import scipy.stats as stats
import pickle
import collections
import powerlaw
import operator
import random
import math
import csv



def new_dist(coord1, coord2):
    new_dist = np.zeros((len(coord1), len(coord2)))
    for ii in range(len(coord1)):
        for jj in range(len(coord2)):
            new_dist[ii][jj] = np.double(euclidean(coord1[ii], coord2[jj]))
    return new_dist

def nodes(c1):
    c1 = c1[c1['atom_name']== 'CA']
    nodes = c1['residue_number'].to_numpy()
    return nodes

def nodes_np(c1):
    nodes = c1['residue_number'].to_numpy()
    return nodes

def edges(c1, node, cutoff):
    c1 = c1[c1['atom_name']== 'CA']
    nodes = c1['residue_number'].to_numpy()
    connect = list(itertools.combinations(node, 2))
    edges = []
#     cutoff=7

    for ii in range(len(connect)):
        #print(ii)

        x0 = c1[c1['residue_number']==connect[ii][0]]['x_coord']
        y0 = c1[c1['residue_number']==connect[ii][0]]['y_coord']
        z0 = c1[c1['residue_number']==connect[ii][0]]['z_coord']
        xyz0 = pd.concat([x0, y0, z0]).to_numpy()

        x1 = c1[c1['residue_number']==connect[ii][1]]['x_coord']
        y1 = c1[c1['residue_number']==connect[ii][1]]['y_coord']
        z1 = c1[c1['residue_number']==connect[ii][1]]['z_coord']
        xyz1 = pd.concat([x1, y1, z1]).to_numpy()

        distance = euclidean(xyz0, xyz1)
        if distance <= cutoff:
            edges.append(connect[ii])

    return edges

def edges_np(c1, node, cutoff):
    connect = list(itertools.combinations(node, 2))
    edges = []
#     cutoff=7

    for ii in range(len(connect)):
        #print(ii)

        x0 = c1[c1['residue_number']==connect[ii][0]]['x_coord']
        y0 = c1[c1['residue_number']==connect[ii][0]]['y_coord']
        z0 = c1[c1['residue_number']==connect[ii][0]]['z_coord']
        xyz0 = pd.concat([x0, y0, z0]).to_numpy()

        x1 = c1[c1['residue_number']==connect[ii][1]]['x_coord']
        y1 = c1[c1['residue_number']==connect[ii][1]]['y_coord']
        z1 = c1[c1['residue_number']==connect[ii][1]]['z_coord']
        xyz1 = pd.concat([x1, y1, z1]).to_numpy()

        distance = euclidean(xyz0, xyz1)
        if distance <= cutoff:
            edges.append(connect[ii])

    return edges

def ollivier_ricci(node, edge, alpha):
    G = nx.Graph()
    G.add_nodes_from(node)
    G.add_edges_from(edge)
    orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
    orc.compute_ricci_curvature()
    c_sum = []
    for ii in node:
        L = list(orc.G[ii].values())
        c = []
        for jj in range(len(L)):
            c.append(list(L[jj].values())[1])

        c_sum.append(np.sum(c))

    return c_sum

def forman_ricci(node, edge):
    G = nx.Graph()
    G.add_nodes_from(node)
    G.add_edges_from(edge)
    frc = FormanRicci(G)
    frc.compute_ricci_curvature()
    c_sum = []
    for ii in node:
        L = list(frc.G[ii].values())
        c = []
        for jj in range(len(L)):
            c.append(list(L[jj].values())[0])

        c_sum.append(np.sum(c))

    return c_sum

def gnm_sum_mode(pdb_name, n, chain):
    protein = parsePDB(pdb_name)
    calphas1 = protein.select('calpha and chain %s' %chain)
    gnm1 = GNM('protein')
    gnm1.buildKirchhoff(calphas1, cutoff=7.)
    kirchhoff1 = gnm1.getKirchhoff()
    M1 = gnm1.calcModes('all')

    if gnm1.numModes() > n:
        print('num mode is larger than 10')
        mode_sq1 = []
        for ii in range(n):
            mode1 = gnm1[ii].getEigvecs().round(3)
            mode_sq1.append(mode1**2)
        sum_mode_sq1 = np.sum(mode_sq1, axis=0)
    else:
        print('num mode is smaller than 10')
        mode_sq1 = []
        for ii in range(gnm1.numModes()):
            mode1 = gnm1[ii].getEigvecs().round(3)
            mode_sq1.append(mode1**2)
        sum_mode_sq1 = np.sum(mode_sq1, axis=0)


    return sum_mode_sq1

def gnm_sum_mode_np(pdb_name, n):
    nano = parsePDB(pdb_name)
    gnm = GNM('nanoparticle')
    gnm.buildKirchhoff(nano, cutoff=4.)
    kirchhoff = gnm.getKirchhoff()
    M = gnm.calcModes('all')
    if gnm.numModes() > n:
        print('num mode is larger than 10')
        mode_sq1 = []
        for ii in range(n):
            mode1 = gnm[ii].getEigvecs().round(3)
            mode_sq1.append(mode1**2)
        sum_mode_sq1 = np.sum(mode_sq1, axis=0)
    else:
        print('num mode is smaller than 10')
        mode_sq1 = []
        for ii in range(gnm1.numModes()):
            mode1 = gnm1[ii].getEigvecs().round(3)
            mode_sq1.append(mode1**2)
        sum_mode_sq1 = np.sum(mode_sq1, axis=0)

    return sum_mode_sq1

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
    gn = nx.from_numpy_matrix(netn)

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

def CA_coord_os(pdb_name, chain1):
    coord = PandasPdb()
    protein_dir = os.path.join(os.getcwd(), 'proteins')
    coord.read_pdb(os.path.join(protein_dir, pdb_name+'.pdb'))
    c1 = coord.df['ATOM'][coord.df['ATOM']['chain_id']==chain1]
    c1_CA_only = c1[c1['atom_name']=='CA']

    c1_all = pd.concat([c1['x_coord'], c1['y_coord'], c1['z_coord']], axis=1).to_numpy()

    c1_CA = pd.concat([c1[c1['atom_name']=='CA']['x_coord'],
                      c1[c1['atom_name']=='CA']['y_coord'],
                      c1[c1['atom_name']=='CA']['z_coord']],
                      axis=1).to_numpy()


    return c1_CA_only, c1_all, c1_CA

MW = {'ALA':89.1,
     'ARG': 174.2,
     'ASN': 132.1,
     'ASP': 133.1,
     'CYS': 121.2,
     'GLU': 147.1,
     'GLN': 146.2,
     'GLY': 75.1,
     'HIS': 155.2,
     'ILE': 131.2,
     'LEU': 131.2,
     'LYS': 146.2,
     'MET': 149.2,
     'PHE': 165.2,
     'PRO': 115.1,
     'SER': 105.1,
     'THR': 119.1,
     'TRP': 204.2,
     'TYR': 181.2,
     'VAL': 117.1,
     'MOL': 120}

def n_nearest_coord(pdb_name, chain, n):

    c1_CA_df, c1_all, c1_CA = CA_coord_os(pdb_name, chain)
    distance_mat = new_dist(c1_CA, c1_CA)
    n_nearest_coord = []
    for ii in range(len(c1_CA)):
        n_nearest = distance_mat[ii].argsort()[:n]
        res_num = c1_CA_df.iloc[n_nearest,:]['residue_number']

        x = []
        y = []
        z = []
        res_mw = []
        for jj in range(n):
            x.append(c1_CA_df[c1_CA_df['residue_number']==res_num.iloc[jj]]['x_coord'].values)
            y.append(c1_CA_df[c1_CA_df['residue_number']==res_num.iloc[jj]]['y_coord'].values)
            z.append(c1_CA_df[c1_CA_df['residue_number']==res_num.iloc[jj]]['z_coord'].values)
            res_mw.append(MW[c1_CA_df[c1_CA_df['residue_number']==res_num.iloc[jj]]['residue_name'].values[0]])

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        res_mw = np.array(res_mw).reshape(-1,1)
        res_num = np.array(res_num).reshape(-1,1)


        n_nearest_coord.append(np.hstack([res_num, x, y, z, res_mw]))

    return n_nearest_coord

def n_nearest_coord_np(np_name, n):
    nano_dir = os.path.join(os.getcwd(), 'np')
    coord2 = PandasPdb()
    coord2.read_pdb(os.path.join(nano_dir, np_name+'.pdb'))
    c2 = coord2.df['ATOM']
    c2_all = pd.concat([c2['x_coord'], c2['y_coord'], c2['z_coord']], axis=1).to_numpy()

    distance_mat = new_dist(c2_all, c2_all)
    n_nearest_coord = []
    for ii in range(len(c2_all)):
        n_nearest = distance_mat[ii].argsort()[:n]
        res_num = c2.iloc[n_nearest,:]['residue_number']
        
        x = []
        y = []
        z = []
        res_mw = []

        for jj in range(n):
            x.append(c2[c2['residue_number']==res_num.iloc[jj]]['x_coord'].values)
            y.append(c2[c2['residue_number']==res_num.iloc[jj]]['y_coord'].values)
            z.append(c2[c2['residue_number']==res_num.iloc[jj]]['z_coord'].values)
            res_mw.append(MW[c2[c2['residue_number']==res_num.iloc[jj]]['residue_name'].values[0]])

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        res_mw = np.array(res_mw).reshape(-1,1)
        res_num = np.array(res_num).reshape(-1,1)

        n_nearest_coord.append(np.hstack([res_num, x, y, z, res_mw]))

    return n_nearest_coord



def nearest_coord(nearest):

    nearest_coord =  [[] for _ in range(len(nearest))]
    for ii in range(len(nearest)):
        nearest_coord[ii] = nearest[ii][:,1:4]
#         print(nearest_coord[ii])

    return nearest_coord

def weight(nearest):

    weight = [[] for _ in range(len(nearest))]
    for ii in range(len(nearest)):
        weight[ii] = nearest[ii][:,4]

    return weight

def osipov(coord, weight, N):
    n = 2
    m = 1

    P = list(itertools.permutations(np.arange(N),4))
    print(len(P))

    G = [[] for _ in range(len(coord))]
    r_ij = [[] for _ in range(len(P))]
    r_kl = [[] for _ in range(len(P))]
    r_il = [[] for _ in range(len(P))]
    r_jk = [[] for _ in range(len(P))]
    r_ij_mag = [[] for _ in range(len(P))]
    r_kl_mag = [[] for _ in range(len(P))]
    r_il_mag = [[] for _ in range(len(P))]
    r_jk_mag = [[] for _ in range(len(P))]

    mw = [[] for _ in range(len(P))]

    G_p_up = [[] for _ in range(len(P))]
    G_p_down = [[] for _ in range(len(P))]
    G_p = [[] for _ in range(len(P))]

    for ii in range(len(coord)):
        for kk in range(len(P)):
            r_ij[kk] = coord[ii][P[kk][0]]-coord[ii][P[kk][1]]
            r_kl[kk] = coord[ii][P[kk][2]]-coord[ii][P[kk][3]]
            r_il[kk] = coord[ii][P[kk][0]]-coord[ii][P[kk][3]]
            r_jk[kk] = coord[ii][P[kk][1]]-coord[ii][P[kk][2]]
            r_ij_mag[kk] = np.linalg.norm(r_ij[kk])
            r_kl_mag[kk] = np.linalg.norm(r_kl[kk])
            r_il_mag[kk] = np.linalg.norm(r_il[kk])
            r_jk_mag[kk] = np.linalg.norm(r_jk[kk])

            mw[kk] = weight[ii][P[kk][0]]*weight[ii][P[kk][1]]*weight[ii][P[kk][2]]*weight[ii][P[kk][3]]

            G_p_up[kk] = np.dot(np.cross(r_ij[kk], r_kl[kk]),r_il[kk])*(np.dot(r_ij[kk], r_jk[kk]))*(np.dot(r_jk[kk], r_kl[kk]))
            G_p_down[kk] = ((r_ij_mag[kk]*r_jk_mag[kk]*r_kl_mag[kk])**n)*((r_il_mag[kk])**m)
            G_p[kk] = mw[kk]*G_p_up[kk]/G_p_down[kk]

            G[ii].append(G_p[kk])


    G_os = [[] for _ in range(len(coord))]

    for ii in range(len(G_os)):
        G_os[ii] = (4*3*2*1)/((N)**4)*(1/3)*np.sum(G[ii])


    return G, G_os

def GOS(pdb_name, chain1, chain2):
    n_list = [5,7,10,15]
    G_os1 = [[] for _ in range(len(n_list))]
    G_os2 = [[] for _ in range(len(n_list))]

    for jj in range(len(n_list)):

        if n_list[jj] > len(CA_coord_os(pdb_name, chain1)[2]):
            nearest1 = n_nearest_coord(pdb_name, chain1,len(CA_coord(pdb_name, chain1)[2]))
            nearest_coord1 = nearest_coord(nearest1)
            mw1 = weight(nearest1)
            G_os1[jj] = (osipov(nearest_coord1, mw1, len(CA_coord(pdb_name, chain1)[2]))[1])

        else:
            nearest1 = n_nearest_coord(pdb_name, chain1, n_list[jj])
            nearest_coord1 = nearest_coord(nearest1)
            mw1 = weight(nearest1)
            G_os1[jj]=(osipov(nearest_coord1, mw1, n_list[jj])[1])

    G_os1_arr = np.hstack([np.array(G_os1[0]).reshape(-1,1),
                           np.array(G_os1[1]).reshape(-1,1),
                           np.array(G_os1[2]).reshape(-1,1),
                           np.array(G_os1[3]).reshape(-1,1)])

    for jj in range(len(n_list)):

        if n_list[jj] > len(CA_coord_os(pdb_name, chain2)[2]):
            nearest2 = n_nearest_coord(pdb_name, chain2, len(CA_coord_os(pdb_name, chain2)[2]))
            nearest_coord2 = nearest_coord(nearest2)
            mw2 = weight(nearest2)
            G_os2[jj] = (osipov(nearest_coord2, mw2, len(CA_coord_os(pdb_name, chain2)[2]))[1])

        else:
            nearest2 = n_nearest_coord(pdb_name, chain2, n_list[jj])
            nearest_coord2 = nearest_coord(nearest2)
            mw2 = weight(nearest2)
            G_os2[jj]=(osipov(nearest_coord2, mw2, n_list[jj])[1])

    G_os2_arr = np.hstack([np.array(G_os2[0]).reshape(-1,1),
                           np.array(G_os2[1]).reshape(-1,1),
                           np.array(G_os2[2]).reshape(-1,1),
                           np.array(G_os2[3]).reshape(-1,1)])

    return G_os1_arr, G_os2_arr

def GOS_np(pdb_name, chain1, np_name):

    nano_dir = os.path.join(os.getcwd(), 'np')
    coord2 = PandasPdb()
    coord2.read_pdb(os.path.join(nano_dir, np_name+'.pdb'))
    c2 = coord2.df['ATOM']
    c2_all = pd.concat([c2['x_coord'], c2['y_coord'], c2['z_coord']], axis=1).to_numpy()

    n_list = [5,7,10,15]
    G_os1 = [[] for _ in range(len(n_list))]
    G_os2 = [[] for _ in range(len(n_list))]

    for jj in range(len(n_list)):

        if n_list[jj] > len(CA_coord_os(pdb_name, chain1)[2]):
            G_os1[jj] = np.zeros((len(CA_coord_os(pdb_name, chain1)[2]),1))

        else:
            nearest1 = n_nearest_coord(pdb_name, chain1, n_list[jj])
            nearest_coord1 = nearest_coord(nearest1)
            mw1 = weight(nearest1)
            G_os1[jj]=(osipov(nearest_coord1, mw1, n_list[jj])[1])

    G_os1_arr = np.hstack([np.array(G_os1[0]).reshape(-1,1),
                           np.array(G_os1[1]).reshape(-1,1),
                           np.array(G_os1[2]).reshape(-1,1),
                           np.array(G_os1[3]).reshape(-1,1)])

    for jj in range(len(n_list)):

        if n_list[jj] > len(c2_all):
            G_os2[jj] = np.zeros((len(c2_all),1))

        else:
            nearest2 = n_nearest_coord_np(np_name, n_list[jj])
            nearest_coord2 = nearest_coord(nearest2)
            mw2 = weight(nearest2)
            G_os2[jj]=(osipov(nearest_coord2, mw2, n_list[jj])[1])

    G_os2_arr = np.hstack([np.array(G_os2[0]).reshape(-1,1),
                           np.array(G_os2[1]).reshape(-1,1),
                           np.array(G_os2[2]).reshape(-1,1),
                           np.array(G_os2[3]).reshape(-1,1)])

    return G_os1_arr, G_os2_arr




