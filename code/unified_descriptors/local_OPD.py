#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os
from biopandas.pdb import PandasPdb
import pandas as pd
import itertools
from sklearn.neighbors import NearestNeighbors

data_dir = os.getcwd()
protein_dir = os.path.join(data_dir, 'pdb_protein')
nano_dir = os.path.join(data_dir, 'pdb_np')

MW = {'ALA': 89.1,
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


def n_nearest_coord(c1, n):
    # print(c1['x_coord'])
    c1_all = pd.concat([c1['x_coord'], c1['y_coord'], c1['z_coord']], axis=1).to_numpy()

    nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(c1_all)

    distances, indices = nbrs.kneighbors(c1_all)
    n_nearest = [np.squeeze(i) for i in c1_all[indices]]
    names = np.array(c1['residue_name'])

    weight = [[] for _ in range(len(n_nearest))]
    for ii in range(len(indices)):
        weight[ii] = list(map(MW.get, names[indices[ii]]))

    return np.array(n_nearest), np.array(weight)


def osipov(coord, weight):
    n = 2
    m = 1

    N = len(coord)

    P = np.array(list(itertools.permutations(np.arange(N), 4)))  # Get permutations

    coords_P = coord[P]
    r = coords_P - np.roll(coords_P, -1, axis=1)
    r[:, 3] = -r[:, 3]
    r_mag = np.linalg.norm(r, axis=-1)

    # try:
    mw_all = weight[P].prod(axis=-1)
    # except:
    #     print('no weight used or weight invalid')
    #     mw_all = np.ones(len(P))

    cross_vecs = np.cross(r[:, 0], r[:, 2])

    G_p_up = np.einsum('ij,ij->i', cross_vecs, r[:, 3]) * np.einsum('ij,ij->i', r[:, 0], r[:, 1]) * np.einsum(
        'ij,ij->i', r[:, 1], r[:, 2])
    G_p_down = np.power(np.prod(r_mag[:, 0:3], axis=-1), n) * np.power(r_mag[:, 3], m)

    G_p_weighted = (1 / 3) * np.sum(mw_all * G_p_up / G_p_down)
    G_p = (1 / 3) * np.sum(G_p_up / G_p_down)

    G_os_weighted = (24) / (N ** 4) * G_p_weighted
    G_os = (24) / (N ** 4) * G_p

    return G_os_weighted, G_os



def GOS_single(c1):

    n_list = [5, 7 , 10, 15]
    G_os1 = [[] for _ in range(len(n_list))]


    for jj in range(len(n_list)):
    # if n_list[jj] > len(c1):
        n_nearest1, weight1 = n_nearest_coord(c1, n_list[jj])
        for ii in range(len(n_nearest1)):
            G_os1[jj].append(osipov(n_nearest1[ii], weight1[ii])[0])

        # else:
        #     nearest1, weight1 = n_nearest_coord(c1, n_list[jj])
        #     G_os1[jj] = (osipov(nearest1, weight1)[1])

    G_os1_arr = np.hstack([np.array(G_os1[0]).reshape(-1, 1),
                           np.array(G_os1[1]).reshape(-1, 1),
                           np.array(G_os1[2]).reshape(-1, 1),
                           np.array(G_os1[3]).reshape(-1, 1)])
    # np.array(G_os2[2]).reshape(-1,1),
    # np.array(G_os2[3]).reshape(-1,1)])

    return G_os1_arr


# def GOS_np(c1, c2):
    
#     c2_all = pd.concat([c2['x_coord'], c2['y_coord'], c2['z_coord']], axis=1).to_numpy()

#     n_list = [5, 7]
#     G_os1 = [[] for _ in range(len(n_list))]
#     G_os2 = [[] for _ in range(len(n_list))]
            
#     for jj in range(len(n_list)):

#         if n_list[jj] > len(c1):
#             n_nearest1, weight1 = n_nearest_coord(c1, n_list[jj])
#             G_os1[jj] = (osipov(n_nearest1, weight1, len(c1))[1])

#         else:
#             n_nearest1, weight1 = n_nearest_coord(c1, n_list[jj])
#             G_os1[jj] = (osipov(n_nearest1, weight1, n_list[jj])[1])

#     G_os1_arr = np.hstack([np.array(G_os1[0]).reshape(-1, 1),
#                           np.array(G_os1[1]).reshape(-1, 1)])

#     for jj in range(len(n_list)):

#         if n_list[jj] > len(c2_all):
#             n_nearest2, weight2 = n_nearest_coord(c2_all, n_list[jj])
#             G_os1[jj] = (osipov(n_nearest1, weight1, len(c1))[1])

#         else:
#             nearest2, weight2 = n_nearest_coord(c2_all, n_list[jj])
#             G_os2[jj] = (osipov(nearest2, weight2, n_list[jj])[1])

#     G_os2_arr = np.hstack([np.array(G_os2[0]).reshape(-1, 1),
#                           np.array(G_os2[1]).reshape(-1, 1)])

#     return G_os1_arr, G_os2_arr
