#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
from biopandas.pdb import PandasPdb
import pandas as pd
import itertools
from code.unified_descriptors import utility

data_dir = '../../data'
protein_dir = os.path.join(data_dir, 'pdb_protein')
nano_dir = os.path.join(data_dir, 'pdb_np')


def CA_coord_os(pdb_name, chain1):
    coord = PandasPdb()
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
    distance_mat = utility.new_dist(c1_CA, c1_CA)
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
    
    coord2 = PandasPdb()
    coord2.read_pdb(os.path.join(nano_dir, np_name+'.pdb'))
    c2 = coord2.df['ATOM']
    c2_all = pd.concat([c2['x_coord'], c2['y_coord'], c2['z_coord']], axis=1).to_numpy()

    distance_mat = utility.new_dist(c2_all, c2_all)
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




