import numpy as np
import os
from biopandas.pdb import PandasPdb
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import Bio
from Bio.PDB import PDBParser
from scipy.spatial.distance import euclidean
import itertools
import utility
import GT_curvature, GNM, MFD, local_OPD
import sys
import time
from sklearn import preprocessing

def CA_coord(pdb_name, chain1, np_name):
    coord1 = PandasPdb()
    coord2 = PandasPdb()
    coord1.read_pdb(os.path.join(protein_dir, pdb_name+'.pdb'))
    coord2.read_pdb(os.path.join(nano_dir, np_name+'.pdb'))
    c1 = coord1.df['ATOM'][coord1.df['ATOM']['chain_id']==chain1]
    c2 = coord2.df['ATOM']
    
    c1_all = pd.concat([c1['x_coord'], c1['y_coord'], c1['z_coord']], axis=1).to_numpy()
    c2_all = pd.concat([c2['x_coord'], c2['y_coord'], c2['z_coord']], axis=1).to_numpy()
    
    c1_CA = pd.concat([c1[c1['atom_name']=='CA']['x_coord'], 
                      c1[c1['atom_name']=='CA']['y_coord'], 
                      c1[c1['atom_name']=='CA']['z_coord']], 
                      axis=1).to_numpy()
    
    return c1, c2, c1_CA, c2_all


def distance_data_range(c1, c2, c1_CA, c2_all):
    data = pd.DataFrame(np.zeros((1, 10)))
    # data = data.astype('object')
    data.columns=['min_dist','CA_CA', 'CA_1.3-3.0', 'CA_3.0-4.0', 
                  'CA_4.0-5.0','CA_5.0-6.0','CA_6.0-7.0', 'CA_7.0-8.0', 'CA_8.0-9.0', 'CA_9.0-10']

    # Need to change combintaiton of chain_0_CA / chain_1_CA / chain_2_CA
    updated_dist = utility.new_dist(c1_CA, c2_all) 

    min_val = np.amin(updated_dist)
    CA_no = len(np.argwhere((updated_dist<10) & (updated_dist>1.3)))
    
    data.iloc[:,0] = min_val
    data.iloc[:,1] = CA_no
    data.iloc[:,2] = len(np.argwhere((updated_dist>1.3) & (updated_dist<=3.0)))
    data.iloc[:,3] = len(np.argwhere((updated_dist>3.0) & (updated_dist<=4.0)))
    data.iloc[:,4] = len(np.argwhere((updated_dist>4.0) & (updated_dist<=5.0)))
    data.iloc[:,5] = len(np.argwhere((updated_dist>5.0) & (updated_dist<=6.0)))
    data.iloc[:,6] = len(np.argwhere((updated_dist>6.0) & (updated_dist<=7.0)))
    data.iloc[:,7] = len(np.argwhere((updated_dist>7.0) & (updated_dist<=8.0)))
    data.iloc[:,8] = len(np.argwhere((updated_dist>8.0) & (updated_dist<=9.0)))
    data.iloc[:,9] = len(np.argwhere((updated_dist>9.0) & (updated_dist<=10.0)))
    
    CA_all =np.argwhere((updated_dist<np.max(updated_dist.flatten()+1)))
    CA_all_res = np.zeros((len(CA_all),2))
    for ii in range(len(CA_all)):
        CA_all_res[ii,0] =c1[c1['atom_name']=='CA'].iloc[CA_all[ii,0],:]['residue_number']
        CA_all_res[ii,1] =c2.iloc[CA_all[ii,1],:]['residue_number']
    print('Len_CA_all', len(CA_all))
    print('Len_CA_all_res', len(CA_all_res))
    
    pair_table = pd.DataFrame(np.zeros((((len(updated_dist.flatten())),5))))
    # CA_table = CA_table.astype('object')
    pair_table.columns=['c1_residue_number','c1_residue_name', 'c2_residue_number', 'c2_residue_name', 'distance']

    pair_table.iloc[:,0] = CA_all_res[:,0].astype(int)
    pair_table.iloc[:,2] = CA_all_res[:,1].astype(int)

    for ii in range(len(pair_table)):
        pair_table.iloc[ii,1] = c1['residue_name'][c1['residue_number'] == pair_table.iloc[ii,0]].reset_index().iloc[0,1]
        pair_table.iloc[ii,3] = c2['residue_name'][c2['residue_number'] == pair_table.iloc[ii,2]].reset_index().iloc[0,1]
        pair_table.iloc[ii,4] = updated_dist[CA_all[ii,0], CA_all[ii,1]]

#     pair_table.head()
    
    return data, pair_table


def binary_fill(data_Feature, pair_table):
    data_Feature['c1'] = pair_table['c1_residue_name']
    data_Feature['c2'] = pair_table['c2_residue_name']
    for ii in range(len(pair_table)):
        
        #c1
        if pair_table['c1_residue_name'].iloc[ii] in ['ARG','LYS','HIS','MOL']:
            data_Feature['c1_pos'].iloc[ii] = 1
        else: 
            data_Feature['c1_pos'].iloc[ii] = 0
    
        if pair_table['c1_residue_name'].iloc[ii] in ['ASP','GLU']:
            data_Feature['c1_neg'].iloc[ii] = 1
        else: 
            data_Feature['c1_neg'].iloc[ii] = 0
        
        if pair_table['c1_residue_name'].iloc[ii] in ['GLN','ASN', 'SER', 'THR', 'TYR', 'CYS']:
            data_Feature['c1_polar'].iloc[ii] = 1
        else: 
            data_Feature['c1_polar'].iloc[ii] = 0
    
        if pair_table['c1_residue_name'].iloc[ii] in ['TRP','TYR', 'MET']:
            data_Feature['c1_amp'].iloc[ii] = 1
        else: 
            data_Feature['c1_amp'].iloc[ii] = 0
    
        if pair_table['c1_residue_name'].iloc[ii] in ['ALA','ILE', 'LEU', 'MET', 'PHE', 'VAL', 'PRO', 'GLY']:
            data_Feature['c1_hp'].iloc[ii] = 1
        else: 
            data_Feature['c1_hp'].iloc[ii] = 0
        
        #c2
        if pair_table['c2_residue_name'].iloc[ii] in ['ARG','LYS','HIS', 'MOL']:
            data_Feature['c2_pos'].iloc[ii] = 1
        else: 
            data_Feature['c2_pos'].iloc[ii] = 0
    
        if pair_table['c2_residue_name'].iloc[ii] in ['ASP','GLU']:
            data_Feature['c2_neg'].iloc[ii] = 1
        else: 
            data_Feature['c2_neg'].iloc[ii] = 0
        
        if pair_table['c2_residue_name'].iloc[ii] in ['GLN','ASN', 'SER', 'THR', 'TYR', 'CYS']:
            data_Feature['c2_polar'].iloc[ii] = 1
        else: 
            data_Feature['c2_polar'].iloc[ii] = 0
    
        if pair_table['c2_residue_name'].iloc[ii] in ['TRP','TYR', 'MET']:
            data_Feature['c2_amp'].iloc[ii] = 1
        else: 
            data_Feature['c2_amp'].iloc[ii] = 0
    
        if pair_table['c2_residue_name'].iloc[ii] in ['ALA','ILE', 'LEU', 'MET', 'PHE', 'VAL', 'PRO', 'GLY']:
            data_Feature['c2_hp'].iloc[ii] = 1
        else: 
            data_Feature['c2_hp'].iloc[ii] = 0
   
    return data_Feature


def hydrophobicity_fill(data_Feature, pair_table):
    for ii in range(len(pair_table)):  
    #c1
        if pair_table['c1_residue_name'].iloc[ii] in ['MOL']:
            data_Feature['c1_hp_idx'].iloc[ii] = 4.5
        if pair_table['c1_residue_name'].iloc[ii] in ['ILE']:
            data_Feature['c1_hp_idx'].iloc[ii] = 4.5
        if pair_table['c1_residue_name'].iloc[ii] in ['VAL']:
            data_Feature['c1_hp_idx'].iloc[ii] = 4.2
        if pair_table['c1_residue_name'].iloc[ii] in ['LEU']:
            data_Feature['c1_hp_idx'].iloc[ii] = 3.8
        if pair_table['c1_residue_name'].iloc[ii] in ['PHE']:
            data_Feature['c1_hp_idx'].iloc[ii] = 2.8
        if pair_table['c1_residue_name'].iloc[ii] in ['CYS']:
            data_Feature['c1_hp_idx'].iloc[ii] = 2.5
        if pair_table['c1_residue_name'].iloc[ii] in ['MET']:
            data_Feature['c1_hp_idx'].iloc[ii] = 1.9
        if pair_table['c1_residue_name'].iloc[ii] in ['ALA']:
            data_Feature['c1_hp_idx'].iloc[ii] = 1.8
        if pair_table['c1_residue_name'].iloc[ii] in ['GLY']:
            data_Feature['c1_hp_idx'].iloc[ii] = -0.4
        if pair_table['c1_residue_name'].iloc[ii] in ['THR']:
            data_Feature['c1_hp_idx'].iloc[ii] = -0.7
        if pair_table['c1_residue_name'].iloc[ii] in ['SER']:
            data_Feature['c1_hp_idx'].iloc[ii] = -0.8
        if pair_table['c1_residue_name'].iloc[ii] in ['TRP']:
            data_Feature['c1_hp_idx'].iloc[ii] = -0.9
        if pair_table['c1_residue_name'].iloc[ii] in ['TYR']:
            data_Feature['c1_hp_idx'].iloc[ii] = -1.3
        if pair_table['c1_residue_name'].iloc[ii] in ['PRO']:
            data_Feature['c1_hp_idx'].iloc[ii] = -1.6
        if pair_table['c1_residue_name'].iloc[ii] in ['HIS']:
            data_Feature['c1_hp_idx'].iloc[ii] = -3.2
        if pair_table['c1_residue_name'].iloc[ii] in ['GLU']:
            data_Feature['c1_hp_idx'].iloc[ii] = -3.5
        if pair_table['c1_residue_name'].iloc[ii] in ['GLN']:
            data_Feature['c1_hp_idx'].iloc[ii] = -3.5
        if pair_table['c1_residue_name'].iloc[ii] in ['ASP']:
            data_Feature['c1_hp_idx'].iloc[ii] = -3.5
        if pair_table['c1_residue_name'].iloc[ii] in ['ASN']:
            data_Feature['c1_hp_idx'].iloc[ii] = -3.5
        if pair_table['c1_residue_name'].iloc[ii] in ['LYS']:
            data_Feature['c1_hp_idx'].iloc[ii] = -3.9
        if pair_table['c1_residue_name'].iloc[ii] in ['ARG']:
            data_Feature['c1_hp_idx'].iloc[ii] = -4.5
    
    #c2    
        if pair_table['c2_residue_name'].iloc[ii] in ['MOL']:
            data_Feature['c2_hp_idx'].iloc[ii] = 4.5
        if pair_table['c2_residue_name'].iloc[ii] in ['ILE']:
            data_Feature['c2_hp_idx'].iloc[ii] = 4.5
        if pair_table['c2_residue_name'].iloc[ii] in ['VAL']:
            data_Feature['c2_hp_idx'].iloc[ii] = 4.2
        if pair_table['c2_residue_name'].iloc[ii] in ['LEU']:
            data_Feature['c2_hp_idx'].iloc[ii] = 3.8
        if pair_table['c2_residue_name'].iloc[ii] in ['PHE']:
            data_Feature['c2_hp_idx'].iloc[ii] = 2.8
        if pair_table['c2_residue_name'].iloc[ii] in ['CYS']:
            data_Feature['c2_hp_idx'].iloc[ii] = 2.5
        if pair_table['c2_residue_name'].iloc[ii] in ['MET']:
            data_Feature['c2_hp_idx'].iloc[ii] = 1.9
        if pair_table['c2_residue_name'].iloc[ii] in ['ALA']:
            data_Feature['c2_hp_idx'].iloc[ii] = 1.8
        if pair_table['c2_residue_name'].iloc[ii] in ['GLY']:
            data_Feature['c2_hp_idx'].iloc[ii] = -0.4
        if pair_table['c2_residue_name'].iloc[ii] in ['THR']:
            data_Feature['c2_hp_idx'].iloc[ii] = -0.7
        if pair_table['c2_residue_name'].iloc[ii] in ['SER']:
            data_Feature['c2_hp_idx'].iloc[ii] = -0.8
        if pair_table['c2_residue_name'].iloc[ii] in ['TRP']:
            data_Feature['c2_hp_idx'].iloc[ii] = -0.9
        if pair_table['c2_residue_name'].iloc[ii] in ['TYR']:
            data_Feature['c2_hp_idx'].iloc[ii] = -1.3
        if pair_table['c2_residue_name'].iloc[ii] in ['PRO']:
            data_Feature['c2_hp_idx'].iloc[ii] = -1.6
        if pair_table['c2_residue_name'].iloc[ii] in ['HIS']:
            data_Feature['c2_hp_idx'].iloc[ii] = -3.2
        if pair_table['c2_residue_name'].iloc[ii] in ['GLU']:
            data_Feature['c2_hp_idx'].iloc[ii] = -3.5
        if pair_table['c2_residue_name'].iloc[ii] in ['GLN']:
            data_Feature['c2_hp_idx'].iloc[ii] = -3.5
        if pair_table['c2_residue_name'].iloc[ii] in ['ASP']:
            data_Feature['c2_hp_idx'].iloc[ii] = -3.5
        if pair_table['c2_residue_name'].iloc[ii] in ['ASN']:
            data_Feature['c2_hp_idx'].iloc[ii] = -3.5
        if pair_table['c2_residue_name'].iloc[ii] in ['LYS']:
            data_Feature['c2_hp_idx'].iloc[ii] = -3.9
        if pair_table['c2_residue_name'].iloc[ii] in ['ARG']:
            data_Feature['c2_hp_idx'].iloc[ii] = -4.5
        
    return data_Feature


def distance_class_fill(pair_table, data_Feature):
    for ii in range(len(pair_table)):  
        if pair_table['distance'].iloc[ii] <= 7.0:
            data_Feature['distance'].iloc[ii] = 0
        if (pair_table['distance'].iloc[ii] > 7.0) & (pair_table['distance'].iloc[ii] <= 10):
            data_Feature['distance'].iloc[ii] = 1
        if (pair_table['distance'].iloc[ii] > 10) & (pair_table['distance'].iloc[ii] <= 20):
            data_Feature['distance'].iloc[ii] = 2
        if (pair_table['distance'].iloc[ii] > 20) & (pair_table['distance'].iloc[ii] <= 50):
            data_Feature['distance'].iloc[ii] = 3
        if pair_table['distance'].iloc[ii] > 50:
            data_Feature['distance'].iloc[ii] = 4
    
    return data_Feature


def atom_count(pair_table, data_Feature):
    aa_atom_count = os.path.join(new_data_dir, 'amino_acid_atom_count.csv')

    atom_count = pd.read_csv(aa_atom_count)
    
    for ii in range(len(pair_table)):
        for jj in range(len(atom_count)):
            if pair_table['c1_residue_name'].iloc[ii] == atom_count.iloc[jj,0]:
                data_Feature['c1_N_count'].iloc[ii] = atom_count['N'][jj]
                data_Feature['c1_C_count'].iloc[ii] = atom_count['C'][jj]
                data_Feature['c1_O_count'].iloc[ii] = atom_count['O'][jj]
                data_Feature['c1_H_count'].iloc[ii] = atom_count['H'][jj]
                data_Feature['c1_S_count'].iloc[ii] = atom_count['S'][jj]
                
    for ii in range(len(pair_table)):
        for jj in range(len(atom_count)):
            if pair_table['c2_residue_name'].iloc[ii] == atom_count.iloc[jj,0]:
                data_Feature['c2_N_count'].iloc[ii] = atom_count['N'][jj]
                data_Feature['c2_C_count'].iloc[ii] = atom_count['C'][jj]
                data_Feature['c2_O_count'].iloc[ii] = atom_count['O'][jj]
                data_Feature['c2_H_count'].iloc[ii] = atom_count['H'][jj]
                data_Feature['c2_S_count'].iloc[ii] = atom_count['S'][jj]
                
    return data_Feature


def atom_charge(pair_table, data_Feature):
    aa_atom_charge = os.path.join(new_data_dir, 'amino_acid_charge.csv')

    atom_charge = pd.read_csv(aa_atom_charge)
    
    for ii in range(len(pair_table)):
        for jj in range(len(atom_charge)):
            if pair_table['c1_residue_name'].iloc[ii] == atom_charge.iloc[jj,0]:
                data_Feature['c1_N_charge'].iloc[ii] = atom_charge['N'][jj]
                data_Feature['c1_C_charge'].iloc[ii] = atom_charge['C'][jj]
                data_Feature['c1_O_charge'].iloc[ii] = atom_charge['O'][jj]
                data_Feature['c1_H_charge'].iloc[ii] = atom_charge['H'][jj]
                data_Feature['c1_S_charge'].iloc[ii] = atom_charge['S'][jj]
                
    for ii in range(len(pair_table)):
        for jj in range(len(atom_charge)):
            if pair_table['c2_residue_name'].iloc[ii] == atom_charge.iloc[jj,0]:
                data_Feature['c2_N_charge'].iloc[ii] = atom_charge['N'][jj]
                data_Feature['c2_C_charge'].iloc[ii] = atom_charge['C'][jj]
                data_Feature['c2_O_charge'].iloc[ii] = atom_charge['O'][jj]
                data_Feature['c2_H_charge'].iloc[ii] = atom_charge['H'][jj]
                data_Feature['c2_S_charge'].iloc[ii] = atom_charge['S'][jj]
                
    return data_Feature


def graph_curvature(c1, c2, pair_table, data_Feature):

    cutoff = 7
    cutoff_np = 4
    node1 = utility.nodes(c1)
    edge1 = utility.edges(c1, node1, cutoff)
    node2 = utility.nodes_np(c2)
    edge2 = utility.edges_np(c2, node2, cutoff_np)
    alpha = 0.5

    ollivier1 = np.array(GT_curvature.ollivier_ricci(node1, edge1, alpha))
    ollivier2 = np.array(GT_curvature.ollivier_ricci(node2, edge2, alpha))
    forman1 = np.array(GT_curvature.forman_ricci(node1, edge1))
    forman2 = np.array(GT_curvature.forman_ricci(node2, edge2))

    ollivier1_pd = pd.DataFrame({"res_number":c1[c1['atom_name']=='CA']['residue_number'], "ollivier1":ollivier1})
    ollivier2_pd = pd.DataFrame({"res_number":c2['residue_number'], "ollivier2":ollivier2})
    
    forman1_pd = pd.DataFrame({"res_number":c1[c1['atom_name']=='CA']['residue_number'], "forman1":forman1})
    forman2_pd = pd.DataFrame({"res_number":c2['residue_number'], "forman2":forman2})
    
    for ii in range(len(pair_table)):
        for jj in range(len(ollivier1_pd)):
            if pair_table['c1_residue_number'].iloc[ii] == ollivier1_pd['res_number'].iloc[jj]:
                data_Feature['c1_ollivier'].iloc[ii] = ollivier1_pd['ollivier1'].iloc[jj]
                data_Feature['c1_forman'].iloc[ii] = forman1_pd['forman1'].iloc[jj]
    
    for ii in range(len(pair_table)):
        for jj in range(len(ollivier2_pd)):
            if pair_table['c2_residue_number'].iloc[ii] == ollivier2_pd['res_number'].iloc[jj]:
                data_Feature['c2_ollivier'].iloc[ii] = ollivier2_pd['ollivier2'].iloc[jj]
                data_Feature['c2_forman'].iloc[ii] = forman2_pd['forman2'].iloc[jj]
    
    return data_Feature

def graph_gnm(c1, c2, c1_CA, pdb_name, chain1, np_name, pair_table, data_Feature):

    pdb_dir = os.path.join(protein_dir, pdb_name+'.pdb')
    np_dir = os.path.join(nano_dir, np_name+'.pdb')

    gnm1 = np.array(GNM.gnm_sum_mode(pdb_dir, 10, chain1))
    gnm2 = np.array(GNM.gnm_sum_mode_np(np_dir, 10))

    print('gnm1', len(gnm1))
    print('gnm2', len(gnm2))

    if len(gnm1)==len(c1_CA):
        gnm1_pd = pd.DataFrame({"res_number":c1[c1['atom_name']=='CA']['residue_number'], "gnm1":gnm1})
        print('1_same')

    if len(gnm1)>len(c1_CA):
        gnm1 = gnm1[:len(c1_CA)]
        gnm1_pd = pd.DataFrame({"res_number":c1[c1['atom_name']=='CA']['residue_number'], "gnm1":gnm1})
        print('1_large')

    if len(gnm1)<len(c1_CA):
        gnm1_pd = pd.DataFrame({"res_number":c1[c1['atom_name']=='CA']['residue_number'], "gnm1":np.nan})
        print('1_small')
        for ii in range(len(gnm1)):
            gnm1_pd['gnm1'].iloc[ii] = gnm1[ii]
        gnm1_pd = gnm1_pd.fillna(method='ffill')


    if len(gnm2)==len(c2):
        gnm2_pd = pd.DataFrame({"res_number":c2['residue_number'], "gnm2":gnm2}) 
        print('2_same')
    
    if len(gnm2)>len(c2): 
        gnm2 = gnm2[:len(c2)]
        gnm2_pd = pd.DataFrame({"res_number":c2['residue_number'], "gnm2":gnm2}) 
        print('2_large')
    
    if len(gnm2)<len(c2):
        gnm2_pd = pd.DataFrame({"res_number":c2['residue_number'], "gnm2":np.nan})
        print('2_small')
        for ii in range(len(gnm2)):
            gnm2_pd['gnm2'].iloc[ii] = gnm2[ii]
        gnm2_pd = gnm2_pd.fillna(method='ffill')


    for ii in range(len(pair_table)):
        for jj in range(len(gnm1_pd)):
            if pair_table['c1_residue_number'].iloc[ii] == gnm1_pd['res_number'].iloc[jj]:
                data_Feature['c1_gnm'].iloc[ii] = gnm1_pd['gnm1'].iloc[jj]
                
    for ii in range(len(pair_table)):
        for jj in range(len(gnm2_pd)):
            if pair_table['c2_residue_number'].iloc[ii] == gnm2_pd['res_number'].iloc[jj]:
                data_Feature['c2_gnm'].iloc[ii] = gnm2_pd['gnm2'].iloc[jj]
    
    return data_Feature

def graph_fd(c1, c2, pair_table, data_Feature):

    cutoff = 7
    cutoff_np = 4
    node1 = utility.nodes(c1)
    edge1 = utility.edges(c1, node1, cutoff)
    node2 = utility.nodes_np(c2)
    edge2 = utility.edges_np(c2, node2, cutoff_np)

    fd1 = MFD.fractal_dimension(node1,edge1)
    fd2 = MFD.fractal_dimension(node2,edge2)
    print('fractal dimension ok')

    r_d = 5
    more_fd1 = MFD.more_box(node1, edge1, r_d)
    more_fd2 = MFD.more_box(node2, edge2, r_d)


    for ii in range(len(pair_table)):
        for jj in range(len(fd1)):
            if pair_table['c1_residue_number'].iloc[ii] == fd1[jj,0]:
                data_Feature['fd1'].iloc[ii] = fd1[jj,1]
                data_Feature['more_fd1_1'].iloc[ii] = more_fd1[jj,1]
                data_Feature['more_fd1_2'].iloc[ii] = more_fd1[jj,2]
                data_Feature['more_fd1_3'].iloc[ii] = more_fd1[jj,3]
                data_Feature['more_fd1_4'].iloc[ii] = more_fd1[jj,4]

    for ii in range(len(pair_table)):
        for jj in range(len(fd2)):
            if pair_table['c2_residue_number'].iloc[ii] == fd2[jj,0]:
                data_Feature['fd2'].iloc[ii] = fd2[jj,1]
                data_Feature['more_fd2_1'].iloc[ii] = more_fd2[jj,1]
                data_Feature['more_fd2_2'].iloc[ii] = more_fd2[jj,2]
                data_Feature['more_fd2_3'].iloc[ii] = more_fd2[jj,3]
                data_Feature['more_fd2_4'].iloc[ii] = more_fd2[jj,4]

    return data_Feature

def graph_os(c1, c2, pdb_name, np_name, chain1, pair_table, data_Feature):

    cutoff = 7
    node1 = utility.nodes(c1)
    node2 = utility.nodes_np(c2)

    G_os1_arr, G_os2_arr =local_OPD.GOS_np(pdb_name, chain1, np_name)
    for ii in range(len(pair_table)):
        for jj in range(len(G_os1_arr[:,0])):
                if pair_table['c1_residue_number'].iloc[ii] == node1[jj]:
                    data_Feature['G_os1_5'].iloc[ii] = G_os1_arr[jj,0]
                    data_Feature['G_os1_7'].iloc[ii] = G_os1_arr[jj,1]
                    data_Feature['G_os1_10'].iloc[ii] = G_os1_arr[jj,2]
                    data_Feature['G_os1_15'].iloc[ii] = G_os1_arr[jj,3]


    for ii in range(len(pair_table)):
        for jj in range(len(G_os2_arr[:,0])):
                if pair_table['c2_residue_number'].iloc[ii] == node2[jj]:
                    data_Feature['G_os2_5'].iloc[ii] = G_os2_arr[jj,0]
                    data_Feature['G_os2_7'].iloc[ii] = G_os2_arr[jj,1]
                    data_Feature['G_os2_10'].iloc[ii] = G_os2_arr[jj,2]
                    data_Feature['G_os2_15'].iloc[ii] = G_os2_arr[jj,3]

    return data_Feature

# geometry data needs to be calculated separately
def geometry(pdb_name, chain1, np_name, data_Feature, pair_table):
    geo_txt1 = os.path.join(geometry_dir, pdb_name+'_'+chain1+'.txt')
    geo_txt2 = os.path.join(geometry_dir, np_name+'.txt')
    res_num1, shellAcc1, Rinacc1, Pocketness1 = np.loadtxt(geo_txt1, skiprows=43, usecols=(0,3,4,7), unpack=True)
    res_num2, shellAcc2, Rinacc2, Pocketness2 = np.loadtxt(geo_txt2, skiprows=43, usecols=(0,3,4,7), unpack=True)
    
    for ii in range(len(pair_table)):
        for jj in range(len(res_num1)):
            if pair_table['c1_residue_number'].iloc[ii] == res_num1[jj]:
                data_Feature['c1_rd'].iloc[ii] = Rinacc1[jj]
                data_Feature['c1_shell'].iloc[ii] = shellAcc1[jj]
                data_Feature['c1_poc'].iloc[ii] = Pocketness1[jj]
    
    for ii in range(len(pair_table)):
        for jj in range(len(res_num2)):
            if pair_table['c2_residue_number'].iloc[ii] == res_num2[jj]:
                data_Feature['c2_rd'].iloc[ii] = Rinacc2[jj]
                data_Feature['c2_shell'].iloc[ii] = shellAcc2[jj]
                data_Feature['c2_poc'].iloc[ii] = Pocketness2[jj]
    
    return data_Feature



def feature_matrix(pdb_name, chain1, np_name):
    c1, c2, c1_CA, c2_all = CA_coord(pdb_name, chain1, np_name)
    data, pair_table = distance_data_range(c1, c2, c1_CA, c2_all)
    data_Feature = pd.DataFrame(np.zeros((((len(pair_table)),65))))
    data_Feature.columns=['c1', 'c1_pos', 'c1_neg', 'c1_polar', 'c1_amp', 'c1_hp', 'c1_hp_idx',
                          'c1_rd', 'c1_shell', 'c1_poc',
                          'c1_N_count', 'c1_C_count', 'c1_O_count','c1_H_count', 'c1_S_count',
                          'c1_N_charge', 'c1_C_charge', 'c1_O_charge','c1_H_charge', 'c1_S_charge',
                          'c1_ollivier', 'c1_forman', 'c1_gnm',
                          'fd1', 'more_fd1_1', 'more_fd1_2', 'more_fd1_3', 'more_fd1_4',
                          'G_os1_5', 'G_os1_7', 'G_os1_10', 'G_os1_15',
                          'c2', 'c2_pos', 'c2_neg','c2_polar', 'c2_amp', 'c2_hp', 'c2_hp_idx',
                          'c2_rd', 'c2_shell','c2_poc',
                          'c2_N_count', 'c2_C_count', 'c2_O_count', 'c2_H_count','c2_S_count',
                          'c2_N_charge', 'c2_C_charge', 'c2_O_charge','c2_H_charge', 'c2_S_charge',
                          'c2_ollivier', 'c2_forman', 'c2_gnm',
                          'fd2', 'more_fd2_1', 'more_fd2_2', 'more_fd2_3', 'more_fd2_4',
                          'G_os2_5', 'G_os2_7', 'G_os2_10', 'G_os2_15',
                          'distance']
    data_Feature['c1'] = pair_table['c1_residue_name']
    data_Feature['c2'] = pair_table['c2_residue_name']

    data_Feature = binary_fill(data_Feature, pair_table)
    print('binary_fill ok')
    data_Feature = hydrophobicity_fill(data_Feature, pair_table)
    print('hydrophobicity_fill ok')
    data_Feature = distance_class_fill(pair_table, data_Feature)
    print('distace_class_fill ok')
    data_Feature = geometry(pdb_name, chain1, np_name, data_Feature, pair_table)
    print('geometry_fill ok')
    data_Feature = atom_count(pair_table, data_Feature)
    print('atom_count_fill ok')
    data_Feature = atom_charge(pair_table, data_Feature)
    print('atom_charge_fill ok')
    data_Feature = graph_curvature(c1, c2, pair_table, data_Feature)
    print('graph_curvature_fill ok')
    data_Feature = graph_gnm(c1, c2, c1_CA, pdb_name, chain1, np_name, pair_table, data_Feature)
    print('graph_gnm_fill ok')
    data_Feature = graph_fd(c1, c2, pair_table, data_Feature)
    print('graph_fd_fill ok')
    data_Feature = graph_os(c1, c2, pdb_name, np_name, chain1, pair_table, data_Feature)
    print('os fill ok')

    pair_table.to_csv(os.path.join(pair_table_dir, 'pair_table'+'_'+pdb_name+'_'+chain1+np_name+'.csv'), index=False)

    aa_mol = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL', 'MOL']
    le = preprocessing.LabelEncoder()
    le.fit(aa_mol)
    data_Feature['c1'] = le.transform(data_Feature['c1'])#.astype(int)
    data_Feature['c2'] = le.transform(data_Feature['c2'])
    np.set_printoptions(suppress=True)
    feature_data_ = data_Feature.to_numpy().astype(np.float)
    np.save(os.path.join(data_feature_dir, 'data_feature'+'_'+pdb_name+'_'+chain1+np_name+'.npy'), feature_data_)

    #data_Feature.to_csv(os.path.join(data_feature_dir, 'data_feature'+'_'+pdb_name+'_'+chain1+np_name+'.csv'), index=False)
    
    return print(pdb_name, chain1, np_name, 'FINISHED')

code_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
data_dir = '../../data'
protein_dir = os.path.join(data_dir, 'pdb_protein')
nano_dir = os.path.join(data_dir, 'pdb_np')
geometry_dir = os.path.join(data_dir, 'geometry')
pdb_list_dir = os.path.join(data_dir, 'PPI_list')
new_data_dir = os.path.join(data_dir, 'charge_count')
pair_table_dir = os.path.join(data_dir, 'pair_table')
data_feature_dir = os.path.join(data_dir, 'descriptors_matrix')

if __name__ == "__main__":
    pdb_name = sys.argv[1]
    chain1 = sys.argv[2]
    np_name = sys.argv[3]
    feature_matrix(pdb_name, chain1, np_name)













