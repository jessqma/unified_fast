# %%
import numpy as np
import os
from biopandas.pdb import PandasPdb
import pandas as pd
from scipy.spatial.distance import cdist
import itertools
from descriptor_calc import GNM, MFD, local_OPD, Protrusion_Index, GT_curvature
import sys
import re
from pymol import cmd, stored
from sklearn.neighbors import KNeighborsClassifier
import igraph as ig

def make_graph(c1, cutoff):
    c1 = c1[c1['atom_name'] == 'CA']
    c1_CA = pd.concat([c1[c1['atom_name'] == 'CA']['x_coord'],
                       c1[c1['atom_name'] == 'CA']['y_coord'],
                       c1[c1['atom_name'] == 'CA']['z_coord']], axis=1).to_numpy()
    num_nodes = len(c1_CA)
    dists = cdist(c1_CA, c1_CA)

    connects = np.array(np.where((dists <= cutoff) == (dists > 0))).T
    edges = np.unique(np.sort(connects, axis=1), axis=0)
    g_ig = ig.Graph(n=num_nodes, edges=edges)
    G = g_ig.to_networkx()

    return G

def CA_coord(pdb_name, chain1, chain2):

    coord1 = PandasPdb()
    coord1.fetch_pdb(pdb_name)
    # coord1.read_pdb(os.path.join(protein_dir, pdb_name + '.pdb'))
    prot1_df = coord1.df['ATOM']
    prot1_df = prot1_df[(prot1_df['alt_loc'] == "") | (prot1_df['alt_loc'] == "A")]
    
    c1_ = [[] for _ in range(len(chain1))]
    for ii in range(len(chain1)):
        c1_[ii] = prot1_df[prot1_df['chain_id'] == chain1[ii]]
    c1 = pd.concat(c1_).reset_index(drop=True)
    
    c1_all_res = c1[['chain_id', 'residue_number', 'insertion']].drop_duplicates().reset_index(drop=True)
    c1_ca_res = c1[c1['atom_name'] == 'CA'][['chain_id', 'residue_number', 'insertion']]
    c1_no_cas = pd.merge(c1_all_res, c1_ca_res, how='left', indicator=True)['_merge']=='left_only'
    
    if sum(c1_no_cas) != 0:
        c1_incomplete_res = np.squeeze(np.where(c1[['chain_id', 'residue_number', 'insertion']].astype(str).agg('_'.join, axis=1).to_numpy() ==
               				        c1_all_res[c1_no_cas].astype(str).agg('_'.join, axis=1).to_numpy()))
        c1 = c1.drop(np.atleast_1d((c1_incomplete_res))).reset_index(drop=True)
    else:
        c1_incomplete_res = np.array([])
    
    c2_ = [[] for _ in range(len(chain2))]
    for ii in range(len(chain2)):
        c2_[ii] = prot1_df[prot1_df['chain_id'] == chain2[ii]]
    c2 = pd.concat(c2_).reset_index(drop=True)

    c2_all_res = c2[['chain_id', 'residue_number', 'insertion']].drop_duplicates().reset_index(drop=True)
    c2_ca_res = c2[c2['atom_name'] == 'CA'][['chain_id', 'residue_number', 'insertion']]
    c2_no_cas = pd.merge(c2_all_res, c2_ca_res, how='left', indicator=True)['_merge'] == 'left_only'

    if sum(c2_no_cas) != 0:
        c2_incomplete_res = np.squeeze(np.where(c2[['chain_id', 'residue_number', 'insertion']].astype(str).agg('_'.join, axis=1).to_numpy() ==
                     c2_all_res[c2_no_cas].astype(str).agg('_'.join, axis=1).to_numpy()))
        c2 = c2.drop(np.atleast_1d((c2_incomplete_res))).reset_index(drop=True)
    else:
        c2_incomplete_res = np.array([])
    
    c1_CA = pd.concat([c1[c1['atom_name'] == 'CA']['x_coord'],
                       c1[c1['atom_name'] == 'CA']['y_coord'],
                       c1[c1['atom_name'] == 'CA']['z_coord']],
                      axis=1).to_numpy()
    c2_CA = pd.concat([c2[c2['atom_name'] == 'CA']['x_coord'],
                       c2[c2['atom_name'] == 'CA']['y_coord'],
                       c2[c2['atom_name'] == 'CA']['z_coord']],
                      axis=1).to_numpy()

    return c1, c2, c1_CA, c2_CA, c1_incomplete_res, c2_incomplete_res

def distance_data_range(c1, c2, c1_CA, c2_CA):
    c1 = c1[c1['element_symbol'] != 'H']
    c2 = c2[c2['element_symbol'] != 'H']
    c1_all = pd.concat([c1['x_coord'], c1['y_coord'], c1['z_coord']], axis=1).to_numpy()
    c2_all = pd.concat([c2['x_coord'], c2['y_coord'], c2['z_coord']], axis=1).to_numpy()

    dists_all = cdist(c1_all, c2_all)
    int_index = np.full([len(c1_CA), len(c2_CA)], 2)

    intsA = c1.iloc[np.where(dists_all < 7)[0]][['chain_id', 'residue_number']].astype(str).agg('_'.join, axis=1)
    indexA = dict(
        (j, i) for i, j in enumerate(c1[['chain_id', 'residue_number']].astype(str).agg('_'.join, axis=1).unique()))
    indexA_ = [indexA[key] for key in intsA]

    intsB = c2.iloc[np.where(dists_all < 7)[1]][['chain_id', 'residue_number']].astype(str).agg('_'.join, axis=1)
    indexB = dict(
        (j, i) for i, j in enumerate(c2[['chain_id', 'residue_number']].astype(str).agg('_'.join, axis=1).unique()))
    indexB_ = [indexB[key] for key in intsB]

    for i in range(len(indexA_)):
        int_index[indexA_[i]][indexB_[i]] = 1

    intsA = c1.iloc[np.where(dists_all < 4)[0]][['chain_id', 'residue_number']].astype(str).agg('_'.join, axis=1)
    indexA = dict(
        (j, i) for i, j in enumerate(c1[['chain_id', 'residue_number']].astype(str).agg('_'.join, axis=1).unique()))
    indexA_ = [indexA[key] for key in intsA]

    intsB = c2.iloc[np.where(dists_all < 4)[1]][['chain_id', 'residue_number']].astype(str).agg('_'.join, axis=1)
    indexB = dict(
        (j, i) for i, j in enumerate(c2[['chain_id', 'residue_number']].astype(str).agg('_'.join, axis=1).unique()))
    indexB_ = [indexB[key] for key in intsB]

    for i in range(len(indexA_)):
        int_index[indexA_[i]][indexB_[i]] = 0
    ints = int_index.flatten()

    c1_CA_res = c1[c1['atom_name'] == 'CA']['residue_number']
    c2_CA_res = c2[c2['atom_name'] == 'CA']['residue_number']
    CA_all_res = np.array(list(itertools.product(c1_CA_res, c2_CA_res)))
    # print('Len_CA_all', len(CA_all))
    print('Len_CA_all_res', len(CA_all_res))

    c1_CA_names = c1[c1['atom_name'] == 'CA']['residue_name']
    c2_CA_names = c2[c2['atom_name'] == 'CA']['residue_name']
    CA_res_names = np.array(list(itertools.product(c1_CA_names, c2_CA_names)))
    # pair_dists = updated_dist[np.arange(len(CA_all_res)) // len(c2_CA), np.arange(len(CA_all_res)) % len(c2_CA)]

    c1_chain = c1[c1['atom_name'] == 'CA']['chain_id']
    c2_chain = c2[c2['atom_name'] == 'CA']['chain_id']
    CA_chain = np.array(list(itertools.product(c1_chain, c2_chain)))

    c1_ins = c1[c1['atom_name'] == 'CA']['insertion']
    c2_ins = c2[c2['atom_name'] == 'CA']['insertion']
    CA_ins = np.array(list(itertools.product(c1_ins, c2_ins)))
    
    res_info_dict = {'c1_chain_id': CA_chain[:, 0],
                     'c1_residue_number': CA_all_res[:, 0],
                     'c1_insertion': CA_ins[:, 0],
                     'c1_residue_name': CA_res_names[:, 0],
                     'c2_chain_id': CA_chain[:, 1],
                     'c2_residue_number': CA_all_res[:, 1],
                     'c2_insertion': CA_ins[:, 1],
                     'c2_residue_name': CA_res_names[:, 1],
                     'distance': ints}

    pair_table = pd.DataFrame(res_info_dict)

    return pair_table


# geometry data needs to be calculated separately
def geometry(pdb_name, chain1, chain2, data_feature1, data_feature2, c1_CA, c2_CA):
    geo_txt1 = os.path.join(geometry_dir, pdb_name + '_' + chain1 + '.txt')
    geo_txt2 = os.path.join(geometry_dir, pdb_name + '_' + chain2 + '.txt')
    shellAcc1, Rinacc1, Pocketness1 = np.loadtxt(geo_txt1, skiprows=43, usecols=(3, 4, 7), unpack=True)
    shellAcc2, Rinacc2, Pocketness2 = np.loadtxt(geo_txt2, skiprows=43, usecols=(3, 4, 7), unpack=True)

    data_feature1['rd'] = Rinacc1[:len(c1_CA)]
    data_feature1['shell'] = shellAcc1[:len(c1_CA)]
    data_feature1['poc'] = Pocketness1[:len(c1_CA)]
    data_feature2['rd'] = Rinacc2[:len(c2_CA)]
    data_feature2['shell'] = shellAcc2[:len(c2_CA)]
    data_feature2['poc'] = Pocketness2[:len(c2_CA)]

    return data_feature1, data_feature2

def graph_curvature(c1, c2, data_feature1, data_feature2):
    cutoff = 7
    G1 = make_graph(c1, cutoff)
    G2 = make_graph(c2, cutoff)
    alpha = 0.5

    ollivier1 = np.array(GT_curvature.ollivier_ricci(G1))
    ollivier2 = np.array(GT_curvature.ollivier_ricci(G2))
    forman1 = np.array(GT_curvature.forman_ricci(G1))
    forman2 = np.array(GT_curvature.forman_ricci(G2))

    data_feature1['ollivier'] = ollivier1
    data_feature2['ollivier'] = ollivier2
    data_feature1['forman'] = forman1
    data_feature2['forman'] = forman2

    return data_feature1, data_feature2

def graph_gnm(c1, c2, c1_CA, c2_CA, pdb_name, chain1, chain2, data_feature1, data_feature2):
    pdb_dir = os.path.join(protein_dir, pdb_name + '.pdb')
    gnm1 = np.array(GNM.gnm_sum_mode(pdb_dir, 10, chain1))
    gnm2 = np.array(GNM.gnm_sum_mode(pdb_dir, 10, chain2))

    print('gnm1', len(gnm1))
    print('gnm2', len(gnm2))

    data_feature1['gnm'] = gnm1
    print('1_same')

    data_feature2['gnm'] = gnm2
    print('2_same')

    return data_feature1, data_feature2

def graph_fd(c1, c2, data_feature1, data_feature2):
    cutoff = 7

    G1 = make_graph(c1, cutoff)
    G2 = make_graph(c2, cutoff)

    fd1 = MFD.fractal_dimension(G1)
    fd2 = MFD.fractal_dimension(G2)
    print('fractal dimension ok')

    r_d = 5
    more_fd1 = MFD.more_box(G1, r_d)
    more_fd2 = MFD.more_box(G2, r_d)

    data_feature1['fd'] = fd1[:, 1]
    data_feature1['more_fd_1'] = more_fd1[:, 1]
    data_feature1['more_fd_2'] = more_fd1[:, 2]
    data_feature1['more_fd_3'] = more_fd1[:, 3]
    data_feature1['more_fd_4'] = more_fd1[:, 4]

    data_feature2['fd'] = fd2[:, 1]
    data_feature2['more_fd_1'] = more_fd2[:, 1]
    data_feature2['more_fd_2'] = more_fd2[:, 2]
    data_feature2['more_fd_3'] = more_fd2[:, 3]
    data_feature2['more_fd_4'] = more_fd2[:, 4]

    return data_feature1, data_feature2

def protrusion(c1, c2, data_feature1, data_feature2):
    cutoff_cx = 10
    cx_1 = Protrusion_Index.cx(c1, cutoff_cx)
    cx_2 = Protrusion_Index.cx(c2, cutoff_cx)

    data_feature1['cx'] = cx_1
    data_feature2['cx'] = cx_2

    return data_feature1, data_feature2

def graph_os(c1, c2, data_feature1, data_feature2):
    c1 = c1[c1['atom_name'] == 'CA']
    c2 = c2[c2['atom_name'] == 'CA']

    G_os1_arr, G_os2_arr = local_OPD.GOS(c1, c2)

    data_feature1['G_os_5'] = G_os1_arr[:, 0]
    data_feature1['G_os_7'] = G_os1_arr[:, 1]

    data_feature2['G_os_5'] = G_os2_arr[:, 0]
    data_feature2['G_os_7'] = G_os2_arr[:, 1]

    return data_feature1, data_feature2

def get_pointcloud(pdb_name, chain):
    # cmd.load(os.path.join(pdb_dir, pdb_name + '.pdb'))
    cmd.load(os.path.join('https://files.rcsb.org/download/', pdb_name.upper() + '.pdb'))
    cmd.select(('chain %s' + ' or chain %s' * (len(chain) - 1)) % tuple([_ for _ in chain]))
    cmd.create('prot', ('(chain %s' + ' or chain %s' * (len(chain) - 1)) % tuple(
        [_ for _ in chain]) + ') and not HETATM and alt ""+A')
    cmd.delete(pdb_name)
    cmd.set('surface_quality', '0')
    cmd.show_as('surface', 'all')
    cmd.set_view('1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,300,1')
    res_xyz = cmd.get_coordset('prot', 1)
    cmd.save(os.path.join(wrl_dir, pdb_name + '_' + chain + '.wrl'))
    cmd.delete('all')

    coords = []
    norms = []
    cf = 0
    nf = 0

    with open(os.path.join(wrl_dir, pdb_name + '_' + chain + '.wrl'), "r") as vrml:
        for lines in vrml:
            if 'point [' in lines:
                cf = 1
            if cf == 1:
                if ']' not in lines:
                    a = re.findall(r"[-+]?\d*\.\d+|\d+", lines)
                    if len(a) == 3:
                        coords.append(tuple(map(float, a)))
                else:
                    cf = 0
            if 'vector [' in lines:
                nf = 1
                cf = 0
            if nf == 1:
                if ']' not in lines:
                    a = re.findall(r"[-+]?\d*\.\d+|\d+", lines)
                    if len(a) == 3:
                        norms.append(tuple(map(float, a)))
                else:
                    nf = 0

    coords, li = np.unique(coords, axis=0, return_index=True)
    norms = np.array(norms)[li, :]

    return np.array(res_xyz), np.array(coords), np.array(norms)

def pointcloud2res(res_nums, res_xyz, coords):
    classes = np.arange(len(res_xyz))
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(res_xyz, y=classes)
    atom_nums = knn.predict(coords)
    res_keys = res_nums.iloc[atom_nums]

    return res_keys

def get_intpoints(pair_table, res_keys1, res_keys2, pdb_name, chain1, chain2):
    c1_res = pair_table[['c1_chain_id', 'c1_residue_number', 'c1_insertion']].astype(str).agg('_'.join, axis=1).reset_index(drop=True)
    c2_res = pair_table[['c2_chain_id', 'c2_residue_number', 'c2_insertion']].astype(str).agg('_'.join, axis=1).reset_index(drop=True)
    distances = pair_table['distance']

    c1_ints = c1_res[distances == 0].unique()
    c2_ints = c2_res[distances == 0].unique()  # distance 4

    # ints = res_pairs[(res_pairs['distance'] == 0) | (res_pairs['distance'] == 1)] #distance 7
    # c1_ints = ints['c1_residue_number'].unique()
    # c2_ints = ints['c2_residue_number'].unique()

    c1_cloud_ints = np.ones_like(res_keys1)
    c2_cloud_ints = np.ones_like(res_keys2)

    for idx, res in enumerate(res_keys1):
        if res in c1_ints:
            c1_cloud_ints[idx] = 0

    for idx, res in enumerate(res_keys2):
        if res in c2_ints:
            c2_cloud_ints[idx] = 0

    return c1_cloud_ints, c2_cloud_ints

def feature_matrix(pdb_name, chain1, chain2):
    print(pdb_name, chain1, chain2)
    c1, c2, c1_CA, c2_CA, c1_incomplete_res, c2_incomplete_res = CA_coord(pdb_name, chain1, chain2)

    # res_nums1 = c1[c1['atom_name']=='CA'][['chain_id', 'residue_number', 'insertion']].astype(str).agg('_'.join, axis=1).reset_index(drop=True)
    # res_nums2 = c2[c2['atom_name']=='CA'][['chain_id', 'residue_number', 'insertion']].astype(str).agg('_'.join, axis=1).reset_index(drop=True)

    res_nums1 = c1[['chain_id', 'residue_number', 'insertion']].astype(str).agg('_'.join, axis=1).reset_index(drop=True)
    res_nums2 = c2[['chain_id', 'residue_number', 'insertion']].astype(str).agg('_'.join, axis=1).reset_index(drop=True)

    data_feature1 = {}
    data_feature2 = {}

    feature_names = ['res_name', 'pos', 'neg', 'polar', 'amp', 'hp', 'hp_idx',
                     'rd', 'shell', 'poc',
                     'N_count', 'C_count', 'O_count', 'H_count', 'S_count',
                     'N_charge', 'C_charge', 'O_charge', 'H_charge', 'S_charge',
                     'ollivier', 'forman', 'gnm',
                     'fd', 'more_fd_1', 'more_fd_2', 'more_fd_3', 'more_fd_4',
                     'G_os_5', 'G_os_7', 'G_os_10', 'G_os_15']

    data_feature1, data_feature2 = geometry(pdb_name, chain1, chain2, data_feature1, data_feature2, c1_CA, c2_CA)
    print('geometry_fill ok')
    data_feature1, data_feature2 = graph_curvature(c1, c2, data_feature1, data_feature2)
    print('graph_curvature_fill ok')
    data_feature1, data_feature2 = graph_gnm(c1, c2, c1_CA, c2_CA, pdb_name, chain1, chain2, data_feature1,
                                             data_feature2)
    print('graph_gnm_fill ok')
    data_feature1, data_feature2 = graph_fd(c1, c2, data_feature1, data_feature2)
    print('graph_fd_fill ok')
    data_feature1, data_feature2 = protrusion(c1, c2, data_feature1, data_feature2)
    print('protrusion_fill ok')
    data_feature1, data_feature2 = graph_os(c1, c2, data_feature1, data_feature2)
    print('os fill ok')

    for item in data_feature1.keys():
        print(item, len(data_feature1[item]))
    for item in data_feature2.keys():
        print(item, len(data_feature2[item]))

    data_feature1_pd = pd.DataFrame(data_feature1)
    data_feature2_pd = pd.DataFrame(data_feature2)
    res_xyz1, coords1, norms1 = get_pointcloud(pdb_name, chain1)
    if np.size(c1_incomplete_res) != 0:
        res_xyz1 = np.delete(res_xyz1, c1_incomplete_res, axis=0)
    res_keys1 = pointcloud2res(res_nums1, res_xyz1, coords1)
    _1 = {v: k for k, v in enumerate(np.unique(res_nums1), 0)}
    keys1 = [_1[_] for _ in res_keys1]
    feats1 = data_feature1_pd.iloc[keys1]
    cloud1 = np.concatenate([coords1, norms1, feats1], axis=1)

    res_xyz2, coords2, norms2 = get_pointcloud(pdb_name, chain2)
    if np.size(c2_incomplete_res) != 0:
        res_xyz2 = np.delete(res_xyz2, c2_incomplete_res, axis=0)
    res_keys2 = pointcloud2res(res_nums2, res_xyz2, coords2)
    _2 = {v: k for k, v in enumerate(np.unique(res_nums2), 0)}
    keys2 = [_2[_] for _ in res_keys2]
    feats2 = data_feature2_pd.iloc[keys2]
    cloud2 = np.concatenate([coords2, norms2, feats2], axis=1)

    np.savetxt(os.path.join(pts_dir, pdb_name + '_' + chain1 + '.pts'), cloud1)
    np.savetxt(os.path.join(pts_dir, pdb_name + '_' + chain2 + '.pts'), cloud2)

    pair_table = distance_data_range(c1, c2, c1_CA, c2_CA)

    c1_cloud_ints, c2_cloud_ints = get_intpoints(pair_table, res_keys1, res_keys2, pdb_name, chain1, chain2)
    np.savetxt(os.path.join(labels_dir, pdb_name + chain1 + '_' + pdb_name + chain2 + '_0.seg'), c1_cloud_ints)
    np.savetxt(os.path.join(labels_dir, pdb_name + chain1 + '_' + pdb_name + chain2 + '_1.seg'), c2_cloud_ints)

    data_feature1_pd.to_csv(os.path.join(data_feature_dir, pdb_name + '_' + chain1 + '_features.csv'),
                            index=False)
    data_feature2_pd.to_csv(os.path.join(data_feature_dir, pdb_name + '_' + chain2 + '_features.csv'),
                            index=False)

    pair_table.to_csv(
        os.path.join(pair_table_dir, 'pair_table' + '_' + pdb_name + chain1 + '_' + pdb_name + chain2 + '.csv'),
        index=False)

    return print(pdb_name, chain1, chain2, 'FINISHED')

data_dir = os.path.join(os.getcwd(), 'Direct')
protein_dir = os.path.join(data_dir, 'pdb_protein')
geometry_dir = os.path.join(data_dir, 'geometry')
pair_table_dir = os.path.join(data_dir, 'pair_table')
data_feature_dir = os.path.join(data_dir, 'descriptors_chem')

wrl_dir = os.path.join(data_dir, 'wrls')
pts_dir = os.path.join(data_dir, 'pts_chem')
labels_dir = os.path.join(data_dir, 'pts_label_chem')

# data_dir = os.getcwd()
# protein_dir = os.path.join(data_dir, 'pdb_protein')
# geometry_dir = os.path.join(data_dir, 'geometry')
# pair_table_dir = os.path.join(data_dir, 'obj')
# data_feature_dir = os.path.join(data_dir, 'obj')
#
# wrl_dir = os.path.join(data_dir, 'obj')
# pts_dir = os.path.join(data_dir, 'obj')
# labels_dir = os.path.join(data_dir, 'obj')



if __name__ == "__main__":
    pdb_name = sys.argv[1]
    chain1 = sys.argv[2]
    chain2 = sys.argv[3]
    folder = sys.argv[4]
    
    data_dir = os.path.join(os.getcwd(), folder)
    protein_dir = os.path.join(data_dir, 'pdb_protein')
    geometry_dir = os.path.join(data_dir, 'geometry')
    pair_table_dir = os.path.join(data_dir, 'pair_table')
    data_feature_dir = os.path.join(data_dir, 'descriptors_matrix')

    wrl_dir = os.path.join(data_dir, 'wrls')
    pts_dir = os.path.join(data_dir, 'pts')
    labels_dir = os.path.join(data_dir, 'pts_label')
    
    feature_matrix(pdb_name, chain1, chain2)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
