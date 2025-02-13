#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os
from biopandas.pdb import PandasPdb
import pandas as pd
import Bio
from Bio.PDB import PDBParser
from scipy.spatial.distance import euclidean
import itertools
import utility
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import sys



#directory setting
data_dir = '../../data'

processed_data_dir = os.path.join(data_dir, 'processed_data_training')

test_data_dir = os.path.join(data_dir, 'processed_test_data')
ppi_test_dir = os.path.join(test_data_dir, 'PPI')
pni_test_dir = os.path.join(test_data_dir, 'PNI')
# this folder needs to be changed accordingly 
#pni_test_dir2 = os.path.join(pni_test_dir, 'human_islet_gqd')
model_dir = os.path.join(data_dir, 'dnn_models')

protein_dir = os.path.join(data_dir, 'pdb_protein')
nano_dir = os.path.join(data_dir, 'pdb_np')

result_dir = '../../results'
#predict_dir = os.path.join(result_dir, 'predicted_results')



# data loading

def data_loading(filename):
    
    feature_all = np.load(os.path.join(processed_data_dir, 'binary_under_all7.npy'))
    GE_GT = np.load(os.path.join(processed_data_dir, 'GE_GT_descriptor_7.npy'))
    GT = np.load(os.path.join(processed_data_dir, 'GT_descriptor_7.npy'))

    test_feature = np.load(os.path.join(pni_test_dir,'data_feature_%s.npy' % filename)) #need to change accordingly

    return feature_all, GE_GT, GT, test_feature



# train and test data set

def train_test(data_option, feature_all, GE_GT, GT, test_feature):
    options = ["all", "GE_GT", "GT"]
    # check that the two arguments have valid values:
    try:
        data_option = options.index(data_option);
    except ValueError:
        return "invalid choice: {}".format(data_option)
    
   

    test_data = pd.DataFrame(data=test_feature, columns=['c1', 'c1_pos', 'c1_neg', 'c1_polar', 'c1_amp', 'c1_hp', 'c1_hp_idx',
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
                                       'distance'])

  

    if data_option == 'all': # all data
        X_train = feature_all[:,:64]
        y_train = feature_all[:,64]

        X_test = test_feature[:,:64]
        y_test = test_feature[:,64]
        y_test_pd = pd.DataFrame(y_test)

        y_test_pd.iloc[:,0].replace([2, 3, 4], 1, inplace=True)
        y_test = np.asarray(y_test_pd).reshape(1,-1)[0]
    

    elif data_option == 'GE+GT': # GE+GT
        X_train = GE_GT[:,:30]
        y_train = GE_GT[:,30]

    
        data = test_data.fillna(method='ffill')
        data = data[['c1_rd', 'c1_shell', 'c1_poc', 'G_os1_5', 'G_os1_7', 'G_os1_10', 'G_os1_15',
             'c1_ollivier', 'c1_forman', 'c1_gnm',
             'fd1', 'more_fd1_1', 'more_fd1_2', 'more_fd1_3', 'more_fd1_4',  
             'c2_rd', 'c2_shell', 'c2_poc', 'G_os2_5', 'G_os2_7', 'G_os2_10', 'G_os2_15',
             'c2_ollivier', 'c2_forman', 'c2_gnm', 
             'fd2', 'more_fd2_1', 'more_fd2_2', 'more_fd2_3', 'more_fd2_4',
             'distance']]
    
        X_test = np.asarray(data)[:,:30]
        y_test = np.asarray(data)[:,30]
        y_test_pd = pd.DataFrame(y_test)

        y_test_pd.iloc[:,0].replace([2, 3, 4], 1, inplace=True)
        y_test = np.asarray(y_test_pd).reshape(1,-1)[0]
    
    
    else: # GT
       X_train = GT[:,:16]
       y_train = GT[:,16]

       data=test_data.fillna(method='ffill')
       data = data[['c1_ollivier', 'c1_forman', 'c1_gnm',
             'fd1', 'more_fd1_1', 'more_fd1_2', 'more_fd1_3', 'more_fd1_4',  
             'c2_ollivier', 'c2_forman', 'c2_gnm', 
             'fd2', 'more_fd2_1', 'more_fd2_2', 'more_fd2_3', 'more_fd2_4',
             'distance']]
    
       X_test = np.asarray(data)[:,:16]
       y_test = np.asarray(data)[:,16]
       y_test_pd = pd.DataFrame(y_test)
       y_test_pd.iloc[:,0].replace([2, 3, 4], 1, inplace=True)
       y_test = np.asarray(y_test_pd).reshape(1,-1)[0]

    return X_train, y_train, X_test, y_test
    


# test data pdb coordinates

def TrueAB(pdb_name):
    
    coord = PandasPdb()
    coord.read_pdb(os.path.join(protein_dir, '%s.pdb' % pdb_name))
    coord2 = PandasPdb()
    coord2.read_pdb(os.path.join(nano_dir, '%s.pdb' % np_name))

    chain_0 = coord.df['ATOM'][coord.df['ATOM']['chain_id']=='A']
    chain_1 = coord2.df['ATOM']
    c1 = chain_0
    c2 = chain_1

    c1_CA = pd.concat([c1[c1['atom_name']=='CA']['x_coord'], 
                      c1[c1['atom_name']=='CA']['y_coord'], 
                      c1[c1['atom_name']=='CA']['z_coord']], 
                      axis=1).to_numpy()


    c2_CA = pd.concat([c2['x_coord'], 
                      c2['y_coord'], 
                      c2['z_coord']], 
                      axis=1).to_numpy()

    updated_dist = utility.new_dist(c1_CA, c2_CA)
    idx0 = np.where(y_test==0)


    # true residue number
    interA = np.where(updated_dist<7)[0] 
    TrueA = np.unique(interA)
    interB = np.where(updated_dist<7)[1] 
    TrueB = np.unique(interB)

    return c1, c2, c1_CA, c2_CA, TrueA, TrueB


# ML TRAIN 

# RFC

from sklearn.ensemble import RandomForestClassifier

def RFC(X_train, y_train, X_test, y_test, threshold):
    rfc =RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=50)
    rfc_fit = rfc.fit(X_train, y_train)
    y_pred = rfc_fit.predict(X_test)
    y_proba_rfc = rfc_fit.predict_proba(X_test)
    # y_proba_rfc = rfc_fit.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
  
    N = np.int(y_proba_rfc.shape[0]*threshold)
    max_prob_rfc = np.argsort(y_proba_rfc[:,0])[-N:] 
    
    return cm, max_prob_rfc
    

# XGB

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance

def XGB(X_train, y_train, X_test, y_test, threshold):
    xgb = XGBClassifier(max_depth=13,n_estimator=120,subsample=0.8,colsample_bytree=1,gamma=0.05, random_state=0)
    xgb_fit = xgb.fit(X_train, y_train)
    y_pred = xgb_fit.predict(X_test)
    y_proba_xgb = xgb_fit.predict_proba(X_test)
    # y_proba_xgb = xgb_fit.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    N = np.int(y_proba_xgb.shape[0]*threshold)
    max_prob_xgb = np.argsort(y_proba_xgb[:,0])[-N:] 
    
    return cm, max_prob_xgb
    
# DNN

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint

def DNN(X_train, y_train, X_test, y_test, data_option, threshold):
    
    model_dir = os.path.join(data_dir, 'dnn_models')
    
    options = ["all", "GE_GT", "GT"]
    try:
        data_option = options.index(data_option);
    except ValueError:
        return "invalid choice: {}".format(data_option)
  

    y_pred__ = []
    y_proba_dnn__ = []
    
    for ii in range(10):

        if data_option == 'all':
            reconstructed_model = keras.models.load_model(os.path.join(model_dir, 're_model_all%d.h5' %ii))
        elif data_option == 'GE+GT':
            reconstructed_model = keras.models.load_model(os.path.join(model_dir, 're_model_GE_GT%d.h5' %ii))
        else:
            reconstructed_model = keras.models.load_model(os.path.join(model_dir, 're_model_GT%d.h5' %ii))
        
        # y_pred_ = reconstructed_model.predict_classes(X_test)
        y_pred_ = np.argmax(reconstructed_model.predict(X_test),axis=1)

        # y_proba_dnn_ = reconstructed_model.predict_proba(X_test)[:,0]
        y_proba_dnn_ = reconstructed_model.predict(X_test)[:, 0]
        y_pred__.append(y_pred_)
        y_proba_dnn__.append(y_proba_dnn_)
        
    pred_arr = np.vstack(y_pred__)    
    y_pred = np.zeros(len(y_test))
    for ii in range(len(y_test)):
        y_pred[ii]=np.bincount(pred_arr[:,ii]).argmax()
    

    y_proba_dnn = np.mean(y_proba_dnn__, axis=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    N = np.int(y_proba_dnn.shape[0]*threshold)
    max_prob_dnn = np.argsort(y_proba_dnn)[-N:]
    
    return cm, max_prob_dnn
    



if __name__ == "__main__":
    filename = sys.argv[1]
    pdb_name = sys.argv[2]
    np_name = sys.argv[3]
    data_option = sys.argv[4]
    threshold = float(sys.argv[5])
    

    feature_all, GE_GT, GT, test_feature = data_loading(filename)
    X_train, y_train, X_test, y_test = train_test(data_option, feature_all, GE_GT, GT, test_feature)
    c1, c2, c1_CA, c2_CA,TrueA, TrueB = TrueAB(pdb_name)

    rfc_cm, max_prob_rfc = RFC(X_train, y_train, X_test, y_test, threshold)
    xgb_cm, max_prob_xgb = XGB(X_train, y_train, X_test, y_test, threshold)
    dnn_cm, max_prob_dnn = DNN(X_train, y_train, X_test, y_test, data_option, threshold)

    
    A_rfc = max_prob_rfc//len(c2_CA)
    A_rfc_res = c1['residue_number'].unique()[np.unique(A_rfc)]
    B_rfc = max_prob_rfc%len(c2_CA)
    B_rfc_res = c2['residue_number'].unique()[np.unique(B_rfc)]
   
    
    A_xgb = max_prob_xgb//len(c2_CA)
    A_xgb_res = c1['residue_number'].unique()[np.unique(A_xgb)]
    B_xgb = max_prob_xgb%len(c2_CA)
    B_xgb_res = c2['residue_number'].unique()[np.unique(B_xgb)]
    
    
    A_dnn = max_prob_dnn//len(c2_CA)
    A_dnn_res = c1['residue_number'].unique()[np.unique(A_dnn)]
    B_dnn = max_prob_dnn%len(c2_CA)
    B_dnn_res = c2['residue_number'].unique()[np.unique(B_dnn)]
    
    A_true = c1['residue_number'].unique()[TrueA]
    B_true = c2['residue_number'].unique()[TrueB]
    
    
    np.savetxt(os.path.join(result_dir, '%s_true_A.csv' % filename), A_true, delimiter=',')
    np.savetxt(os.path.join(result_dir, '%s_true_B.csv' % filename), B_true, delimiter=',')
    
    np.savetxt(os.path.join(result_dir, '%s_rfc_A_%s.csv' % (filename, data_option)), A_rfc_res, delimiter=',')
    np.savetxt(os.path.join(result_dir, '%s_rfc_B_%s.csv' % (filename, data_option)), B_rfc_res, delimiter=',')
    np.save(os.path.join(result_dir, '%s_rfc_cm_%s.npy' % (filename, data_option)), rfc_cm)
    
    np.savetxt(os.path.join(result_dir, '%s_xgb_A_%s.csv' % (filename, data_option)), A_xgb_res, delimiter=',')
    np.savetxt(os.path.join(result_dir, '%s_xgb_B_%s.csv' % (filename, data_option)), B_xgb_res, delimiter=',')
    np.save(os.path.join(result_dir, '%s_xgb_cm_%s.npy' % (filename, data_option)), xgb_cm)
    
    np.savetxt(os.path.join(result_dir, '%s_dnn_A_%s.csv' % (filename, data_option)), A_dnn_res, delimiter=',')
    np.savetxt(os.path.join(result_dir, '%s_dnn_B_%s.csv' % (filename, data_option)), B_dnn_res, delimiter=',')
    np.save(os.path.join(result_dir, '%s_dnn_cm_%s.npy' % (filename, data_option)), dnn_cm)
    

