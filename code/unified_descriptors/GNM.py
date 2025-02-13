#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
from biopandas.pdb import PandasPdb
import pandas as pd
import networkx as nx
from prody import *


def gnm_sum_mode(pdb_name, n, chain):
    protein = parsePDB(pdb_name)
    calphas1 = protein.select('not hetatm and calpha and ' + ('(chain %s' + ' or chain %s' * (len(chain) - 1)) % tuple([_ for _ in chain]) + ')')

    gnm = GNM('protein')
    gnm.buildKirchhoff(calphas1, cutoff=7.)
    kirchhoff1 = gnm.getKirchhoff()
    M1 = gnm.calcModes('all')

    if gnm.numModes() > n:
        print('num mode is larger than 10')
        test_sq1 = np.square(gnm[0:n].getEigvecs().round(3))
        sum_mode_sq1 = np.sum(test_sq1, axis=1)
    else:
        print('num mode is smaller than 10')
        test_sq1 = np.square(gnm[0:gnm.numModes()].getEigvecs().round(3))
        sum_mode_sq1 = np.sum(test_sq1, axis=1)

    return sum_mode_sq1


def gnm_sum_mode_np(pdb_name, n):
    nano_ = parsePDB(pdb_name)
    nano = nano_.select('not element H')
    gnm = GNM('nanoparticle')
    gnm.buildKirchhoff(nano, cutoff=4.)
    kirchhoff = gnm.getKirchhoff()
    M = gnm.calcModes('all')
    if gnm.numModes() > n:
        print('num mode is larger than 10')
        test_sq1 = np.square(gnm[0:n].getEigvecs().round(3))
        sum_mode_sq1 = np.sum(test_sq1, axis=1)
    else:
        print('num mode is smaller than 10')
        mode_sq1 = []
        test_sq1 = np.square(gnm[0:gnm.numModes()].getEigvecs().round(3))
        sum_mode_sq1 = np.sum(test_sq1, axis=1)

    return sum_mode_sq1