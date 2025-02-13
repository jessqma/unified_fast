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


def gnm_sum_mode(pdb_name, n, chain):
    protein = parsePDB(pdb_name)
    calphas1 = protein.select('calpha and chain %s' %chain)
    gnm = GNM('protein')
    gnm.buildKirchhoff(calphas1, cutoff=7.)
    kirchhoff1 = gnm.getKirchhoff()
    M1 = gnm.calcModes('all')

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
        for ii in range(gnm.numModes()):
            mode1 = gnm[ii].getEigvecs().round(3)
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


