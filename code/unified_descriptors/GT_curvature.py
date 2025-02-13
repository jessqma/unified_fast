#!/usr/bin/env python
# coding: utf-8



import numpy as np
import os
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci


def ollivier_ricci(G):

    node = G.number_of_nodes()
    orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
    orc.compute_ricci_curvature()
    c_sum = []
    for ii in np.arange(node):
        L = list(orc.G[ii].values())
        c = []
        for jj in range(len(L)):
            # c.append(list(L[jj].values())[2])
            c.append(L[jj]['ricciCurvature'])

        c_sum.append(np.sum(c))

    return c_sum

def forman_ricci(G):

    node = G.number_of_nodes()
    frc = FormanRicci(G)
    frc.compute_ricci_curvature()
    c_sum = []
    for ii in np.arange(node):
        L = list(frc.G[ii].values())
        c = []
        # print('L', L)
        for jj in range(len(L)):
            # print(list(L[jj].values()))
            # c.append(list(L[jj].values())[2])
            c.append(L[jj]['formanCurvature'])

        c_sum.append(np.sum(c))

    return c_sum



