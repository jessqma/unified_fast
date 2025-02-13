#!/usr/bin/env python
# coding: utf-8



import numpy as np
import os
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci


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



