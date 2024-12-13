#!/usr/bin/python3
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# cNMF.py
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Run cNMF for state discovery
#
# conda: .snakemake/conda/63ea5141
# Author: Jurriaan Janssen (j.janssen4@amsterdamumc.nl)
#
# Usage:
#
# TODO:
# 1) 
#
# History:
#  09-01-2024: File creation, write code
#  13-12-2024: Edits for Statescope FrameWork
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 0.1  Import Libraries
#-------------------------------------------------------------------------------
import lib.pymf
from lib.pymf import cnmf
from lib.cnmf_helpers import *
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import random
from scipy.cluster.hierarchy import average, cophenet
from scipy.spatial.distance import  pdist
import pickle
from joblib import Parallel, delayed


#-------------------------------------------------------------------------------
# 3.1 Run cNMF analysis
#------------------------------------------------------------------------------- 
def StateDiscovery_FrameWork(GEX,Omega,Fractions,celltype,K=None,n_iter=10,n_final_iter=100,min_cophentic,max_clusters,Ncores):
    data_scaled = Create_Cluster_Matrix(GEX,Omega,Fractions,weighing)
    # Run Initial cNMF runs
    data_dict = dict()
    if K == None:
        for k in range(2,max_clusters):
            cNMF_model, cophcor, consensus_matrix = cNMF(data_scaled, k, n_iter, Ncores)
            H = cNMF_model.H
            cluster_assignments = []
            for i in range(H.shape[1]):
                cluster_assignments.append(int(np.where(H[:,i] == max(H[:,i]))[0] + 1))    
            data_dict[k] = {'model':cNMF_model,'cophcor':cophcor, 'consensus': consensus_matrix,'cluster_assignments':cluster_assignments}

            # Determine K
            cophcors = [d['cophcor'] for d in data_dict.values()]
            ks = [k for k in data_dict.keys()]

            nclust = find_threshold(cophcors,ks,min_cophenetic=min_coph)
            drop = biggest_drop(cophcors)
            if not nclust:
                nclust = drop
    else:
        nclust = K
    # Run Final model
    cNMF_model, cophcor, consensus_matrix = cNMF(data_scaled, nclust, n_iter_final, Ncores)

                
    


