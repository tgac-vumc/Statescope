#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# cnmf_helpers.py
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# # Helper functions for cNMF Discovery
# Author: Jurriaan Janssen (j.janssen4@amsterdamumc.nl)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 0.1  Import Libraries
#-------------------------------------------------------------------------------
from StateDiscovery.lib.pymf import cnmf
import numpy as np
from scipy.spatial.distance import  pdist
from scipy.cluster.hierarchy import average, cophenet
from joblib import Parallel, delayed
import random
import pandas as pd


#-------------------------------------------------------------------------------
# 0.1  Define functions
#-------------------------------------------------------------------------------
# Function to extract hard cluster assignments form soft clusters
def cluster_assignment(H):
    return [np.argmax(H[:,i]) for i in range(H.shape[1])]

# Function to calculate connectivity matrix for hard cluster assignments
def connectivity(clusters):
    Nsample = len(clusters)
    connectivity_matrix = np.zeros((Nsample,Nsample))
    for i in range(Nsample):
        connectivity_matrix[i,np.where(clusters == clusters[i])] = 1
    return connectivity_matrix

# calculate cophenetic coefficient for consensus matrices
def Calculate_cophcor(X):
    dmat = pdist(X)
    Z = average(dmat)
    coph = cophenet(Z)
    return np.corrcoef(dmat,coph)[0,1]

    
def cNMF(data,k,nrun,ncores,niter=1000):
    def Run_cNMF(data,k,seed,niter=1000):
        random.seed(seed)
        model = cnmf.CNMF(np.transpose(data), num_bases=k, niter=niter)
        model.factorize(niter=niter)
        return model
    
    # perform NMF in parralel with different random seeds
    models = Parallel(n_jobs=ncores, verbose=10)(
        delayed(Run_cNMF)(data, k, i, niter)
        for i in range(nrun))
    # Fetch cluster assignments, connectivity matrices and calculate the consensus matrix
    consensus_matrix = sum([connectivity(cluster_assignment(mod.H)) for mod in models]) / nrun
    # Calculate the cophenetic correlation coefficient
    cophcor = Calculate_cophcor(consensus_matrix)
    # Calculate objective function and select best model
    objective = [mod.frobenius_norm() for mod in models]
    # pick model with lowest objective value
    model = models[objective.index(min(objective))]
    return model, cophcor, consensus_matrix

def find_threshold(cophcors,ks,min_cophenetic=0.95):
    cross = [False, False] + [((cophcors[i] < min_cophenetic) & (cophcors[i-1] >= min_cophenetic) & (cophcors[i-2] >= min_cophenetic)) for i in range(2,len(cophcors))]
    if any(cross) == False:
        return ks[-1]
    else:
        last_cross = [i for i,x in enumerate(cross) if x][-1]
        diff = [abs(min_cophenetic-cophcors[last_cross-1]),min_cophenetic-cophcors[last_cross]]
        return ks[last_cross - 2 + [i for i,x in enumerate(diff) if x == min(diff)][0]]

def biggest_drop(cophcors):
    cross = [-100] + [cophcors[i-1] - cophcors[i] for i in range(1,len(cophcors))]
    return cross.index(max(cross))-1



def Create_Cluster_Matrix(GEX,Omega,Fractions,celltype,weighing = 'Omega'):
    if weighing == 'Omega':
        Cluster_matrix = ((GEX - np.mean(GEX,axis=0))*Omega.loc[:,celltype].transpose()).to_numpy()
    elif weighing == 'OmegaFractions':
        Cluster_matrix = ((GEX - np.mean(GEX,axis=0))*Omega.loc[:,celltype].transpose())
        Cluster_matrix = Cluster_matrix.mul(Fractions[celltype], axis = 0).to_numpy()
    elif weighing == 'centering':
        Cluster_matrix = GEX - np.mean(GEX,axis=0).to_numpy()
    elif weighing == 'no_weighing':
        Cluster_matrix = GEX.to_numpy()
    return Cluster_matrix





