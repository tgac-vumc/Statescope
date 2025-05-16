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
#  29-04-2025: New version with set your own k functionality 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 0.1  Import Libraries
#-------------------------------------------------------------------------------
import StateDiscovery.lib.pymf
from StateDiscovery.lib.pymf import cnmf
from StateDiscovery.lib.cnmf_helpers import *
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
def StateDiscovery_FrameWork(
        GEX, Omega, Fractions, celltype,
        weighing='Omega',
        K=None,
        n_iter=10,
        n_final_iter=100,
        min_cophenetic=0.95,
        max_clusters=10,
        Ncores=10):
    """
      Run the cNMF‐based state-discovery workflow for a single cell type.

      :param GEX:           Purified gene-expression matrix (genes × samples).
      :param Omega:         Gene-wise variance estimates for the same genes.
      :param Fractions:     Cell-type fractions per sample (used by some
                            weighing modes).
      :param celltype:      Name of the cell type being analysed (for logging).
      :param weighing:      Scaling applied in ``Create_Cluster_Matrix``  
                            (``'Omega'`` | ``'OmegaFractions'`` | ``'centering'``
                            | ``'no_weighing'``).
      :param K:             Desired number of states.  
                            • *None* ⇒ run a cophenetic sweep to choose *k*  
                            • int   ⇒ use that value and skip the sweep.
      :param n_iter:        Number of cNMF restarts **per k** during the sweep.
      :param n_final_iter:  Restarts for the final model at the chosen *k*.
      :param min_cophenetic:Threshold for the cophenetic coefficient; the first
                            k ≥ this value is selected.
      :param max_clusters:  Maximum k tested in the sweep (upper bound, excl.).
      :param Ncores:        CPU cores used for parallel cNMF runs.

      :return: ``(final_model, coeffs)``, where *coeffs* is the full list of
              cophenetic coefficients if a sweep was run, otherwise the single
              final coefficient.
      """
    # build the cluster matrix 
    data_scaled = Create_Cluster_Matrix(GEX, Omega, Fractions, celltype, weighing)

    data_dict   = {}         
    sweep_curve = None        # list of cophenetic coefficients if we sweep

   
    # 1) cophenetic sweep  (only when K is None)
 
    if K is None:
        print(f'A value of K is automatically selected between 2 and {max_clusters}')
        for k in range(2, max_clusters):
            print(f'Running initial cNMF ({n_iter} iterations) with K={k}')
            cNMF_model_k, cophcor_k, consensus_k = cNMF(data_scaled, k, n_iter, Ncores)

            
            H = cNMF_model_k.H
            cluster_assignments = [
                int(np.where(H[:, i] == H[:, i].max())[0] + 1)
                for i in range(H.shape[1])
            ]
            data_dict[k] = {
                "model":              cNMF_model_k,
                "cophcor":            cophcor_k,
                "consensus":          consensus_k,
                "cluster_assignments": cluster_assignments,
            }

        ks         = sorted(data_dict)
        sweep_curve = [data_dict[k]["cophcor"] for k in ks]
        nclust      = find_threshold(sweep_curve, ks, min_cophenetic) \
                      or biggest_drop(sweep_curve)
    else:
        nclust = K
    
    # 2) final long run at chosen k
    
    print(f'The selected value for K is {nclust}')
    print(f'Running final cNMF ({n_final_iter} iterations) with K={nclust}')
    cNMF_model, cophcors_final, consensus_matrix = \
        cNMF(data_scaled, nclust, n_final_iter, Ncores)


    # 3) return 
 
    if sweep_curve is not None:
        return cNMF_model, sweep_curve     # list of coefficients
    else:
        return cNMF_model, cophcors_final  # single float


# StateRetrieveal: Calculate state scores with predefined state loadings in external dataset
def StateRetrieval(GEX,Omega,celltype,StateLoadings,weighing = 'Omega',Fractions = None):
    """
      Run cNMF‐based state-retrieval for a single cell type with predefined Stateloadings.

      :param GEX:           Cell type-specifc gene expression matrix (genes × samples (or single cells)).
      :param Omega:         Gene-wise variance estimates for the same genes, can also be derived from scRNAseq (e.g. scVar).
      :param Fractions:     Cell-type fractions per sample (used by some
                            weighing modes).
      :param celltype:      Name of the cell type being analysed (for logging).
      :param weighing:      Scaling applied in ``Create_Cluster_Matrix``  
                            (``'Omega'`` | ``'OmegaFractions'`` | ``'centering'``
                            | ``'no_weighing'``).
      :return: ``StateScores``, StateScores of states retrieved in new data
    """
    # Find overlapping genes and subset
    Genes = [gene for gene in GEX.columns if gene in StateLoadings.index]
    data_scaled = Create_Cluster_Matrix(GEX.loc[:,Genes],Omega.loc[Genes,:],Fractions,celltype,weighing)
    StateLoadings = StateLoadings.loc[Genes,:]
    # Run cNMF recovery
    cNMF_model = cNMF_Retrieval(data_scaled,StateLoadings)
    StateScores = pd.DataFrame(np.apply_along_axis(lambda x: x/ sum(x),1,cNMF_model.H.T))
    return StateScores

                
    


