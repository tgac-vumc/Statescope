#!/usr/bin/python3
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CreateSignature.py
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Functions for Signature creation
# Author: Jurriaan Janssen (j.janssen4@amsterdamumc.nl)
#
# Usage:"""
#
# TODO:
# 1) 
#
# History:
#  15-12-2024: File creation, write code
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 0.1  Import Libraries
#-------------------------------------------------------------------------------
import anndata as ad
import pandas as pd
import numpy as np
import subprocess
#import autogenes as ag

#-------------------------------------------------------------------------------
# 1.1  Define functions
#-------------------------------------------------------------------------------
def CreateSignature(adata, celltype_key = 'celltype', CorrectVariance = True):
    """ 
    Create Signature from AnnData object
    :param AnnData adata: phenotyped scRNAseq data with: adata.X (log, library-size corrected) 
    :param str celltype_key: column in adata.obs containing cell phenotypes [default = 'celltype']
    :param bool CorrectVariance: Whether to run scran fitTrendVar in R to correct variance [default = True]
    
    :returns: pandas.DataFrame Signature: Signature for deconvolution
    """
    # define celltypes
    celltypes = pd.unique(adata.obs[celltype_key])
    # Calculate mean and std expression in cell types
    scExp = pd.concat([pd.DataFrame(np.mean(adata[adata.obs.celltype == ct].X.toarray(),0).transpose(),columns=['scExp_'+ct], index = adata.var_names) for ct in celltypes], axis=1)
    scVar = pd.concat([pd.DataFrame(np.std(adata[adata.obs.celltype == ct].X.toarray(),0).transpose(),columns=['scExp_'+ct], index = adata.var_names) for ct in celltypes], axis=1).replace(0,0.001) # replace 0s with small pseudovalue
    # Correct variance
    if CorrectVariance == True:
        # Write to file
        scExp.to_csv('.tmp_Exp.txt', sep = '\t')
        scVar.to_csv('.tmp_Var.txt', sep = '\t')
        # Run R code
        subprocess.call("Rscript --vanilla Framework/BLADE_Deconvolution/CorrectVariance.R --input_mean .tmp_Exp.txt --input_var .tmp_Var.txt --output .tmp_CorrVar.txt", shell=True)
        # Read corrected variance
        scVar = pd.read_csv('.tmp_CorrVar.txt', sep = '\t')
        
    # Run AutoGeneS and define markers
    AutoGeneS = Run_AutoGeneS(adata,celltype_key)
    IsMarker = pd.DataFrame({'IsMarker':[(gene in AutoGeneS) for gene in adata_var_names]},index = adata.var_names)
    # Concatenate dataframes
    Signature = pd.concat([IsMarker,scExp,scVar],index=1)
    return Signature

def Run_AutoGeneS(adata,celltype_key):
    """ 
    Perform AutoGeneS marker selection from AnnData object
    :param AnnData adata: phenotyped scRNAseq data with: adata.X (log, library-size corrected) 
    :param str celltype_key: column in adata.obs containing cell phenotypes [default = 'celltype']
    
    :returns: list AutoGeneS: list of marker genes
    """
    # define celltypes
    celltypes = pd.unique(adata.obs[celltype_key])
    centroids_sc_hv = pd.DataFrame(index=adata.var_names,columns=celltypes)
    # Calculate celltype centroids
    for celltype in celltypes:
        adata_filtered = adata[adata.obs[args.celltype_level] == celltype]
        sc_part = adata_filtered.X.T
        centroids_sc_hv[celltype] = pd.DataFrame(np.mean(sc_part,axis=1),index=adata.var_names)
    # Run AutogeneS
    ag.init(centroids_sc_hv.T)
    ag.optimize(ngen=5000,seed=0,offspring_size=100,verbose=False)
    # Fetch AutoGeneS in list
    AutoGenes = ag.select(index=0).tolist()
    return AutoGenes
    
