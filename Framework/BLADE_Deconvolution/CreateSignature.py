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
#  08-01-2025: Add python implementation of scran fitTrendVar, ChatGPT version
#              (tested, copmared to R scran function)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 0.1  Import Libraries
#-------------------------------------------------------------------------------
import anndata as ad
import pandas as pd
import numpy as np
import subprocess
from scipy.optimize import curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess
import autogenes as ag

#-------------------------------------------------------------------------------
# 1.1  Define functions
#-------------------------------------------------------------------------------

def fitTrendVar(means,variances, min_mean = 0.1,frac = 0.025, parametric=True, lowess_flag=True, density_weights=True, nls_args=None, **kwargs):
    # Filtering out zero-variance and low-abundance genes
    valid_indices = (variances > 1e-8) & (means >= min_mean)
    means_valid = means[valid_indices]
    variances = variances[valid_indices]
    if len(variances) < 2:
        raise ValueError("Need at least 2 points for non-parametric curve fitting")
    # Default weighting by inverse density
    if density_weights:
        # Compute histogram of means_valid
        hist, bin_edges = np.histogram(means_valid, bins=100, density=True)
        
        # Ensure that bin indices are valid
        bin_indices = np.digitize(means_valid, bin_edges[:-1]) - 1  # -1 to ensure indices are within the range
        
        # Weights based on inverse density
        weights = 1 / (hist[bin_indices] + 1e-5)  # Add small value to avoid division by zero
    else:
        weights = np.ones_like(means_valid)
    # Parametric model (nonlinear curve fitting): y = ax / (x^n + b)
    def parametric_model(x, a, b, n):
        return a * x / (x**n + b)
    # Fitting the parametric curve
    if parametric:
        try:
            params, _ = curve_fit(parametric_model, means_valid, variances, p0=[1, 1, 1], sigma=weights)
            parametric_fun = lambda x: parametric_model(x, *params)
        except Exception as e:
            print("Parametric fit failed:", e)
            parametric_fun = lambda x: np.zeros_like(x)  # Fall back to zeros
    else:
        parametric_fun = lambda x: x  # Identity function if no parametric fitting
    # Log-transform the variances for fitting residuals
    log_variances = np.log(variances)
    left_edge = np.min(means_valid)
    if lowess_flag:
        # LOWESS smoothing, frac paramter is empirically chosen
        lowess_fit = lowess(log_variances - np.log(parametric_fun(means_valid)), means_valid,frac=frac, **kwargs)
        loess_fun = lambda x: np.exp(np.interp(x, lowess_fit[:, 0], lowess_fit[:, 1]))
        def unscaled_fun(x):
            return loess_fun(x) * parametric_fun(x)
    else:
        unscaled_fun = parametric_fun
    # Correct the scaling for the fit (unlogged values)
    corrected_fit = unscaled_fun(means)
    return  corrected_fit



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
    if CorrectVariance:
        scVar = fitTrendVar(scExcp, scVar)
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
    
