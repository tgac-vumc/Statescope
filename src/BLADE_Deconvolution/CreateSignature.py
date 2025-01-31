#!/usr/bin/python3
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CreateSignature.py
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Functions for Signature creation
# Author: Jurriaan Janssen (j.janssen4@amsterdamumc.nl) , Aryamaan Bose (a.bose1@amsterdamumc.nl)
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
from scipy.optimize import curve_fit
import scipy.sparse as sp
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


def Check_adata_validity(adata, celltype_key='celltype'):
    """
    Check the validity of an AnnData object and report descriptive statistics and potential issues.
    :param AnnData adata: Single-cell data in AnnData format.
    :param str celltype_key: column in adata.obs specifying cell types [default = 'celltype']
    """
    print("\n=== Validating AnnData Object ===")

    # Determine if the data matrix is sparse
    is_sparse = sp.issparse(adata.X)

    # Check for negative values
    if is_sparse:
        has_negative_values = np.any(adata.X.data < 0)
    else:
        has_negative_values = np.any(adata.X < 0)
    
    if has_negative_values:
        raise AssertionError("The data matrix (adata.X) contains negative values, which are not allowed. adata.X has to be in log, library-size corrected format.")

    print("No negative values found in the data matrix (adata.X).")

    ###CHECK AGAIN NEEDS FIXING 
    # Check if normalization and log transformation were performed
    if 'log1p' not in adata.uns:
        raise AssertionError(
            "The data does not appear to be log-transformed. "
            "Please use Scanpy's `sc.pp.normalize_total(adata, target_sum=1e4)` and `sc.pp.log1p(adata)`."
        )

    print("Validation passed: Data appears to be log-transformed.")
    
    # Check if the data is scaled to 1e4
    total_counts = adata.X.sum(axis=1).A1 if is_sparse else adata.X.sum(axis=1)
    
    # Check for excessively large values
    max_value = adata.X.max() if not is_sparse else adata.X.data.max()
    print(f"Maximum value in the data matrix (adata.X): {max_value}")
    if max_value > 100:
        print("Warning: Extremely large values detected in the data matrix (adata.X).  This might influence performance. adata.X has to be in log, library-size corrected format.")

    # Check the number of cells and genes
    num_cells, num_genes = adata.shape
    print(f"Number of cells: {num_cells}")
    print(f"Number of genes: {num_genes}")


    # Calculate total counts and check for outlier cells
    median_total_counts = np.median(total_counts)
    mad_total_counts = np.median(np.abs(total_counts - median_total_counts))
    upper_threshold = median_total_counts + 3 * mad_total_counts
    lower_threshold = median_total_counts - 3 * mad_total_counts
    outlier_cells = np.sum((total_counts > upper_threshold) | (total_counts < lower_threshold))
    print(f"Number of outlier cells based on total counts: {outlier_cells}")

    gene_counts = adata.X.sum(axis=0).A1 if is_sparse else adata.X.sum(axis=0)
    median_gene_counts = np.median(gene_counts)
    mad_gene_counts = np.median(np.abs(gene_counts - median_gene_counts))
    upper_gene_threshold = median_gene_counts + 3 * mad_gene_counts
    highly_expressed_genes = np.sum(gene_counts > upper_gene_threshold)
    print(f"Number of genes with extremely high expression: {highly_expressed_genes}")

    
    # Check for specified cell types if provided in adata.obs
    if celltype_key in adata.obs:
        unique_cell_types = adata.obs[celltype_key].unique()
        print(f"Cell types present: {list(unique_cell_types)}")
        print(f"Number of unique cell types: {len(unique_cell_types)}")
    else:
        raise AssertionError(f"No cell type information available under the specified key '{celltype_key}'.")

    # Summary statistics for data matrix
    data_matrix = adata.X.data if is_sparse else adata.X
    data_summary = {
        "mean": np.mean(data_matrix),
        "median": np.median(data_matrix),
        "std": np.std(data_matrix),
        "min": np.min(data_matrix),
        "max": np.max(data_matrix),
    }
    print("\nSummary statistics for the data matrix (adata.X):")
    for key, value in data_summary.items():
        print(f"  {key}: {value}")

    print("\nValidation complete.")



def CreateSignature(adata, celltype_key='celltype', CorrectVariance=True, n_highly_variable=3000):
    """ 
    Create Signature from AnnData object.
    :param AnnData adata: phenotyped scRNAseq data with: adata.X (log, library-size corrected) 
    :param str celltype_key: column in adata.obs containing cell phenotypes [default = 'celltype']
    :param bool CorrectVariance: Whether to run scran fitTrendVar in R to correct variance [default = True]
    :param int n_highly_variable: Number of hvgs to select for AutoGeneS marker detection [default = 3000]
    :returns: pandas.DataFrame Signature: Signature for deconvolution
    """


    # Validate the AnnData object
    Check_adata_validity(adata, celltype_key)

    print("\n=== Creating Signature ===")

    # Check if the celltype_key exists in adata.obs
    ###move it to validity script
    if celltype_key not in adata.obs:
        raise AssertionError(f"{celltype_key} is not in adata.obs. Specify which column contains the cell phenotypes")

    ### Drop cells with missing cell type annotations
    missing_count = adata.obs[celltype_key].isna().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} cells have missing '{celltype_key}' annotations. These cells will be excluded.")
        adata = adata[~adata.obs[celltype_key].isna()].copy()

    print("Cell type annotations are validated. Proceeding with calculations.")

    # Define cell types
    celltypes = pd.unique(adata.obs[celltype_key])

    # Calculate mean and std expression in cell types
    print("Calculating mean and standard deviation expressions for each cell type.")
    scExp = pd.concat(
        [
            pd.DataFrame(
                np.mean(adata[adata.obs[celltype_key] == ct].X.toarray(), axis=0).transpose(),
                columns=[f"scExp_{str(ct)}"],  # Ensure `ct` is a string
                index=adata.var_names,
            )
            for ct in celltypes
        ],
        axis=1,
    )
    
    scVar = pd.concat(
        [
            pd.DataFrame(
                np.std(adata[adata.obs[celltype_key] == ct].X.toarray(), axis=0).transpose(),
                columns=[f"scVar_{str(ct)}"],  # Ensure `ct` is a string
                index=adata.var_names,
            )
            for ct in celltypes
        ],
        axis=1,
    ).replace(0, 0.001)  # Replace 0s with a small pseudovalue
    print("Expression calculations completed.")

    # Correct variance if needed
    if CorrectVariance:
        print("Correcting variance using fitTrendVar.")
        scVar = pd.concat(
            [
                pd.DataFrame(
                    np.var(adata[adata.obs[celltype_key] == ct].X.toarray(), axis=0).transpose(),
                    columns=[f"scVar_{str(ct)}"],
                    index=adata.var_names,
                )
                for ct in celltypes
            ],
            axis=1,
        ).replace(0, 0.001)
        scVar = pd.concat(
            [fitTrendVar(scExp[f"scExp_{ct}"], scVar[f"scVar_{ct}"]) for ct in celltypes],
            axis=1,
        ).replace(0, 0.001)
        scVar.columns = [col.replace("scExp", "scVar") for col in scVar.columns]
        print("Variance correction completed.")

    print("Running AutoGeneS to select marker genes.")
    # Run AutoGeneS and define markers
    AutoGeneS = Run_AutoGeneS(adata, celltype_key, n_highly_variable)
    IsMarker = pd.DataFrame(
        {"IsMarker": [(gene in AutoGeneS) for gene in adata.var_names]}, index=adata.var_names
    )
    
    print("AutoGeneS completed. Compiling the final signature matrix.")

    # Concatenate dataframes to create the signature
    Signature = pd.concat([IsMarker, scExp, scVar], axis=1)
    Signature.index = adata.var_names
    
    print("Signature matrix creation complete.")

    return Signature

def Run_AutoGeneS(adata, celltype_key, n_highly_variable=3000):
    """ 
    Perform AutoGeneS marker selection from AnnData object
    :param AnnData adata: phenotyped scRNAseq data with: adata.X (log, library-size corrected) 
    :param str celltype_key: column in adata.obs containing cell phenotypes [default = 'celltype']
    :param int n_highly_variable: Number of hvgs to select for AutoGeneS marker detection [default = 3000]
    
    :returns: list AutoGeneS: list of marker genes
    """
    print("\n=== Gene Selection  ===")

    # Ensure data is in dense array format if needed
    if isinstance(adata.X, np.ndarray):
        data_array = adata.X
    else:
        data_array = adata.X.toarray()
    
    print(f"Subsetting to top {n_highly_variable} highly variable genes...")
    # Subset adata to top n hvgs
    hvgs = pd.DataFrame(data_array.transpose(), index=adata.var_names).apply(np.var, axis=1).sort_values().nlargest(n_highly_variable).index
    adata = adata[:, hvgs]
    
    # Define cell types
    celltypes = pd.unique(adata.obs[celltype_key])
    centroids_sc_hv = pd.DataFrame(index=adata.var_names, columns=celltypes)
    
    print("Calculating centroids for each cell type...")
    # Calculate celltype centroids
    for celltype in celltypes:
        print(f"Processing cell type: {celltype}")
        adata_filtered = adata[adata.obs[celltype_key] == celltype]
        sc_part = adata_filtered.X.T if isinstance(adata_filtered.X, np.ndarray) else adata_filtered.X.toarray().T
        centroids_sc_hv[celltype] = pd.DataFrame(np.mean(sc_part, axis=1), index=adata.var_names)
    
    print("Initializing AutoGeneS optimization...")
    # Run AutoGeneS
    ag.init(centroids_sc_hv.T)
    print("Running AutoGeneS optimization process. This may take some time...")
    ag.optimize(ngen=5000, seed=0, offspring_size=100, verbose=False)
    print("AutoGeneS optimization complete.")
    
    # Fetch AutoGeneS in list
    print("Fetching marker genes selected by AutoGeneS...")
    AutoGenes = centroids_sc_hv[ag.select(index=0)].index.tolist()
    print("Marker genes successfully extracted.")
    return AutoGenes
