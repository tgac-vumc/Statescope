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
# 30-04-2025: Functions fixed, added more informative print statements, refined functionality and added more parameters to the functions

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


def looks_logged(adata, max_cutoff=50, int_tolerance=0.99):
    """
    Heuristically decide if ``adata.X`` is log-transformed.

    Returns True / False  (or None if it’s ambiguous).
    """
    X = adata.X
    if hasattr(X, "_view_args"):   # AnnData view
        X = X.copy()

    if sp.issparse(X):
        X = X.toarray()  

    # 1) value-range heuristic
    if X.max() > max_cutoff:
        return False          # definitely raw

    # 2) integer-ness heuristic
    frac_part = np.abs(X - np.rint(X))
    if (frac_part < 1e-10).mean() > int_tolerance:
        return False          # almost all integers → raw

    # 3) raw layer present?
    if getattr(adata, "raw", None) is not None:
        return True           # Scanpy default after log1p

    # 4) total-count correlation
    totals = X.sum(axis=1)
    means  = X.mean(axis=1)
    rho    = np.corrcoef(totals, means)[0, 1]
    if rho < 0.2:
        return True
    if rho > 0.7:
        return False

    # ambiguous
    return None



def Check_adata_validity(adata, celltype_key='celltype'):
    
    
    print("\n=== Validating AnnData scRNAseq Object ===")

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

    is_logged = looks_logged(adata)
    if is_logged is False:
        raise AssertionError("adata.X looks like raw counts; please normalise & log-transform.")
    elif is_logged is None:
        print("Warning: unable to decide confidently whether adata.X is logged.")
    else:
        print("Log-normalised data detected.")

    
    # # Check if the data is scaled to 1e4
    # total_counts = adata.X.sum(axis=1).A1 if is_sparse else adata.X.sum(axis=1)
    
    # Check for excessively large values
    # max_value = adata.X.max() if not is_sparse else adata.X.data.max()
    # print(f"Maximum value in the data matrix (adata.X): {max_value}")
    # if max_value > 100:
    #     print("Warning: Extremely large values detected in the data matrix (adata.X).  This might influence performance. adata.X has to be in log, library-size corrected format.")



    # Check the number of cells and genes
    num_cells, num_genes = adata.shape
    print(f"Number of cells: {num_cells}")
    print(f"Number of genes: {num_genes}")


        
    # # Calculate total counts per cell and define outlier thresholds
    # median_total = np.median(total_counts)
    # mad_total = np.median(np.abs(total_counts - median_total))
    # lower_total_thresh = median_total - 3 * mad_total
    # upper_total_thresh = median_total + 3 * mad_total
    # num_outlier_cells = np.sum((total_counts < lower_total_thresh) | (total_counts > upper_total_thresh))

    # # Calculate total gene expression and define a threshold for highly expressed genes
    # if is_sparse:
    #     gene_totals = adata.X.sum(axis=0).A1
    # else:
    #     gene_totals = np.sum(adata.X, axis=0)
    # median_gene = np.median(gene_totals)
    # mad_gene = np.median(np.abs(gene_totals - median_gene))
    # upper_gene_thresh = median_gene + 3 * mad_gene
    # num_high_expr_genes = np.sum(gene_totals > upper_gene_thresh)

    # # Construct and print a summary message
    # print("=== QC Summary ===")
    # print(f"Total counts per cell:")
    # print(f"  - Median: {median_total:.2f}")
    # print(f"  - Median Absolute Deviation (MAD): {mad_total:.2f}")
    # print(f"  - Lower threshold (median - 3*MAD): {lower_total_thresh:.2f}")
    # print(f"  - Upper threshold (median + 3*MAD): {upper_total_thresh:.2f}")
    # print(f"  -> Outlier cells (counts outside this range): {num_outlier_cells} out of {len(total_counts)} cells")
    # print("")
    # print("Gene expression summary:")
    # print(f"  - Median total gene expression: {median_gene:.2f}")
    # print(f"  - MAD: {mad_gene:.2f}")
    # print(f"  - Threshold for high expression (median + 3*MAD): {upper_gene_thresh:.2f}")
    # print(f"  -> Genes with extremely high expression: {num_high_expr_genes} genes")

        
    # # Check for specified cell types if provided in adata.obs
    if celltype_key in adata.obs:
        unique_cell_types = adata.obs[celltype_key].unique()
        print(f"Cell types present: {list(unique_cell_types)}")
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

    # print("\nValidation complete.")

# ──────────────────────────────────────────────────────────────────────────
# Bland–Altman QC   (Bulk  vs.  raw scRNA counts in `adata.X`)
# ──────────────────────────────────────────────────────────────────────────
def bland_altman_bulk_vs_scrna(Bulk: pd.DataFrame,
                               adata: ad.AnnData,
                               *,
                               cutoff: float = 1.96,
                               return_outliers: bool = False) -> None:
    """
    Print how many genes show bulk–scRNA disagreement outside ±cutoff·SD.

    Parameters
    ----------
    Bulk   : DataFrame
        Bulk-expression matrix (genes × samples).
    adata  : AnnData
        The **same** AnnData you pass to ``CreateSignature``; gene index must
        overlap Bulk’s.
    cutoff : float, default 1.96
        Multiples of the SD that define the limits of agreement.
    """
    common = Bulk.index.intersection(adata.var_names)
    if len(common) == 0:
        print("Bland-Altman: no common genes — skipped.")
        return set() if return_outliers else None

    X = adata[:, common].X
    if sp.issparse(X):
        X = X.toarray()

    scrna_mean = X.mean(axis=0)                 # (n_genes,)
    bulk_mean  = Bulk.loc[common].mean(axis=1).values
    diff       = scrna_mean - bulk_mean
    mdiff      = diff.mean()
    sdiff      = diff.std()
    upper      = mdiff + cutoff * sdiff
    lower      = mdiff - cutoff * sdiff
    mask_out   = (diff > upper) | (diff < lower)
    n_sig      = mask_out.sum()

    print(f"{n_sig}/{len(common)} gene expression differ between bulk and signature "
            f"beyond ±{cutoff:.2f}·SD ")
    
    if return_outliers:
        return set(common[mask_out])


def CreateSignature(adata, celltype_key='celltype', CorrectVariance=True, n_highly_variable=2000, fixed_n_features = None, MarkerList = None, Bulk: pd.DataFrame | None = None, drop_sigdiff: bool = False):
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
    if celltype_key not in adata.obs:
        raise AssertionError(f"{celltype_key} is not in adata.obs. Specify which column contains the cell phenotypes")

    # Drop cells with missing cell type annotations
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
                columns=[f"scExp_{str(ct)}"],  
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
                columns=[f"scVar_{str(ct)}"],  
                index=adata.var_names,
            )
            for ct in celltypes
        ],
        axis=1,
    )
    # Add 0.01 to all values (instead of replacing zeros)
    scVar = scVar + 0.01

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
        )
        scVar = scVar + 0.01

       
        scVar = pd.concat(
            [fitTrendVar(scExp[f"scExp_{ct}"], scVar[f"scVar_{ct}"]) for ct in celltypes],
            axis=1,
        )
        # Add 0.01 to all values again, after fitTrendVar
        scVar = scVar + 0.01
        
        # Rename columns
        scVar.columns = [col.replace("scExp", "scVar") for col in scVar.columns]
        print("Variance correction completed.")

    g_mean = scExp.values.flatten()
    g_var  = scVar.values.flatten()
    print(f"» Gene‐wise *means*  range:  min={g_mean.min():.3f}  "
          f"max={g_mean.max():.3f}")
    print(f"» Gene‐wise *vars*   range:  min={g_var.min():.3f}  "
          f"max={g_var.max():.3f}")

    if g_var.min() < 1e-6 or g_var.max() > 10:
        print("Warning: extremely small or extremely large variances detected; "
              "this may affect downstream performance.")

    if MarkerList:
        print("MarkerList provided by user – AutoGeneS will be skipped.")
        IsMarker = pd.Series(
            adata.var_names.isin(MarkerList),
            index=adata.var_names,
            name="IsMarker"
        )
    else:
        print("Running AutoGeneS to select marker genes.")
        AutoGenes = Run_AutoGeneS(
            adata,
            celltype_key=celltype_key,
            n_highly_variable=n_highly_variable,
            fixed_n_features=fixed_n_features
        )
        print(f"{len(AutoGenes)} marker genes selected by AutoGeneS.")

        print("AutoGeneS completed. Compiling the final signature matrix.")
        IsMarker = pd.Series(
            adata.var_names.isin(AutoGenes),
            index=adata.var_names,
            name="IsMarker"
        )

    # ---- Assemble signature ------------------------------------------- #
    Signature = pd.concat([IsMarker, scExp, scVar], axis=1)
    Signature.index = adata.var_names
    print("Signature matrix creation complete.")

    if Bulk is not None:
        outliers = bland_altman_bulk_vs_scrna(
            Bulk, adata, return_outliers=drop_sigdiff
        )
        if drop_sigdiff and outliers:
            before = Signature.shape[0]
            Signature = Signature.drop(outliers, errors="ignore")
            print(f"Removed {before - Signature.shape[0]} "
                    f"Genes with significantly differing expression levels between bulk and signature")

    return Signature



def _get_mean_var(X):
    """
    Calculate mean and variance for each gene across cells.
    
    Parameters
    ----------
    X : np.ndarray or sp.sparse.spmatrix
        Expression matrix (cells × genes).
        
    Returns
    -------
    mean : np.ndarray
        Mean expression for each gene.
    var : np.ndarray
        Variance expression for each gene.
    """
    if sp.issparse(X):
        mean = X.mean(axis=0)
        var = X.power(2).mean(axis=0) - np.square(mean)
        mean = mean.A1
        var = var.A1
    else:
        mean = np.mean(X, axis=0)
        var = np.var(X, axis=0)
    return mean, var


def seurat_hvg_custom(
    adata,
    n_top_genes: int | None = None,
    min_disp: float = 0.5,
    max_disp: float = float('inf'),
    min_mean: float = 0.0125,
    max_mean: float = 3,
    n_bins: int = 20,
) -> pd.Index:
    
    """
Select highly-variable genes (HVGs) with a Seurat-style workflow.

The function reproduces the “Seurat” flavour of Scanpy’s
*highly_variable_genes*:

1.  Revert log-transformed counts back to linear space (handles
    ``adata.uns['log1p']`` if present).
2.  Compute gene-wise mean and variance, then raw dispersion =
    *variance / mean*.
3.  Log-transform both mean and dispersion to stabilise scale.
4.  Bin genes by mean expression (*n_bins*) and normalise dispersion in
    each bin:  
    *(dispersion – bin-mean) / bin-sd*.
5.  **Return**  
    • the top *n_top_genes* by normalised dispersion **or**  
    • genes that satisfy user-defined cut-offs on mean and dispersion.

Parameters
----------
adata : AnnData
    Single-cell object whose ``adata.X`` contains **raw counts** that have
    already been log-normalised via *sc.pp.log1p*.
n_top_genes : int | None, default ``None``
    If given, ignore all cut-offs and simply return the *n_top_genes*
    genes with the largest normalised dispersion.
min_disp, max_disp : float, default 0.5, ∞
    Lower / upper thresholds applied to the *normalised* dispersion when
    *n_top_genes is None*.
min_mean, max_mean : float, default 0.0125, 3
    Lower / upper thresholds applied to the (log-scaled) mean expression
    when *n_top_genes is None*.
n_bins : int, default 20
    Number of expression bins used for dispersion normalisation.

Returns
-------
pandas.Index
    Index of gene names flagged as highly variable.
"""
    # Get the data matrix
    X = adata.X

    # Handle log1p transformation
    if hasattr(X, "_view_args"):  # AnnData array view
        X = X.copy()

    X = X.copy()
    log1p_info = adata.uns.get("log1p", {})
    base = log1p_info.get("base", None)
    if base is not None:
        X *= np.log(base)
    # use out if possible. only possible since we copy the data matrix
    if isinstance(X, np.ndarray):
        np.expm1(X, out=X)
    else:
        X = np.expm1(X)

    # Compute mean and variance
    mean, var = _get_mean_var(X)
    
    # Compute dispersion
    mean[mean == 0] = 1e-12  # set entries equal to zero to small value
    dispersion = var / mean
    
    # Logarithmize mean and dispersion as in Seurat
    dispersion[dispersion == 0] = np.nan
    dispersion = np.log(dispersion)
    mean = np.log1p(mean)
    
    # Create DataFrame with metrics
    df = pd.DataFrame(dict(zip(["means", "dispersions"], (mean, dispersion))))
    df.index = adata.var_names  # Set gene names as index
    
    # Bin genes by mean expression
    df["mean_bin"] = pd.cut(df["means"], bins=n_bins)
    
    # Compute bin statistics
    disp_grouped = df.groupby("mean_bin", observed=True)["dispersions"]
    disp_bin_stats = disp_grouped.agg(avg="mean", dev="std")
    
    # Handle single-gene bins
    one_gene_per_bin = disp_bin_stats["dev"].isnull()
    if one_gene_per_bin.any():
        disp_bin_stats.loc[one_gene_per_bin, "dev"] = disp_bin_stats.loc[one_gene_per_bin, "avg"]
        disp_bin_stats.loc[one_gene_per_bin, "avg"] = 0
    
    # Map bin statistics back to genes
    df["avg"] = df["mean_bin"].map(disp_bin_stats["avg"]).astype(float)
    df["dev"] = df["mean_bin"].map(disp_bin_stats["dev"]).astype(float)
    
    # Compute normalized dispersion
    df["dispersions_norm"] = (df["dispersions"] - df["avg"]) / df["dev"]
    
    if n_top_genes is not None:
        # When n_top_genes is provided, ignore all cutoffs and just select top genes
        # by normalized dispersion.
        df = df.sort_values("dispersions_norm", ascending=False, na_position="last")
        df["highly_variable"] = np.arange(df.shape[0]) < n_top_genes
    else:
        # Apply cutoff thresholds
        df["highly_variable"] = (
            (df["means"] > min_mean) & 
            (df["means"] < max_mean) & 
            (df["dispersions_norm"] > min_disp) & 
            (df["dispersions_norm"] < max_disp)
        )
    
    # Return gene names from the DataFrame index
    return df.index[df["highly_variable"]]


# import scanpy as sc




def Run_AutoGeneS(
    adata,
    celltype_key,
    n_highly_variable,
    fixed_n_features,
):
    """ 
    Perform AutoGeneS marker selection from an AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        Phenotyped scRNAseq data with adata.X (log, library-size corrected).
    celltype_key : str
        Column in adata.obs containing cell phenotypes.
    n_highly_variable : int or None, default=3000
        Number of highly variable genes (HVGs) to select for AutoGeneS marker detection.
        If set to None, HVGs are selected solely based on cutoff thresholds.
    fixed_n_features : int or None, default=None
        If provided, the number of genes to fix for the final solution
        (AutoGeneS will be run in 'fixed' mode).
    
    Returns
    -------
    AutoGenes : list
        List of marker genes selected by AutoGeneS.
    """
    print("\n=== Gene Selection  ===")

    # Ensure data is in dense array format if needed
    if not isinstance(adata.X, np.ndarray):
        data_array = adata.X.toarray()
    else:
        data_array = adata.X

    # 1) Select HVGs
    if n_highly_variable is None:
        print("Subsetting to HVGs based on default cutoff thresholds...")
        hvgs = seurat_hvg_custom(
            adata,
            n_top_genes=None,
            min_disp=0.5,
            max_disp=float('inf'),
            min_mean=0.0125,
            max_mean=4,
            n_bins=20
        )

    else:
        print(f"Subsetting to top HVGs using ranking...")
        hvgs = seurat_hvg_custom(
            adata,
            n_top_genes=n_highly_variable,
            min_disp=0.5,
            max_disp=float('inf'),
            min_mean=0.0125,
            max_mean=4,
            n_bins=20
        )
        
    print(f"Number of HVGs retained for AutoGeneS: {len(hvgs)}")

    # # Subset to HVGs
    adata = adata[:, hvgs]
    

    # If using fixed_n_features, ensure it does not exceed the number of available genes
    if fixed_n_features is not None and fixed_n_features > adata.shape[1]:
        raise ValueError(
            f"fixed_n_features={fixed_n_features} exceeds number of HVGs "
            f"({adata.shape[1]}). Please specify a smaller value."
        )

    # 2) Calculate cell-type centroids
    print("\nCalculating centroids for each cell type...")
    celltypes = pd.unique(adata.obs[celltype_key])
    centroids_sc_hv = pd.DataFrame(index=adata.var_names, columns=celltypes)

    for celltype in celltypes:
        print(f"  Processing cell type: {celltype}")
        mask = (adata.obs[celltype_key] == celltype)
        adata_filtered = adata[mask]
        
        # Convert to dense if needed
        sc_part = (
            adata_filtered.X.T if isinstance(adata_filtered.X, np.ndarray)
            else adata_filtered.X.toarray().T
        )
        centroids_sc_hv[celltype] = np.mean(sc_part, axis=1)

    # 3) Run AutoGeneS
    print("\nInitializing AutoGeneS optimization...")
    ag.init(centroids_sc_hv.T)

    if fixed_n_features is not None:
        print(f"AutoGeneS running in fixed mode"
              f"with output of {fixed_n_features} genes.")
    else:
        print(f"AutoGeneS running in default mode")
    

    print("Running AutoGeneS optimization process. This may take some time...")
   
    if fixed_n_features is not None:
        # Fixed mode
        ag.optimize(ngen=5000, nfeatures=fixed_n_features,  offspring_size=100,  seed=0, mode='fixed', verbose = False)
    else:
        # Default mode
        ag.optimize(ngen=5000, seed=0,  offspring_size=100,  verbose=False)

    print("AutoGeneS optimization complete.")

    # 4) Retrieve the solution
    print("Fetching marker genes selected by AutoGeneS...")
    top_solution_genes = ag.select(index=0)
    AutoGenes = centroids_sc_hv[top_solution_genes].index.tolist()
    print("Marker genes successfully extracted.")

    return AutoGenes

