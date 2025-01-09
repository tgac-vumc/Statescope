import sys
import os
os.chdir('../')
sys.path.insert(1, 'Framework/')
sys.path.insert(1, 'Framework/StateDiscovery/lib/pymf/')
import Statescope
from Statescope import Initialize_Statescope
import pandas as pd
import pickle
import anndata as ad

# Run Statescope with adata as signature (with 'celltype' in adata.obs)
Bulk = pd.read_csv('https://github.com/tgac-vumc/OncoBLADE/raw/refs/heads/main/data/Transcriptome_matrix_subset.txt', sep = '\t', index_col = 'symbol')


# --------------------------------------------------------------------
# Run Statescope with adata as signature (with 'celltype' in adata.obs)
Signature = ad.read('/net/beegfs/cfg/tgac/jjanssen4/oncoBLADE/2024NATCANCER/NSCLC_snakemake/output/scRNAseq/Kim/PerState/adata_PerState.h5ad')
# Signature can also be a valid Signature (pd.DataFrame, in the correct format)

Statescope_model = Initialize_Statescope(Bulk,Signature = Signature, Ncores = 40)
Statescope_model.Deconvolution()
Statescope_model.Refinement()
Statescope_model.StateDiscovery()

# --------------------------------------------------------------------
# Run Statescope with predefined signature
Statescope_model = Initialize_Statescope(Bulk,TumorType = 'NSCLC', Ncores = 40)
Statescope_model.Deconvolution()
Statescope_model.Refinement()
Statescope_model.StateDiscovery()

