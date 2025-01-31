#!/usr/bin/python3
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Statescope.py
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Statescope framework
# Author: Jurriaan Janssen (j.janssen4@amsterdamumc.nl), Aryamaan Bose (a.bose1@amsterdamumc.nl)
#
# Usage:
"""
Statescope_model = Initialize_Statescope(Bulk,TumorType = 'NSCLC')
Statescope_model.Deconvolution()
Statescope_model.Refinement()
Statescope_model.Refinement()
Statescope_model.StateDiscovery()

"""
#
# TODO:
# 1) 
#
# History:
#  13-12-2024: File creation, write code, test Deconvolution and Refinement
#  14-12-2024: Finish StateDiscovery testing
#  20-01-2025: Additional checks, Utility and Visualization Functions 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 0.1  Import Libraries
#-------------------------------------------------------------------------------
from BLADE_Deconvolution.CreateSignature import CreateSignature
from BLADE_Deconvolution.BLADE import Framework_Iterative,Purify_AllGenes
from StateDiscovery.cNMF import StateDiscovery_FrameWork
import StateDiscovery.cNMF
from StateDiscovery.lib import pymf
import pandas as pd
import numpy as np
from collections.abc import Iterable
import anndata as ad
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import requests
from io import StringIO
import os  

#-------------------------------------------------------------------------------
# 1.1  Define Statescope Object
#-------------------------------------------------------------------------------
class Statescope:
    def __init__(self, Bulk, scExp,scVar,Samples,Celltypes,Genes,Markers,Ncores):
        self.Bulk = Bulk
        self.scExp = scExp
        self.scVar = scVar
        self.Samples = Samples
        self.Celltypes = Celltypes
        self.Genes = Genes
        self.Markers = Markers
        self.Ncores = Ncores

        self.isDeconvolutionDone = False
        self.isRefinementDone = False
        self.isStateDiscoveryDone = False


    def Deconvolution(self, Ind_Marker=None,
                        Alpha=1, Alpha0=1000, Kappa0=1, sY=1,
                        Nrep=10, Njob=10, fsel=0, Update_SigmaY=False, Init_Trust=10,
                        Expectation=None, Temperature=None, IterMax=100):
        """ 
        Perform BLADE Deconvolution
        :param Statescope self: Initialized Statescope
        :param int Alpha: BLADE Hyperparameter [default = 1]
        :param int Alpha0: BLADE Hyperparameter [default = 1000]
        :param int Kappa0: BLADE Hyperparameter [default = 1]
        :param int sY: BLADE Hyperparameter [default = 1]
        :param int Nrep: Number of BLADE initializations [default = 10]
        :param int Njob: Number of parralel jobs [default = 10]
        :param bool Update_SigmaY: Bool indicating whether SigmaY is iteratively optimized (experimental) [default = False]
        :param int Init_Trust: Parameter to weigh initial expectation fractions [default = 10]
        :param numpy.array Expectation: prior expectation of fractions (Nsample,Ncell) [default = None]
        :param [int1,int2] Temperature: Simulated annealing of optimization (Experimental) [default = None]
        :param int IterMax: Number of maximum iterations of optimzation [default = 100]

        :returns: BLADE_Object self.BLADE: BLADE object
        :returns: pandas.DataFrame self.Fractions: Dataframe with posterior fractions (Nsample,Ncell)
        """

        # Prepare Signature with markers only
        scExp_marker = self.scExp.loc[self.Markers,:].to_numpy()
        scVar_marker = self.scVar.loc[self.Markers, :].to_numpy()
        
        # Prepare Bulk (select/match genes)
        Y = self.Bulk.loc[self.Markers,self.Samples].to_numpy()

        # Check validity of Expectation (prior fractions)
        Check_Expectation_validity(Expectation)
        
        # Excecute BLADE Deconvolution: FrameWork iterative
        final_obj, best_obj, best_set, outs = Framework_Iterative(scExp_marker, scVar_marker, Y, Ind_Marker,
                        Alpha, Alpha0, Kappa0, sY,
                        Nrep, self.Ncores, fsel, Update_SigmaY, Init_Trust,
                        Expectation, Temperature, IterMax)
        # Save BLADE result in Statescope object
        self.BLADE = final_obj
        # Save fractions as dataframe in object
        self.Fractions = pd.DataFrame(final_obj.ExpF(final_obj.Beta), index = self.Samples,columns = self.Celltypes)
        # Add bool
        self.isDeconvolutionDone = True
        print("Deconvolution completed successfully.")

        
    # Perform Gene Expression Refinement
    def Refinement(self,weight=100,GeneList = None):
        """ 
        Perform Gene expression refinement with all genes
        :param Statescope self: Statescope
        :param int weight: Parameter to weigh down fraction estimation objective [default = 100]
        :param list GeneList: Genes to use for refinement [default = None, all genes]

        :returns: BLADE_Object self.BLADE_final: BLADE object
        :returns: {ct:pandas.DataFrame} self.GEX: Dictionary of cell type specific GEX {ct:ctSpecificGEX}
        """
        if not self.isDeconvolutionDone:
            raise Exception("Deconvolution must be completed before Refinement.")
        
        if GeneList:
            self.Genes = [gene for gene in Genes if gene in GeneList]
        # Prepare Signature
        scExp_All = self.scExp.loc[self.Genes, :].to_numpy()
        scVar_All = self.scVar.loc[self.Genes, :].to_numpy()
        # Prepare Bulk (select/match genes)
        Y = self.Bulk.loc[self.Genes,self.Samples].to_numpy()
        # Perform gene expression refinement with all genes in signature
        obj = Purify_AllGenes(self.BLADE, scExp_All, scVar_All,Y,self.Ncores, weight)
        # create output GEX dictionary
        GEX = {ct:pd.DataFrame(obj.Nu[:,:,i],index=self.Samples,columns=self.Genes) for i,ct in enumerate(self.Celltypes)}
        Omega = {ct:pd.DataFrame(obj.Omega[:,i],index=self.Genes,columns=[ct]) for i,ct in enumerate(self.Celltypes)}
        # Store in Statescope object
        self.BLADE_final = obj
        self.GEX = GEX
        self.Omega = Omega
        
        self.isRefinementDone = True
        print("Refinement completed successfully.")


    def StateDiscovery(self, celltype=None, K=None, weighing='Omega', n_iter=10, n_final_iter=100, min_cophenetic=0.9, max_clusters=10):
        """ 
        Perform StateDiscovery from ctSpecificGEX using cNMF.

        :param celltype: List of cell types or a single cell type for which to perform state discovery.
                        If None, will use all available cell types from the data.
        :param K: List or single value of number of states to consider for each cell type.
                If None, an optimal K is determined.
        :param weighing: Method to weigh the data, default is 'Omega'.
        :param n_iter: Number of initial cNMF restarts.
        :param n_final_iter: Number of final cNMF restarts.
        :param min_cophenetic: Minimum cophenetic coefficient to determine K.
        :param max_clusters: Maximum number of clusters/states to consider.
        """
        if not self.isRefinementDone:
            raise Exception("Refinement must be completed before StateDiscovery.")

        if celltype is None:
            celltype = self.Celltypes  # Assume self.Celltypes is populated from initial data processing
        if isinstance(celltype, str):
            celltype = [celltype]  # Ensure celltype is iterable

        if K is None:
            K = [None] * len(celltype)
        elif isinstance(K, int):
            K = [K] * len(celltype)  # Broadcast K if it's a single integer

        # Check for consistency in the length of lists if explicitly provided
        if len(celltype) != len(K):
            raise ValueError("Mismatch in the lengths of 'celltype' and 'K'.")

        State_dict = {}
        CopheneticCoefficients = {}
        StateScores = pd.DataFrame()
        StateLoadings = pd.DataFrame()

        for ct, k in zip(celltype, K):
            cNMF_model, cophcors =  StateDiscovery_FrameWork(self.GEX[ct], self.Omega[ct], self.Fractions, ct, weighing, k, n_iter, n_final_iter, min_cophenetic, max_clusters, self.Ncores)
            CopheneticCoefficients[ct] = cophcors
            State_dict[ct] = cNMF_model
            StateScores = pd.concat([StateScores, pd.DataFrame(State_dict[ct].H.transpose(), index=self.Samples).add_prefix(ct+'_')], axis=1)
            StateLoadings = pd.concat([StateLoadings, pd.DataFrame(State_dict[ct].W, index=self.Genes).add_prefix(ct+'_')], axis=1)

        self.cNMF = State_dict
        self.CopheneticCoefficients = CopheneticCoefficients
        self.StateScores = StateScores
        self.StateLoadings = StateLoadings
        self.isStateDiscoveryDone = True
        print("StateDiscovery completed successfully.")

#-------------------------------------------------------------------------------
# 1.2  Define Statescope Initialization
#-------------------------------------------------------------------------------
def Initialize_Statescope(Bulk, Signature=None, TumorType='', Ncelltypes='', MarkerList=None, celltype_key='celltype', n_highly_variable=3000, Ncores=10):
    """ 
    Initializes Statescope object with Bulk and (pre-defined) Signature.

    :param pandas.DataFrame Bulk: Bulk Gene expression matrix: linear, library-size-corrected counts are expected.
    :param pandas.DataFrame or ad.AnnData or None Signature: Cell type specific gene expression matrix.
    :param str TumorType: Tumor type to select predefined signature.
    :param str Ncelltypes: Number of cell types in the signature.
    :param list MarkerList: Predefined list of markers to use for deconvolution.
    :param str celltype_key: Key to use for cell type in AnnData.
    :param int n_highly_variable: Number of hvgs to select for AutoGeneS marker detection.
    :param int Ncores: Number of cores to use for parallel computing.

    :returns: Statescope object initialized with the given parameters.
    """
    available_signatures = list_available_signatures()  # Fetch the structured list of available tumor types and cell types

    if Signature:
        if isinstance(Signature, pd.DataFrame):
            Check_Signature_validity(Signature)
        elif isinstance(Signature, ad.AnnData):
            Signature = CreateSignature(Signature, celltype_key=celltype_key)
    else:
        if TumorType == '' or TumorType not in available_signatures:
            error_msg = "TumorType not specified or invalid. Available options include:\n"
            for t, cells in available_signatures.items():
                error_msg += f"{t}: {', '.join(cells)} cell types\n"
            raise ValueError(error_msg)
        
        if Ncelltypes == '':
            # Select the signature with the smallest number of cell types if Ncelltypes is not specified
            Ncelltypes = min(available_signatures[TumorType], key=int)
        
        Signature = fetch_signature(TumorType, Ncelltypes)
        if Signature is None:
            error_msg = f"No signature available for {TumorType} with {Ncelltypes} cell types. Available cell types for {TumorType} are:\n"
            error_msg += ', '.join(available_signatures[TumorType])
            raise ValueError(error_msg)
    
    # Continue with the initialization as before
    Samples = Bulk.columns.tolist()
    Celltypes = [col.split('scExp_')[1] for col in Signature.columns if 'scExp_' in col]
    Genes = [gene for gene in Bulk.index if gene in Signature.index]

    Signature = Signature.loc[Genes, :]
    Bulk = Bulk.loc[Genes, :]
    Bulk = Check_Bulk_Format(Bulk)
    Markers = Signature[Signature.IsMarker].index.tolist()
    if MarkerList:
        Markers = [gene for gene in Genes if gene in MarkerList]

    Omega_columns = ['scVar_' + ct for ct in Celltypes]
    Mu_columns = ['scExp_' + ct for ct in Celltypes]

    # Print the number of common markers
    common_markers = set(Markers).intersection(Bulk.index)
    print(f"Number of common markers between Bulk and Signature: {len(common_markers)}")

    # Print the number of genes common between Bulk and Signature
    common_genes = set(Genes).intersection(Signature.index)
    print(f"Number of genes common between Bulk and Signature: {len(common_genes)}")


    Statescope_object = Statescope(Bulk, Signature[Mu_columns], Signature[Omega_columns], Samples, Celltypes, Genes, Markers, Ncores)
    return Statescope_object


#-------------------------------------------------------------------------------
# 1.2  Miscellaneous functions
#-------------------------------------------------------------------------------
# Extract Gene expression matrix
def Extract_GEX(Statescope_model, celltype):
    """
    Extracts the purified gene expression matrix (GEX) for a specified cell type from a Statescope model.

    :param Statescope_model: The Statescope object containing refined gene expression data.
    :param str celltype: The cell type for which the GEX is to be extracted.

    :returns: pandas.DataFrame containing the purified gene expression matrix for the specified cell type,
              with sample names as rows and gene names as columns.
    :raises KeyError: If the specified cell type is not found in the Statescope model's GEX dictionary.
    :raises AttributeError: If the Statescope model does not have a GEX attribute or refinement is incomplete.
    """
    if not hasattr(Statescope_model, 'GEX'):
        raise AttributeError("The Statescope model does not contain gene expression data. Ensure refinement has been completed.")
    
    if not Statescope_model.isRefinementDone:
        raise Exception("Refinement must be completed before extracting gene expression data.")

    if celltype in Statescope_model.GEX:
        # Extract the GEX DataFrame for the specified cell type
        gex_matrix = Statescope_model.GEX[celltype]

        # Ensure the DataFrame retains the sample and gene names
        gex_matrix.index = Statescope_model.Samples
        gex_matrix.columns = Statescope_model.Genes
        gex_matrix.index.name = 'Samples'
        gex_matrix.columns.name = 'Genes'
        return gex_matrix
    else:
        raise KeyError(f"Cell type '{celltype}' not found in the Statescope model. Available cell types: {list(Statescope_model.GEX.keys())}")

def Extract_StateScores(Statescope_model):
    """
    Extracts the state scores from a Statescope model after StateDiscovery has been performed.

    :param Statescope_model: The Statescope object containing state discovery results.
    
    :returns: pandas.DataFrame containing state scores for all samples and cell types.
    :raises AttributeError: If StateDiscovery has not been completed or the StateScores attribute is missing.
    """
    # Check if StateDiscovery has been completed
    if not hasattr(Statescope_model, 'StateScores') or Statescope_model.StateScores is None:
        raise AttributeError("StateScores are not available. Please ensure that StateDiscovery has been completed.")

    # Extract the StateScores DataFrame
    state_scores = Statescope_model.StateScores

    # Verify the DataFrame is not empty
    if state_scores.empty:
        raise ValueError("StateScores DataFrame is empty. Check if StateDiscovery was executed correctly.")

    # Return the state scores
    return state_scores

def Extract_StateLoadings(Statescope_model):
    """
    Extracts the StateLoadings matrix for all cell types from a Statescope model.

    :param Statescope_model: The Statescope object containing StateLoadings.
    
    :returns: pandas.DataFrame containing the state loadings with appropriate row (genes) 
              and column (states) names.
    :raises AttributeError: If the Statescope model does not have StateLoadings.
    """
    if not hasattr(Statescope_model, 'StateLoadings') or Statescope_model.StateLoadings is None:
        raise AttributeError("The Statescope model does not contain StateLoadings. Please ensure that StateDiscovery has been completed.")

    state_loadings = Statescope_model.StateLoadings

    # Check if the row and column names are retained
    if state_loadings.empty:
        raise ValueError("The StateLoadings DataFrame is empty. Ensure StateDiscovery has produced valid results.")

    print(f"StateLoadings matrix extracted successfully. Shape: {state_loadings.shape}")
    return state_loadings

def Create_Cluster_Matrix(GEX, Omega, Fractions, celltype, weighing='Omega'):
    """
    Create a scaled matrix for clustering.
    """
    if weighing == 'Omega':
        Cluster_matrix = ((GEX - np.mean(GEX, axis=0)) * Omega.loc[:, celltype].transpose()).to_numpy()
    elif weighing == 'OmegaFractions':
        Cluster_matrix = ((GEX - np.mean(GEX, axis=0)) * Omega.loc[:, celltype].transpose())
        Cluster_matrix = Cluster_matrix.mul(Fractions[celltype], axis=0).to_numpy()
    elif weighing == 'centering':
        Cluster_matrix = (GEX - np.mean(GEX, axis=0)).to_numpy()
    elif weighing == 'no_weighing':
        Cluster_matrix = GEX.to_numpy()
    else:
        raise ValueError("Invalid weighing method.")

    return Cluster_matrix


# Function to check Bulk format
def Check_Bulk_Format(Bulk):
    # Check that Bulk is a DataFrame and has the expected structure
    if not isinstance(Bulk, pd.DataFrame):
        raise ValueError("Bulk should be a pandas DataFrame.")
    if Bulk.empty:
        raise ValueError("Bulk DataFrame is empty.")
    if Bulk.ndim != 2:
        raise ValueError("Bulk DataFrame is not two-dimensional.")
    if Bulk.index.duplicated().any():
        raise ValueError("Bulk DataFrame contains duplicate genes in index.")
    if Bulk.columns.duplicated().any():
        raise ValueError("Bulk DataFrame contains duplicate sample names in column index.")
    # Original checks and operations
    if np.mean(Bulk > 10).any():
        print('The supplied Bulk matrix is assumed to be raw counts. Library size correction to 10k counts per sample is performed.')
        Bulk = Bulk.apply(lambda x: x / sum(x) * 10000, axis=0)
    elif (Bulk < 0).any().any():
        raise AssertionError('Bulk contains negative values. Library size corrected linear counts are required.')

    return Bulk

# Function to check if custom Signature is valid
def Check_Signature_validity(Signature):
    if isinstance(Signature, pd.DataFrame):
        if not 'IsMarker' in Signature.columns:
            raise AssertionError('IsMarker column is missing in Signature')

# Function to check if Expectation (prior knowledge of fractions) is valid
def Check_Expectation_validity(Expectation):
    if Expectation == None:
        print('No prior knowledge of expected cell type fractions is given.')
    else:
        if (Expectation == 0).any():
            raise ValueError('The Expectation contains 0 which is not allowed. Consider giving a very small value (0.01)')
        if (Expectation == 1).any():
            raise ValueError('The Expectation contains 1 which is not allowed. Consider giving a very large value (0.99)')

        
def fetch_signature(tumor_type, n_celltypes):
    """Fetches the signature file directly from GitHub based on tumor type and number of cell types."""
    file_name = f"{tumor_type}_Signature_{n_celltypes}celltypes.txt"
    file_url = f"https://raw.githubusercontent.com/tgac-vumc/StatescopeData/master/{tumor_type}/{file_name}"
    try:
        response = requests.get(file_url)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text), sep='\t', index_col='Gene')
    except requests.HTTPError:
        return None

#####UPDATE the token after 31/12/2025


def list_available_signatures():
    """Lists the availaible signatures in the StatescopeData repository"""
    base_url = "https://api.github.com/repos/tgac-vumc/StatescopeData/contents/"
    
    # Try without the token first
    response = requests.get(base_url)
    data = response.json()

    if 'message' in data and 'API rate limit exceeded' in data['message']:
        # If rate limit exceeded, try using the environment token
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            print("API rate limit exceeded. No environment token found.")
            print("Learn how to create and set a GitHub token: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token")
            return {}
        
        # Use the token from environment
        headers = {'Authorization': f'token {token}'}
        response = requests.get(base_url, headers=headers)
        data = response.json()
        if 'message' in data and 'API rate limit exceeded' in data['message']:
            print("API rate limit exceeded, even using the environment GitHub token.")
            print("Learn how to create and set a GitHub token: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token")
            return {}

    available_signatures = {}
    if isinstance(data, list):
        for folder in data:
            if 'type' in folder and folder['type'] == 'dir':
                tumor_type = folder['name']
                files_url = folder['url']
                files_response = requests.get(files_url, headers=headers if 'Authorization' in locals() else {})
                files_data = files_response.json()
                cell_types = [file['name'].split('_')[-1].replace('celltypes.txt', '') for file in files_data if 'Signature' in file['name']]
                available_signatures[tumor_type] = cell_types
    else:
        print("Unexpected data structure received from GitHub API:", data)
    return available_signatures




#-------------------------------------------------------------------------------
# 1.2  Define Statescope plotting functions
#-------------------------------------------------------------------------------

def generate_color_map(state_columns):
    """
    Generates a consistent color map for cell types and their states with distinct colors
    and subtle shade variations for states.

    :param state_columns: List of state column names (e.g., ['T_cells_CD8+_0', 'T_cells_CD4+_1', 'NK_cells_0']).
    :returns: A dictionary mapping column names to colors.
    """
    # Extract unique cell types using the full prefix before the last underscore
    unique_cell_types = sorted(set(['_'.join(col.split('_')[:-1]) for col in state_columns]))
    color_map = {}

    # Generate a large base color palette for unique cell types
    base_palette = sns.color_palette("husl", len(unique_cell_types))  # Distinct colors
    cell_type_colors = {cell_type: base_palette[idx] for idx, cell_type in enumerate(unique_cell_types)}

    for cell_type in unique_cell_types:
        # Get the base color for the cell type
        base_color = cell_type_colors[cell_type]

        # Generate distinct shades for states within the cell type
        states = [col for col in state_columns if '_'.join(col.split('_')[:-1]) == cell_type]
        shades = sns.light_palette(base_color, n_colors=len(states), reverse=False, input="rgb")

        # Assign each state a shade
        for i, state in enumerate(states):
            color_map[state] = shades[i]

    return color_map


def Heatmap_Fractions(Statescope_model):
    """
    Visualizes the cell type fractions per sample using a clustered heatmap.
    
    :param Statescope_model: An instance of the Statescope class with Fractions data.
    """
    if not hasattr(Statescope_model, 'Fractions') or Statescope_model.Fractions is None:
        raise ValueError("The Statescope model does not have Fractions data. Ensure Deconvolution has been run.")
    if Statescope_model.Fractions.empty:
        raise ValueError("The Fractions DataFrame is empty. Check data initialization and deconvolution results.")

    # Retrieve the fractions data
    fractions = Statescope_model.Fractions

    # Generate a clustermap
    g = sns.clustermap(fractions, annot=True, fmt=".2f", cmap="coolwarm",
                       figsize=(12, 8), cbar_kws={'label': 'Fraction'},
                       method='average')  # method can be 'single', 'complete', 'average', etc.

    # Enhance the plot aesthetics
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)  # Rotate the y-axis labels for better readability
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90) # Rotate the x-axis labels for better readability

    plt.title('Cell Type Fractions per Sample')
    plt.ylabel('Sample')
    plt.xlabel('Cell Type')

    # Show the plot
    plt.show()

def Heatmap_GEX(Statescope_model, celltype):
    """
    Plots a clustered heatmap of the purified gene expression matrix for a specific cell type,
    with no gene labels on the x-axis and an x-axis title.

    :param Statescope_model: The Statescope object containing refined gene expression data.
    :param str celltype: The cell type for which the heatmap is to be plotted.

    :raises KeyError: If the specified cell type is not found in the Statescope model's GEX dictionary.
    :raises AttributeError: If the Statescope model does not have a GEX attribute or refinement is incomplete.
    """
    # Extract the GEX matrix
    gex_matrix = Extract_GEX(Statescope_model, celltype)

    # Replace gene names (column names) with numerical indexes for the x-axis
    gex_matrix.columns = range(len(gex_matrix.columns))

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    heatmap = sns.clustermap(
        gex_matrix,
        cmap="coolwarm",
        annot=False,
        figsize=(12, 10),
        xticklabels=False,  # Completely remove x-axis tick labels
        yticklabels=gex_matrix.index,   # Retain original y-axis labels
        cbar_kws={'label': 'Expression Level'}
    )

    # Add x-axis title to the heatmap
    heatmap.ax_heatmap.set_xlabel("Genes", fontsize=12, labelpad=20)

    plt.title(f"Clustered Heatmap of Gene Expression for Cell Type: {celltype}", pad=20)
    plt.tight_layout()
    plt.show()


def Heatmap_StateScores(Statescope_model):
    """
    Visualize the state scores matrix as a heatmap with states aligned with their respective cell types.
    """
    # Extract the state scores
    state_scores = Extract_StateScores(Statescope_model)

    # Generate color map for columns
    color_map = generate_color_map(state_scores.columns)

    # Separate cell types and states for labels
    cell_states = state_scores.columns
    cell_types = ['_'.join(col.split('_')[:-1]) for col in cell_states]  # Full cell type names
    state_numbers = [col.split('_')[-1] for col in cell_states]

    # Create a heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(
        state_scores,
        cmap="coolwarm",
        cbar_kws={"label": "State Scores"},
        xticklabels=False,  # Turn off default x-tick labels
        yticklabels=True,
        linewidths=0.5
    )

    # Access the axes
    ax = plt.gca()

    # Add state numbers directly on top of color blocks
    for x, state in enumerate(state_numbers):
        color = color_map[state_scores.columns[x]]  # Get the color for the current state
        ax.text(
            x + 0.5, len(state_scores) + 0.5,  # Position above the heatmap
            state, fontsize=8, ha='center', va='center', color='black',
            bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.3', alpha=0.8)
        )

    # Group cell types and display each name once
    xticks = []
    xticklabels = []
    prev_cell_type = None
    for idx, (col, cell_type) in enumerate(zip(cell_states, cell_types)):
        if cell_type != prev_cell_type:
            xticks.append(idx + 0.5)
            xticklabels.append(cell_type)
            prev_cell_type = cell_type

    # Set x-axis ticks and labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=10, rotation=90, ha='center')
    ax.tick_params(axis='x', pad=25)  # Adjust the padding to push tick labels further down

    # Remove the x-axis title
    ax.set_xlabel("")

    # Set the title
    ax.set_title("State Scores Heatmap", fontsize=14)
    plt.tight_layout()
    plt.show()

def Heatmap_StateLoadings(Statescope_model, top_genes=None):
    """
    Visualize the state loadings matrix as a heatmap with states aligned with their respective cell types.
    """
    # Extract the state loadings
    state_loadings = Extract_StateLoadings(Statescope_model)

    if top_genes:
        # Select top genes by maximum loading
        top_genes_list = state_loadings.abs().max(axis=1).nlargest(top_genes).index
        state_loadings = state_loadings.loc[top_genes_list]

    # Generate color map for columns
    color_map = generate_color_map(state_loadings.columns)

    # Separate cell types and states for labels
    cell_states = state_loadings.columns
    cell_types = ['_'.join(col.split('_')[:-1]) for col in cell_states]  # Full cell type names
    state_numbers = [col.split('_')[-1] for col in cell_states]

    # Create a heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(
        state_loadings,
        cmap="coolwarm",
        cbar_kws={"label": "State Loadings"},
        xticklabels=False,  # Turn off default x-tick labels
        yticklabels=True,
        linewidths=0.5
    )

    # Access the axes
    ax = plt.gca()

    # Add state numbers directly on top of color blocks
    for x, state in enumerate(state_numbers):
        color = color_map[state_loadings.columns[x]]  # Get the color for the current state
        ax.text(
            x + 0.5, len(state_loadings) + 0.2,  # Position closer to the heatmap
            state, fontsize=8, ha='center', va='bottom', color='black',
            bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.2', alpha=0.8)
        )

    # Group cell types and display each name once
    xticks = []
    xticklabels = []
    prev_cell_type = None
    for idx, (col, cell_type) in enumerate(zip(cell_states, cell_types)):
        if cell_type != prev_cell_type:
            xticks.append(idx + 0.5)
            xticklabels.append(cell_type)
            prev_cell_type = cell_type

    # Set x-axis ticks and labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=10, rotation=90, ha='center')
    ax.tick_params(axis='x', pad=15)  # Adjust the padding to push tick labels further down

    # Adjust layout to ensure proper spacing
    plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust the figure bounding box

    # Set the title
    ax.set_title("State Loadings Heatmap", fontsize=14)
    plt.show()


def Plot_CopheneticCoefficients(Statescope_model):
    """
    Create a line plot for cophenetic coefficients of all cell types for different values of k.
    The red dot indiciates the chosen model.

    :param Statescope_model: The Statescope object containing CopheneticCoefficients.
    """
    if not hasattr(Statescope_model, 'CopheneticCoefficients') or Statescope_model.CopheneticCoefficients is None:
        raise AttributeError("CopheneticCoefficients are not available. Please ensure that StateDiscovery has been completed.")

    Plot_data = pd.DataFrame(Statescope_model.CopheneticCoefficients)
    # Fetch k
    Plot_data['k'] = range(2,Plot_data.shape[0]+2)
    # Create long format
    Plot_data = pd.melt(Plot_data, id_vars = 'k',var_name = 'celltype', value_name = 'Cophenetic Coefficient')
    # Fetch chosen models
    Chosen_models = {ct: sum(ct in col for col in Statescope_model.StateScores.columns) for ct in Plot_data.celltype.unique()}
    annotation_list = []
    for k, ct in zip(Plot_data['k'],Plot_data['celltype']):
        if Chosen_models[ct] == k:
            annotation_list.append('Chosen')
        else:
            annotation_list.append('Not Chosen')
    Plot_data['Chosen model'] = annotation_list
    
    
    
    # Create plot
    plot = sns.FacetGrid(Plot_data, col="celltype")
    plot.map(sns.lineplot, "k",'Cophenetic Coefficient', color = 'black')
    plot.map(sns.scatterplot, "k",'Cophenetic Coefficient', 'Chosen model',palette={'Chosen':"red", 'Not Chosen':"black"})
    plt.show()
    
    

    
def BarPlot_StateLoadings(Statescope_model, top_genes=1):
    """
    Create a bar plot for cell types and their states, showing coefficients with top genes labeled.

    :param Statescope_model: The Statescope object containing StateLoadings.
    :param top_genes: Number of top genes to label per state.
    """
    # Extract State Loadings
    state_loadings = Extract_StateLoadings(Statescope_model)

    # Generate color map for states
    color_map = generate_color_map(state_loadings.columns)

    # Prepare data for plotting
    bar_data = []
    for state in state_loadings.columns:
        cell_type = '_'.join(state.split('_')[:-1])
        state_number = int(state.split('_')[-1])  # Convert state number to integer
        # Get top genes for the state
        top_gene_values = state_loadings[state].abs().nlargest(top_genes)
        for gene, coeff in top_gene_values.items():
            bar_data.append((cell_type, state, state_number, coeff, gene))

    # Create a DataFrame for easy sorting and plotting
    bar_df = pd.DataFrame(bar_data, columns=["Cell Type", "State", "State Number", "Coefficient", "Top Gene"])
    bar_df.sort_values(by=["Cell Type", "State Number", "Coefficient"], ascending=[True, True, False], inplace=True)

    # Prepare y-axis labels with cell type names on the first state only
    y_tick_labels = []
    prev_cell_type = None
    for idx, row in bar_df.iterrows():
        if row["Cell Type"] != prev_cell_type:
            y_tick_labels.append(f"{row['Cell Type']} {row['State Number']}")  # Show cell type + state number
            prev_cell_type = row["Cell Type"]
        else:
            y_tick_labels.append(f"{row['State Number']}")  # Show state number only

    # Calculate dynamic font size for state numbers
    font_size = max(10 - top_genes, 5)  # Dynamically reduce font size but keep it readable

    # Plot horizontal bar plot
    plt.figure(figsize=(14, 10))
    bars = plt.barh(
        range(len(bar_df)), bar_df["Coefficient"], 
        color=[color_map[row["State"]] for _, row in bar_df.iterrows()],
        edgecolor='black', alpha=0.8
    )

    # Add gene labels to the right of bars
    for bar, label in zip(bars, bar_df["Top Gene"]):
        plt.text(
            bar.get_width() + 0.01,  # Position slightly beyond the bar
            bar.get_y() + bar.get_height() / 2,
            label,
            ha='left', va='center', fontsize=9, color='black'
        )

    # Add labels and title
    plt.xlabel("Coefficient (Max Loading)", fontsize=12)
    plt.ylabel("States (Grouped by Cell Type)", fontsize=12)
    plt.title(f"State Loadings with Top {top_genes} Genes", fontsize=14)

    # Customize y-axis ticks and labels with dynamic font size
    plt.yticks(ticks=range(len(bar_df)), labels=y_tick_labels, fontsize=font_size)
    
    # Invert y-axis to display states from top to bottom
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def TSNE_AllStates(Statescope_model, weighing='Omega', perplexity=5, n_iter=1000, point_size=80):
    """
    Create a t-SNE plot for visualizing all cell types and states using scaled data.

    :param Statescope_model: The Statescope object containing Fractions, GEX, Omega, and StateScores.
    :param weighing: The weighing method for scaling ('Omega', 'OmegaFractions', 'centering', or 'no_weighing').
    :param perplexity: t-SNE perplexity parameter.
    :param n_iter: Number of iterations for optimization.
    :param point_size: Size of the points in the scatter plot.
    """
    # Extract necessary data
    Fractions = Statescope_model.Fractions  # Cell type fractions
    GEX_all = Statescope_model.GEX          # Purified gene expression
    Omega_all = Statescope_model.Omega      # Gene expression variability
    StateScores = Statescope_model.StateScores  # State scores

    # Prepare data for t-SNE
    all_scaled_data = []
    all_labels = []
    all_states = []

    for celltype, Omega in Omega_all.items():
        if celltype not in GEX_all:
            print(f"Warning: Gene expression data for {celltype} not found. Skipping.")
            continue

        GEX = GEX_all[celltype]

        # Create scaled data for this cell type
        scaled_data = Create_Cluster_Matrix(GEX, Omega, Fractions, celltype, weighing)
        all_scaled_data.append(scaled_data)

        # Label each row with the cell type
        all_labels.extend([celltype] * scaled_data.shape[0])

        # Extract dominant state numbers
        scores = StateScores[[col for col in StateScores.columns if col.startswith(celltype)]]
        dominant_states = scores.idxmax(axis=1).str.split('_').str[-1].astype(int)  # Extract state numbers
        all_states.extend(dominant_states)

    # Combine all scaled data
    combined_data = np.vstack(all_scaled_data)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(combined_data)

    # Create a DataFrame for t-SNE results
    tsne_df = pd.DataFrame(tsne_results, columns=['t-SNE1', 't-SNE2'])
    tsne_df['Cell Type'] = all_labels
    tsne_df['State Number'] = all_states

    # Generate a color map for cell types
    unique_cell_types = tsne_df['Cell Type'].unique()
    color_map = {cell_type: color for cell_type, color in zip(unique_cell_types, sns.color_palette("husl", len(unique_cell_types)))}

    # Plot t-SNE
    plt.figure(figsize=(14, 10))
    for cell_type in unique_cell_types:
        cell_type_df = tsne_df[tsne_df['Cell Type'] == cell_type]
        plt.scatter(
            cell_type_df['t-SNE1'],
            cell_type_df['t-SNE2'],
            label=cell_type,
            c=[color_map[cell_type]] * len(cell_type_df),
            s=point_size,  # Adjust point size
            alpha=0.7,
            edgecolor='k'
        )

    # Add state numbers as labels on the points
    for i in range(len(tsne_df)):
        plt.text(tsne_df['t-SNE1'][i], tsne_df['t-SNE2'][i], tsne_df['State Number'][i],
                 fontsize=6, alpha=0.8, ha='center')

    # Add plot details
    plt.legend(title='Cell Types', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("t-SNE of All Cell Types and States", fontsize=16)
    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.tight_layout()
    plt.show()

def TSNE_CellTypes(Statescope_model, celltype=None, weighing='Omega', perplexity=5, n_iter=1000, point_size=150):
    """
    Create t-SNE plots for a single cell type or for all cell types, displayed individually.

    :param Statescope_model: The Statescope object containing Fractions, GEX, Omega, and StateScores.
    :param celltype: The specific cell type to visualize (default: None, meaning all cell types).
    :param weighing: The weighing method for scaling ('Omega', 'OmegaFractions', 'centering', or 'no_weighing').
    :param perplexity: t-SNE perplexity parameter.
    :param n_iter: Number of iterations for optimization.
    :param point_size: Size of the points in the scatter plot.
    """
    Omega_all = Statescope_model.Omega
    GEX_all = Statescope_model.GEX
    StateScores = Statescope_model.StateScores
    Fractions = Statescope_model.Fractions

    # Generate consistent colors for all cell types
    unique_cell_types = Omega_all.keys()
    color_map = {cell_type: color for cell_type, color in zip(unique_cell_types, sns.color_palette("husl", len(unique_cell_types)))}

    # Determine which cell types to process
    celltypes_to_plot = [celltype] if celltype else list(Omega_all.keys())

    # Loop through each cell type and generate t-SNE
    for celltype in celltypes_to_plot:
        if celltype not in GEX_all or celltype not in Omega_all:
            print(f"Data for cell type {celltype} not found. Skipping.")
            continue

        GEX = GEX_all[celltype]
        Omega = Omega_all[celltype]

        # Create scaled data for this cell type
        scaled_data = Create_Cluster_Matrix(GEX, Omega, Fractions, celltype, weighing)

        # Extract dominant state numbers
        scores = StateScores[[col for col in StateScores.columns if col.startswith(celltype)]]
        dominant_states = scores.idxmax(axis=1).str.split('_').str[-1].astype(int)

        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=n_iter)
        tsne_results = tsne.fit_transform(scaled_data)

        # Create a DataFrame for t-SNE results
        tsne_df = pd.DataFrame(tsne_results, columns=['t-SNE1', 't-SNE2'])
        tsne_df['State Number'] = dominant_states.reset_index(drop=True)

        # Plot t-SNE
        plt.figure(figsize=(10, 8))
        plt.scatter(
            tsne_df['t-SNE1'],
            tsne_df['t-SNE2'],
            label=celltype,
            c=[color_map[celltype]] * len(tsne_df),
            s=point_size,
            alpha=0.7,
            edgecolor='k'
        )

        # Add state numbers as labels on the points
        for i in range(len(tsne_df)):
            plt.text(tsne_df['t-SNE1'][i], tsne_df['t-SNE2'][i], str(tsne_df['State Number'][i]),
                     fontsize=6, alpha=0.8, ha='center')

        # Add plot details
        plt.legend(title='Cell Type', loc='upper left')
        plt.title(f"t-SNE for {celltype} and States", fontsize=16)
        plt.xlabel("t-SNE1")
        plt.ylabel("t-SNE2")
        plt.tight_layout()
        plt.show()



