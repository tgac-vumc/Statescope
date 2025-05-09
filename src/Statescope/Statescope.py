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
# 1) Testing by Jurrian 
#
# History:
#  13-12-2024: File creation, write code, test Deconvolution and Refinement
#  14-12-2024: Finish StateDiscovery testing
#  20-01-2025: Additional checks, Utility and Visualization Functions 
#  30-04-2025: New parameters, critical functions fixed and tested
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
        scExp_celltypes = [ct.replace("scExp_", "") for ct in self.scExp.columns]
        scVar_celltypes = [ct.replace("scVar_", "") for ct in self.scVar.columns]

        Expectation = Check_Expectation_validity(
            Expectation,
            celltype_order=scExp_celltypes,
            sample_names=self.Samples
        )

        
        # Excecute BLADE Deconvolution: FrameWork iterative
        final_obj, best_obj, best_set, outs = Framework_Iterative(scExp_marker, scVar_marker, Y, Ind_Marker,
                        Alpha, Alpha0, Kappa0, sY,
                        Nrep, self.Ncores, fsel, Update_SigmaY, Init_Trust,
                        Expectation = Expectation, Temperature = Temperature, IterMax = IterMax)
        # Save BLADE result in Statescope object
        self.BLADE = final_obj
        # Save fractions as dataframe in object
        self.Fractions = pd.DataFrame(final_obj.ExpF(final_obj.Beta).cpu().numpy(), index=self.Samples, columns=self.Celltypes)

        ###This one is for old oncoblade application 
        #self.Fractions =  pd.DataFrame(final_obj.ExpF(final_obj.Beta), index=self.Samples, columns=self.Celltypes)
        self.isDeconvolutionDone = True
        print("Deconvolution completed successfully.")
    
        # Check convergence flag and print message accordingly
        # Convert to a Python boolean if needed
        if self.BLADE.log:
            print("Model converged.")
        else:
            print("Warning: Model did not converge, estimates might not be optimal.")


        
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


    def StateDiscovery(
            self,
            celltype: str | list[str] | None = None,
            K: int | list[int] | dict[str, int] | None = None,
            weighing: str = 'Omega',
            n_iter: int = 10,
            n_final_iter: int = 100,
            min_cophenetic: float = 0.9,
            max_clusters: int = 10):
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
        :param celltype : str | list[str] | None, optional
        Cell types to analyse (default = all stored in the object).

        :param K : int | list[int] | dict[str, int] | None, optional
            • None   – each cell type gets an automatically chosen k  
            • int    – the same k for every cell type (celltype filter ignored)  
            • list   – one k per entry in `celltype` (same order)  
            • dict   – give k for selected cell types; the rest are automatic

        **Examples**

        >>> Model.StateDiscovery(K=2)
        Forces k = 2 for *every* cell type.

        >>> Model.StateDiscovery(celltype='Monocyte')
        Analyses only Monocytes; k picked automatically.

        >>> Model.StateDiscovery(K={'Monocyte': 2})
        Monocytes fixed at k = 2; every other cell type analysed with
        automatic k.

        >>> Model.StateDiscovery(celltype=['T', 'B'], K=[4, 5])
        T cells get k = 4, B cells k = 5; no other cell types analysed.
                
        """
        
        # 0) checks                                         
        
        if not self.isRefinementDone:
            raise RuntimeError("Run Refinement before StateDiscovery.")

        all_celltypes = list(self.Celltypes)      # stored when the object was built

        
        # 1) normalise `celltype` argument                             
        
        if isinstance(K, int):
            # K is a single integer → analyse *all* cell types,
            # ignoring any `celltype` argument the caller provided.
            celltype_run = all_celltypes
        else:
            # otherwise use the caller's list, defaulting to all
            if celltype is None:
                celltype_run = all_celltypes
            elif isinstance(celltype, str):
                celltype_run = [celltype]
            else:
                celltype_run = list(celltype)

            # if K is a dict, include those keys too
            if isinstance(K, dict):
                celltype_run = list(K.keys())

        
        # 2) build K-mapping {celltype: k or None}                     
       
        if K is None:
            Kmap = {ct: None for ct in celltype_run}

        elif isinstance(K, int):
            Kmap = {ct: K for ct in celltype_run}

        elif isinstance(K, list):
            if len(K) != len(celltype_run):
                raise ValueError("Length of K list must equal number of cell types.")
            Kmap = dict(zip(celltype_run, K))

        elif isinstance(K, dict):
            Kmap = K

        else:
            raise TypeError("K must be int, list, dict, or None.")

        
        # 3) run cNMF / state discovery per cell type                
        
        State_dict, CopheneticCoefficients = {}, {}
        StateScores, StateLoadings = {}, {}

        for ct in celltype_run:
            print('Performing cNMF State Discovery for {ct}')
            model, coph = StateDiscovery_FrameWork(
                self.GEX[ct],
                self.Omega[ct],
                self.Fractions,
                ct,
                weighing,
                Kmap[ct],                 # may be None → auto
                n_iter,
                n_final_iter,
                min_cophenetic,
                max_clusters,
                self.Ncores,
            )

            State_dict[ct]             = model
            CopheneticCoefficients[ct] = coph
            StateScores[ct] =  pd.DataFrame(np.apply_along_axis(lambda x: x/ sum(x),1,model.H.T), index=self.Samples).add_prefix(f"{ct}_")
            StateLoadings[ct] = pd.DataFrame(model.W, index=self.Genes).add_prefix(f"{ct}_")
            

        
        # 4) stash results in the object                               
        if not hasattr(self, 'isStateDiscoveryDone '):
            self.cNMF                   = State_dict
            self.CopheneticCoefficients = CopheneticCoefficients
            self.StateScores            = StateScores
            self.StateLoadings          = StateLoadings
            self.isStateDiscoveryDone   = True
        else:
            self.cNMF.update(State_dict)
            self.CopheneticCoefficients.update(CopheneticCoefficients)
            self.StateScores.update(StateScores)
            self.StateLoadings.update(StateLoadings)
            
        print("StateDiscovery completed successfully.")



#-------------------------------------------------------------------------------
# 1.2  Define Statescope Initialization
#-------------------------------------------------------------------------------
def Initialize_Statescope(Bulk, Signature=None, TumorType='', Ncelltypes='', MarkerList=None, celltype_key='celltype', n_highly_variable=3000, Ncores=10, fixed_n_features=None, drop_sigdiff = False):
    """ 
    Initializes Statescope object with Bulk and Signature.

    :param pandas.DataFrame Bulk: Bulk Gene expression matrix: linear, library-size-corrected counts are expected.
    :param pandas.DataFrame or ad.AnnData or None Signature: Cell type specific gene expression matrix.
    :param str TumorType: Tumor type to select predefined signature.
    :param str Ncelltypes: Number of cell types in the signature.
    :param list MarkerList: Predefined list of markers to use for deconvolution.
    :param str celltype_key: Key to use for cell type in AnnData.
    :param int n_highly_variable: Number of hvgs to select for AutoGeneS marker detection, if set as None default cutoffs will be used
    :param int Ncores: Number of cores to use for parallel computing.
    :param int fixed_n_features: None will allow autogenes to determine the number of genes, selecting for eg 500 will give 500 marker genes to be used for deconvolution

    :returns: Statescope object initialized with the given parameters.
    """
    available_signatures = list_available_signatures()  # Fetch the structured list of available tumor types and cell types
  #subset Markers if supplied before creating signature 
    if Signature is not None:
        if isinstance(Signature, pd.DataFrame):
            Check_Signature_validity(Signature)
        elif isinstance(Signature, ad.AnnData):
             Signature = CreateSignature(
                Signature,
                celltype_key=celltype_key,
                CorrectVariance=True,
                n_highly_variable=n_highly_variable, #hvg genes for autogenes parameter
                fixed_n_features=fixed_n_features, # autogene number of genes paramter 
                MarkerList=MarkerList,          #will use marker list instead of autogenes 
                Bulk = Bulk,
                drop_sigdiff= drop_sigdiff)      # if bulk and drop_sigdiff are true will calculate the and remove genes that differ significantly in expression between the two datasets
            
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
    
    if MarkerList:
        Signature['IsMarker'] = Signature.index.isin(MarkerList)

    # Continue with the initialization as before
    Samples = Bulk.columns.tolist()
    Celltypes = [col.split('scExp_')[1] for col in Signature.columns if 'scExp_' in col]
    Genes = [gene for gene in Bulk.index if gene in Signature.index]

    Signature = Signature.loc[Genes, :]
    Bulk = Bulk.loc[Genes, :]
    Bulk = Check_Bulk_Format(Bulk)
    Markers = Signature[Signature.IsMarker].index.tolist()

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

def Extract_StateScores(Statescope_model, celltype = None):
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
    if celltype == None:
        state_loadings = pd.concat(Statescope_model.StateScores.values(), axis=1)
    else:
        state_loadings = Statescope_model.StateScores[celltype]

    # Verify the DataFrame is not empty
    if state_scores.empty:
        raise ValueError("StateScores DataFrame is empty. Check if StateDiscovery was executed correctly.")

    # Return the state scores
    return state_scores

def Extract_StateLoadings(Statescope_model, celltype = None):
    """
    Extracts the StateLoadings matrix for all cell types (or specified celltype) from a Statescope model.

    :param Statescope_model: The Statescope object containing StateLoadings.
    
    :returns: pandas.DataFrame containing the state loadings with appropriate row (genes) 
              and column (states) names.
    :raises AttributeError: If the Statescope model does not have StateLoadings.
    """
    if not hasattr(Statescope_model, 'StateLoadings') or Statescope_model.StateLoadings is None:
        raise AttributeError("The Statescope model does not contain StateLoadings. Please ensure that StateDiscovery has been completed.")
    if celltype == None:
        state_loadings = pd.concat(Statescope_model.StateLoadings.values(), axis=1)
    else:
        state_loadings = Statescope_model.StateLoadings[celltype]

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



def Check_Expectation_validity(Expectation, celltype_order=None, sample_names=None):
    if Expectation is None:
        print('No prior knowledge of expected cell type fractions is given.')
        return None

    if isinstance(Expectation, dict):
        group_mat = Expectation.get("Group")
        group_expect = Expectation.get("Expectation")

        if group_mat is None or group_expect is None:
            raise ValueError("Both 'Group' and 'Expectation' must be keys in the dictionary.")

        if not isinstance(group_mat, np.ndarray) or not isinstance(group_expect, np.ndarray):
            raise ValueError("Group and Expectation must be numpy arrays in the dictionary format.")

        if group_expect.shape[1] != group_mat.shape[0]:
            raise ValueError(f"Shape mismatch: Expectation has {group_expect.shape[1]} groups, "
                             f"but Group matrix has {group_mat.shape[0]} rows.")

        temp = np.matmul(group_expect, group_mat)

        if (temp == 0).any():
            raise ValueError("The Expectation contains 0 which is not allowed. Use a small value like 0.01.")

        if (temp == 1).any():
            raise ValueError("The Expectation contains 1 which is not allowed. Use a large value like 0.99.")

        print("Grouped prior knowledge is utilised in refining fraction estimates.")
        return Expectation  

    elif isinstance(Expectation, pd.DataFrame):
        if celltype_order is not None:
            missing_cts = [ct for ct in celltype_order if ct not in Expectation.columns]
            if missing_cts:
                raise ValueError(f"Missing cell types in Expectation: {missing_cts}")
            Expectation = Expectation[celltype_order]

        if (Expectation == 0).any().any():
            raise ValueError("The Expectation contains 0 which is not allowed. Use a small value like 0.01.")

        if (Expectation == 1).any().any():
            raise ValueError("The Expectation contains 1 which is not allowed. Use a large value like 0.99.")

        print("Celltype-level prior knowledge is utilised in refining fraction estimates.")
        return Expectation.to_numpy()

    else:
        raise ValueError("Expectation must be a dict or a pandas DataFrame.")

        
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


def Heatmap_StateScores(
        Statescope_model,
        *,
        col_width: float = 0.35,     # inch per state column
        row_height: float = 0.35,    # inch per sample row
        bottom: float = 0.28,        # extra margin for x-labels (0-1)
        label_pad: int = 40):        # gap between heat-map and x-labels
    """
    Heat-map of state scores with auto-scaled figure size so column labels and
    state numbers never overlap.

    Parameters
    ----------
    Statescope_model : Statescope
        Object returned by StateDiscovery.
    col_width : float, optional
        Desired width (inches) allotted to **each state column**.
    row_height : float, optional
        Desired height (inches) allotted to **each sample row**.
    bottom : float, optional
        Fraction of figure height reserved beneath the heat-map for x-labels.
    label_pad : int, optional
        Padding (points) between heat-map edge and x-labels.
    """
    # ------------------------------------------------------------ #
    # 1) data                                                      #
    # ------------------------------------------------------------ #
    state_scores = Extract_StateScores(Statescope_model)

    cell_states = state_scores.columns
    cell_types  = ['_'.join(c.split('_')[:-1]) for c in cell_states]
    state_nums  = [c.split('_')[-1]            for c in cell_states]

    # ------------------------------------------------------------ #
    # 2) dynamic figure size                                       #
    # ------------------------------------------------------------ #
    n_cols = state_scores.shape[1]
    n_rows = state_scores.shape[0]
    fig_w  = max(8, col_width  * n_cols)      # at least 8 inch wide
    fig_h  = max(6, row_height * n_rows)      # at least 6 inch tall
    plt.figure(figsize=(fig_w, fig_h))

    # ------------------------------------------------------------ #
    # 3) base heat-map                                             #
    # ------------------------------------------------------------ #
    sns.heatmap(
        state_scores,
        cmap="coolwarm",
        cbar_kws={"label": "State Scores"},
        xticklabels=False,
        yticklabels=True,
        linewidths=0.5
    )
    ax = plt.gca()

    # annotate state numbers
    palette = generate_color_map(cell_states)
    for x, num in enumerate(state_nums):
        ax.text(
            x + 0.5, n_rows + 0.5,
            num,
            ha='center',
            va='center',
            fontsize=8,
            bbox=dict(facecolor=palette[cell_states[x]], edgecolor='none',
                      boxstyle='round,pad=0.3', alpha=0.8)
        )

    # x-tick labels: one per cell type
    xticks, labels, prev = [], [], None
    for idx, ctype in enumerate(cell_types):
        if ctype != prev:
            xticks.append(idx + 0.5)
            labels.append(ctype)
            prev = ctype

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=90, ha='center', va='top', fontsize=10)
    ax.tick_params(axis='x', pad=label_pad)
    plt.gcf().subplots_adjust(bottom=bottom)   # reserve space for labels

    ax.set_xlabel("")
    ax.set_title("State Scores Heatmap", fontsize=14)
    plt.show()



def Heatmap_StateLoadings(Statescope_model, top_genes=None):
    """
    Plot the cell-type state-loading matrix as a heat-map.
    Columns (states) are grouped by cell type and labelled with the state
    number on the colour bar; rows are genes.  Optionally restrict the plot to
    the genes with the largest absolute loadings.

    :param Statescope_model:  A fitted Statescope object; must contain the
                              ``StateLoadings`` matrix produced by
                              *StateDiscovery*.
    :param top_genes:         If *None* (default) plot all genes.  
                              If an int *N*, plot only the *N* genes with the
                              highest absolute loading across all states
                              (one global ranking).
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
    # ------------------------------------------------------------ #
    # 0) sanity check                                              #
    # ------------------------------------------------------------ #
    if not getattr(Statescope_model, "CopheneticCoefficients", None):
        raise AttributeError(
            "CopheneticCoefficients are not available. "
            "Run StateDiscovery first."
        )

    # number of states retained for each cell type, derived from StateScores
    chosen_K = {
        ct: len(Statescope_model.StateScores[ct].columns) for ct in Statescope_model.CopheneticCoefficients.keys()
    }

    # ------------------------------------------------------------ #
    # 1) build a tidy data-frame: celltype | k | coefficient | flag #
    # ------------------------------------------------------------ #
    records = []

    for ct, coeff_obj in Statescope_model.CopheneticCoefficients.items():

        # case A: array-like (list / np.ndarray / pd.Series)
        if isinstance(coeff_obj, (list, np.ndarray, pd.Series)):
            ks = np.arange(2, 2 + len(coeff_obj))
            coeffs = coeff_obj

        # case B: dict {k: coefficient}
        elif isinstance(coeff_obj, dict):
            ks, coeffs = zip(*sorted(coeff_obj.items()))

        # case C: single scalar → only the chosen K’s coefficient is known
        else:
            ks = [chosen_K[ct]]
            coeffs = [coeff_obj]

        # add rows
        for k, coef in zip(ks, coeffs):
            records.append(
                dict(celltype=ct,
                     k=int(k),
                     coefficient=coef,
                     chosen=("Chosen" if k == chosen_K[ct] else "Not Chosen"))
            )

    plot_df = pd.DataFrame.from_records(records)

    # ------------------------------------------------------------ #
    # 2) plot with Seaborn                                         #
    # ------------------------------------------------------------ #
    g = sns.FacetGrid(plot_df, col="celltype", sharey=False, height=4, aspect=1.2)

    # lines (only appear when ≥2 points for that cell type)
    g.map_dataframe(
        sns.lineplot,
        x="k",
        y="coefficient",
        color="black"
    )

    # scatter: red = chosen, black = not chosen
    g.map_dataframe(
        sns.scatterplot,
        x="k",
        y="coefficient",
        hue="chosen",
        palette={"Chosen": "red", "Not Chosen": "black"},
        legend=False,
        s=50
    )

    g.set_axis_labels("k (number of states)", "Cophenetic coefficient")
    g.fig.tight_layout()
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




def TSNE_AllStates(
        Statescope_model,
        *,
        weighing: str = "Omega",
        perplexity: int = 5,
        n_iter: int = 1_000,
        point_size: int = 80,
        trim_regex: str = r"[_\|].*$",
        random_state: int = 42,
        show_samples: bool | None = None,   # ← NEW
        sample_fs: int = 6):               # ← font-size when shown
    """
    Run a t-SNE on all samples, colour by cell type, print the dominant state
    number inside each dot.  Sample names are printed above the dots only when
    *show_samples=True*.

    :param Statescope_model: The fitted Statescope object.
    :param weighing:         Scaling used in ``Create_Cluster_Matrix``.
    :param perplexity:       t-SNE perplexity (auto-capped).
    :param n_iter:           Optimisation iterations for t-SNE.
    :param point_size:       Marker size.
    :param trim_regex:       Regex to shorten sample names when shown.
    :param random_state:     Seed for reproducibility.
    :param show_samples:     • None/False (default) → hide sample names.  
                             • True → show sample names above each dot.
    :param sample_fs:        Font size for sample names.
    """
    
    Fractions   = Statescope_model.Fractions
    GEX_all     = Statescope_model.GEX
    Omega_all   = Statescope_model.Omega
    StateScores = Extract_StateScores(Statescope_model)

    matrices, labels, states, samples = [], [], [], []

    # ------------------------------------------------------------------ #
    # 1) build per-cell-type matrices + label lists                      #
    # ------------------------------------------------------------------ #
    for ct, Omega in Omega_all.items():
        if ct not in GEX_all:
            print(f"[t-SNE] '{ct}' GEX missing – skipped.")
            continue

        X = Create_Cluster_Matrix(GEX_all[ct], Omega, Fractions, ct, weighing)
        X = np.nan_to_num(X)                         # remove NaN / inf
        X = X[:, X.var(axis=0) > 0]                 # keep informative cols

        # guarantee ≥2 columns
        if X.shape[1] == 0:
            X = np.zeros((X.shape[0], 2))
        elif X.shape[1] == 1:
            X = np.hstack([X, np.zeros((X.shape[0], 1))])

        matrices.append(X)
        labels.extend([ct] * X.shape[0])

        dom_state = (StateScores.filter(regex=f"^{ct}_")
                                  .idxmax(axis=1)
                                  .str.split('_').str[-1].astype(int))
        states.extend(dom_state)

        trimmed_samples = (dom_state.index
                           .to_series()
                           .str.replace(trim_regex, "", regex=True))
        samples.extend(trimmed_samples)

    if not matrices:
        print("[t-SNE] No usable data.")
        return

    # ------------------------------------------------------------------ #
    # 2) pad matrices to equal n_columns, stack                          #
    # ------------------------------------------------------------------ #
    max_cols = max(m.shape[1] for m in matrices)
    padded   = [np.pad(m, ((0, 0), (0, max_cols - m.shape[1])),
                       mode="constant") for m in matrices]
    combined = np.vstack(padded)

    # ------------------------------------------------------------------ #
    # 3) run t-SNE                                                      #
    # ------------------------------------------------------------------ #
    perp = max(1, min(perplexity, combined.shape[0] - 1))
    tsne = TSNE(n_components=2, init="random",
                perplexity=perp, n_iter=n_iter,
                random_state=random_state)
    emb = tsne.fit_transform(combined)

    df = pd.DataFrame(emb, columns=["t-SNE1", "t-SNE2"])
    df["Cell"]   = labels
    df["State"]  = states
    df["Sample"] = samples

    # 4) plot ------------------------------------------------------------
    palette   = sns.color_palette("husl", df["Cell"].nunique())
    ct_colour = dict(zip(df["Cell"].unique(), palette))

    plt.figure(figsize=(12, 9))
    for ct, sub in df.groupby("Cell"):
        plt.scatter(sub["t-SNE1"], sub["t-SNE2"],
                    c=[ct_colour[ct]] * len(sub),
                    s=point_size, alpha=0.7, edgecolor="k",
                    label=ct)

    # state number (always)
    for _, row in df.iterrows():
        plt.text(row["t-SNE1"], row["t-SNE2"],
                 str(row["State"]), fontsize=6,
                 ha="center", va="center", color="white")

    # sample name (optional)
    if show_samples:
        for _, row in df.iterrows():
            plt.text(row["t-SNE1"], row["t-SNE2"] + 0.8,
                     row["Sample"], fontsize=sample_fs,
                     ha="center", va="bottom", alpha=0.9)

    plt.legend(title="Cell Type", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.title("t-SNE of All Cell Types / States", fontsize=15)
    plt.xlabel("t-SNE1"); plt.ylabel("t-SNE2")
    plt.tight_layout()
    plt.show()



def TSNE_CellTypes(
        Statescope_model,
        celltype: str | None = None,
        weighing: str = "Omega",
        perplexity: int = 5,
        n_iter: int = 1_000,
        point_size: int = 150,
        min_samples: int = 2,
        random_state: int = 42,
        *,
        show_samples: bool = True,
        sample_fs: int = 6):
    """
    Generate a t-SNE scatterplot for one cell type (or every cell type) using
    its “cluster matrix”.  
    Each dot is a sample; the dominant state number is printed in the centre
    and, if desired, the sample name is shown just above it.

    :param Statescope_model:  A fitted Statescope object containing ``GEX``,
                              ``Omega``, ``Fractions`` and ``StateScores``.
    :param celltype:          A specific cell type to visualise.  
                              • ``None`` (default) → plot every cell type.  
                              • *str* → plot only the requested cell type.
    :param weighing:          Transformation applied inside
                              ``Create_Cluster_Matrix``  
                              (``'Omega'`` | ``'OmegaFractions'`` |
                              ``'centering'`` | ``'no_weighing'``).
    :param perplexity:        t-SNE perplexity (auto-capped at
                              *n_samples – 1*).
    :param n_iter:            Number of optimisation iterations for t-SNE
                              (default = 1000).
    :param point_size:        Size of the scatter markers.
    :param min_samples:       Minimum samples required for a cell type to be
                              plotted; smaller groups are skipped.
    :param random_state:      Seed for reproducible t-SNE initialisation.
    :param show_samples:      ``True`` prints sample names above the dots;
                              ``False`` omits them.
    :param sample_fs:         Font-size for sample labels when
                              ``show_samples=True``.
    """
    # ── shortcuts ─────────────────────────────────────────────────────
    Omega_all   = Statescope_model.Omega
    GEX_all     = Statescope_model.GEX
    Fractions   = Statescope_model.Fractions
    StateScores = Extract_StateScores(Statescope_model)

    all_cts  = list(Omega_all.keys())
    colors   = sns.color_palette("husl", len(all_cts))
    ct_color = dict(zip(all_cts, colors))

    to_plot = [celltype] if celltype else all_cts

    for ct in to_plot:
        if ct not in GEX_all or ct not in Omega_all:
            print(f"[t-SNE] '{ct}' not found – skipped.")
            continue

        # ── build + clean matrix ──────────────────────────────────────
        mat = Create_Cluster_Matrix(GEX_all[ct], Omega_all[ct],
                                    Fractions, ct, weighing)
        mat = np.nan_to_num(mat)

        var_mask = mat.var(axis=0) > 0
        mat = mat[:, var_mask]

        if mat.shape[0] < min_samples:
            print(f"[t-SNE] '{ct}' has <{min_samples} samples – skipped.")
            continue

        if mat.shape[1] == 0:
            mat = np.zeros((mat.shape[0], 2))
        elif mat.shape[1] == 1:
            mat = np.hstack([mat, np.zeros((mat.shape[0], 1))])

        # ── labels ────────────────────────────────────────────────────
        dom_state = (StateScores.filter(regex=f"^{ct}_")
                                .idxmax(axis=1)
                                .str.split('_').str[-1]
                                .astype(int)
                                .reset_index(drop=True))
        sample_names = StateScores.index.to_series().reset_index(drop=True)

        # ── t-SNE ─────────────────────────────────────────────────────
        perp = max(1, min(perplexity, mat.shape[0] - 1))
        tsne = TSNE(n_components=2, init="random",
                    perplexity=perp, n_iter=n_iter,
                    random_state=random_state)
        emb = tsne.fit_transform(mat)

        df = pd.DataFrame(emb, columns=["t-SNE1", "t-SNE2"])
        df["state"]  = dom_state
        df["sample"] = sample_names

        # ── plot ─────────────────────────────────────────────────────
        plt.figure(figsize=(8, 7))
        plt.scatter(df["t-SNE1"], df["t-SNE2"],
                    c=[ct_color[ct]] * len(df),
                    s=point_size, alpha=0.7, edgecolor="k")

        # state number centred
        for x, y, s in zip(df["t-SNE1"], df["t-SNE2"], df["state"]):
            plt.text(x, y, str(s), fontsize=6,
                     ha='center', va='center', color='white')

        # sample label a bit above
        if show_samples:
            for x, y, smp in zip(df["t-SNE1"], df["t-SNE2"], df["sample"]):
                plt.text(x, y + 0.8, smp, fontsize=sample_fs,
                         ha='center', va='bottom', alpha=0.9)

        plt.title(f"t-SNE for {ct}", fontsize=14)
        plt.xlabel("t-SNE1"); plt.ylabel("t-SNE2")
        plt.tight_layout()
        plt.show()
