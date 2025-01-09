#!/usr/bin/python3
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Statescope.py
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Statescope framework
# Author: Jurriaan Janssen (j.janssen4@amsterdamumc.nl)
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
        self.DeconvolutionDone = True

        
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
        if GeneList:
            self.Genes = [gene for gene in Genes if gene in GeneList]
        # Prepare Signature
        scExp_All = self.scExp.loc[self.Genes, :].to_numpy()
        scVar_All = self.scVar.loc[self.Genes, :].to_numpy()
        # Prepare Bulk (select/match genes
        Y = self.Bulk.loc[self.Genes,self.Samples].to_numpy()
        # Perform gene expression revinement with all genes in signature
        obj = Purify_AllGenes(self.BLADE, scExp_All, scVar_All,Y,self.Ncores, weight)
        # create output GEX dictionary
        GEX = {ct:pd.DataFrame(obj.Nu[:,:,i],index=self.Samples,columns=self.Genes) for i,ct in enumerate(self.Celltypes)}
        Omega = {ct:pd.DataFrame(obj.Omega[:,i],index=self.Genes,columns=[ct]) for i,ct in enumerate(self.Celltypes)}
        # Store in Statescope object
        self.BLADE_final = obj
        self.GEX = GEX
        self.Omega = Omega

    # Perform State Discovery
    def StateDiscovery(self, celltype = [''],
                       K = [None],weighing = 'Omega',n_iter = 10,n_final_iter = 100,min_cophenetic = 0.9,max_clusters=10):
        """ 
        Perform StateDiscovery from ctSpecificGEX using cNMF
        by default all cell types are used and the optimal K is determined
        ctSpecific GEX is by default weighed by Omega

        :param Statescope self Statescope
        :param str celltype: cell type to perform state discovery [default = '' (All)]
        :param int K: number of states to return, by default the optimal K is chosen [None]
        :param str weighing: [default = 'Omega', choices: ['Omega','OmegaFractions','centering','no_weighing']]
        :param int n_iter: Number of initial cNMF restarts [default = 10]
        :param int n_iter_final: Number of final cNMF restarts [default = 100]
        :param float min_cophenetic: minimum cophentic coeffient to determine K [default = 0.9]
        :param int max_cluster: maximum number of states to consider [default = 10]
        :param int Ncores: Number of cores to use for parrallel restarts [default = 10]

        :returns: CNMF_object self.cNMF: cNMF object
        :returns: StateScores self.StateScores: cNMF state scores from all cell types compiled (NSample x (NcellxNState))
        :returns: StateLoadings self.StateLoadings: cNMF state loadings from all cell types compiled (Ngene x (NcellxNState))
        """
        if celltype == '':
            celltype = self.Celltypes
            K = K * len(celltype)
        # by default, perform State Discovery for all celltypes
        if isinstance(celltype, Iterable):
            State_dict = dict()
            # Iterate over celltypes (can be done in parallel)
            for ct,k in zip(celltype,K):
                # save cNMF models in State dict
                State_dict[ct] = StateDiscovery_FrameWork(self.GEX[ct],self.Omega[ct],self.Fractions,ct,weighing,k,n_iter,n_final_iter,min_cophenetic,max_clusters,self.Ncores)
            # save cNMF models as dict
            self.cNMF = State_dict
            # Save State scores and loadings as compile pandas.DataFrame
            StateScores = pd.DataFrame()
            StateLoadings = pd.DataFrame()
            for ct,cNMF in State_dict.items():
                StateScores = pd.concat([StateScores,pd.DataFrame(cNMF.H.transpose(),index = self.Samples).add_prefix(ct+'_')], axis = 1)
                StateLoadings = pd.concat([StateLoadings,pd.DataFrame(cNMF.W,index = self.Genes).add_prefix(ct+'_')], axis = 1)
            self.StateLoadings =  StateLoadings
            self.StateScores = StateScores
        # alternatively, perform State discovery for one cell type
        else:
            # save cNMF models as single cNMF instance
            self.cNMF = StateDiscovery_FrameWork(self.GEX[celltype],self.Omega[celltype],self.Fractions,celltype,weighing,K,n_iter,n_final_iter,min_cophentic,max_clusters,self.Ncores)
            # save State scores and loadings
            self.StateScores = pd.DataFrame(cNMF.H, index = self.Samples).add_prefix(celltype)
            self.StateLoadings =pd.DataFrame(cNMF.W, index = self.Genes).add_prefix(celltype)




#-------------------------------------------------------------------------------
# 1.2  Define Statescope Initialization
#-------------------------------------------------------------------------------
# Function to check Bulk format
def Check_Bulk_Format(Bulk):
    if np.mean(Bulk > 10).any():
        print('The supplied Bulk matrix is assumed to be raw counts. Library size correction to 10k counts per sample is performed')
        Bulk = Bulk.apply(lambda x: x/sum(x)*10000,axis=0)
    elif (Bulk < 0).any().any():
        raise AssertionError('Bulk contains negative values. Library size corrected linear counts are required')
        
    return Bulk
        
    
# Function to check if custom Signature is valid
def Check_Signature_validity(Signature):
    if isinstance(Signature, pd.DataFrame):
        if not 'IsMarker' in Signature.columns:
            raise AssertionError('IsMarker column is missing in Signature')
        

def Initialize_Statescope(Bulk, Signature=None,TumorType='',Ncelltypes='',MarkerList = None,celltype_key = 'celltype',n_highly_variable = 3000, Ncores = 10):
    """ 
        Intialized Statescope object with Bulk and (pre-defined) Signature

        :param Statescope self Statescope
        :param pandas.DataFrame Bulk: Bulk Gene expression matrix: linear, library-size-corrected
                                      counts are expected: if linear,library size correciton is 
                                      performed (to 10k counts per sample)

        :param pandas.DataFrame or ad.AnnData or None: Cell type specific gene expresion matrix, if 
                                                       None a predefined Signature is used by setting
                                                       Tumor Type. 
                                                       If AnnData, a phenotyped scRNAseq
                                                       dataset is used with adata.obs.celltype as cell types
                                                       Ff pandas.Dataframe, a custom signature can be used for 
                                                       which the validity is checked

        :param str TumorType: Tumor type to select predefined signature [default = '', choices = ['NSCLC','PDAC','PBMC']
        :param int n_highly_variable: Number of hvgs to select for AutoGeneS marker detection [default = 3000]
        :param int Ncores: Number of cores to use for parrallel computing [default = 19]
        :param list MarkerList: Predefined list of markers to use for deconvolution [default = None, all Markers identified by AutoGeneS]

        :returns: Statescope Statescope_object: Intialized Statescope object
        """
    TumorTypes = ['NSCLC', 'PBMC', 'PDAC'] # import this list from external source

    # Check if Signature is specified
    if Signature:
        # Check if Signature is custom pd.DataFrame
        if isinstance(Signature, pd.DataFrame): 
            Check_Signature_validity(Signature)
            
        elif  isinstance(Signature, ad.AnnData):
            # Create a Signature from phenotyped AnnData (with phenotypes in AnnData.obs[celltype.key])
            Signature = CreateSignature(Signature, celltype_key = celltype_key)
    elif not TumorType  and Signature == None:
        raise AssertionError("Signature is not specified. Create custom signature or select a pre-defined signature")
    # If invalid predefined signature is specified 
    elif not TumorType in TumorTypes and Signature == None: 
        raise AssertionError(f"{TumorType} is not in {TumorTypes}. Specify custom Signature")
    # Use predefine Signature
    elif TumorType in TumorTypes:
        Signature = pd.read_csv('https://github.com/tgac-vumc/StatescopeSignatures/{TumorType}/{TumorType}_Signature_{Ncelltypes}celltypes.txt',sep ='\t', index_col = 'Gene') # read from external source

    # Fetch Statescope variables
    Samples = Bulk.columns.tolist()
    Celltypes =  [col.split('scExp_')[1] for col in Signature.columns if 'scExp_' in col ]
    Genes = [gene for gene in Bulk.index if gene in Signature.index]
        
    Signature = Signature.loc[Genes,:]
    Bulk = Bulk.loc[Genes,:]
    # Check bulk Format
    Bulk = Check_Bulk_Format(Bulk)
    Markers = Signature[Signature.IsMarker].index.tolist()
    if MarkerList:
        Markers = [gene for gene in Genes if gene in MarkerList]
    
    Omega_columns = ['scVar_'+ct for ct in Celltypes]
    Mu_columns = ['scExp_'+ct for ct in Celltypes]
    
    Statescope_object = Statescope(Bulk,Signature[Mu_columns],Signature[Omega_columns],Samples,Celltypes,Genes,Markers,Ncores)
    return Statescope_object

#-------------------------------------------------------------------------------
# 1.2  Define Statescope plotting functions
#-------------------------------------------------------------------------------
def Heatmap_Fractions(Statescope_model):
    pass

def Heatmap_GEX(Statescope_model, celltype):
    pass

def Heatmap_StateLoadings():
    pass

def Heatmap_StateScores():
    pass




#-------------------------------------------------------------------------------
# 1.3  Miscellaneous functions
#-------------------------------------------------------------------------------
# Extract Gene expression matrix
def Extract_GEX(Statescope, celltype):
    return Statescope.GEX[celltype]

def Extract_StateScores(Statescope):
    pass
