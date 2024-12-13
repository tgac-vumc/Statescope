#!/usr/bin/python3
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Statescope.py
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Statescope framework code
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
#  13-12-2024: File creation, write code
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 0.1  Import Libraries
#-------------------------------------------------------------------------------
from BLADE_Deconvolution.BLADE import Framework_Iterative,Purify_AllGenes
from StateDiscovery.cNMF import cNMF
import StateDiscovery.cNMF
from StateDiscovery.lib import pymf
import pandas as pd
from glob import glob

#-------------------------------------------------------------------------------
# 1.1  Define Statescope Object
#-------------------------------------------------------------------------------
class Statescope_Object:
    def __init__(self, Bulk, scExp,scVar,Samples,Celltypes,Genes,Markers):
        self.Bulk = Bulk
        self.scExp = scExp
        self.scVar = scVar
        self.Samples = Samples
        self.Celltypes = Celltypes
        self.Genes = Genes
        self.Markers = Markers


    def Deconvolution(self, Ind_Marker=None,
                        Alpha=1, Alpha0=1000, Kappa0=1, sY=1,
                        Nrep=10, Njob=10, fsel=0, Update_SigmaY=False, Init_Trust=10,
                        Expectation=None, Temperature=None, IterMax=100):
        """ 
        Perform BLADE Deconvolution
        :param Statescope_Object self: Initialized Statescope_Object
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
                        Nrep, Njob, fsel, Update_SigmaY, Init_Trust,
                        Expectation, Temperature, IterMax)
        # Save BLADE result in Statescope object
        self.BLADE = final_obj
        # Save fractions as dataframe in object
        self.Fractions = pd.DataFrame(final_obj.ExpF(final_obj.Beta), index = self.Samples,columns = self.Celltypes)

        
    # Do Gene Expression Refinement
    def Refinement(self, Ncores=10,weight=100):
        """ 
        Perform Gene expression refinement with all genes
        :param Statescope_Object self: Statescope_Object
        :param int Njob: Number of parralel jobs [default = 10]
        :param int weight: Parameter to weigh down fraction estimation objective [default = 100]

        :returns: BLADE_Object self.BLADE_final: BLADE object
        :returns: {ct:pandas.DataFrame} self.GEX: Dictionary of cell type specific GEX {ct:ctSpecificGEX}
        """
        # Prepare Signature
        scExp_All = self.scExp.loc[self.Genes, :].to_numpy()
        scVar_All = self.scVar.loc[self.Genes, :].to_numpy()
        # Prepare Bulk (select/match genes
        Y = self.Bulk.loc[self.Genes,self.Samples].to_numpy()
        # Perform gene expression revinement with all genes in signature
        obj = Purify_AllGenes(self.BLADE, scExp_All, scVar_All,Y,Ncores, weight)
        # create output GEX dictionary
        GEX = {ct:pd.DataFrame(obj.Nu[:,:,i],index=self.Samples,columns=self.Genes) for i,ct in enumerate(self.Celltypes)}
        # Store in Statescope object
        self.BLADE_final = obj
        self.GEX = GEX

    # Perform State Discovery
    def StateDiscovery(self, celltype = '',
                       K = [None],weighing = 'Omega',n_iter = 10,n_final_iter = 100,min_cophenetic = 0.9,max_clusters=10,Ncores = 10):
        """ 
        Perform StateDiscovery from ctSpecificGEX using cNMF
        by default all cell types are used and the optimal K is determined
        ctSpecific GEX is by default weighed by Omega

        :param Statescope_Object self Statescope_Object
        :param str celltype: cell type to perform state discovery [default = '' (All)]
        :param int K: number of states to return, by default the optimal K is chosen [None]
        :param str weighing: [default = 'Omega', choices: ['Omega','OmegaFractions','centering','no_weighing']]
        :param int n_iter: Number of initial cNMF restarts [default = 10]
        :param int n_iter_final: Number of final cNMF restarts [default = 100]
        :param float min_cophenetic: minimum cophentic coeffient to determine K [default = 0.9]
        :param int max_cluster: maximum number of states to consider [default = 10]
        :param int Ncores: Number of cores to use for parrallel restarts [default = 19]

        :returns: CNMF_object self.cNMF: cNMF object
        :returns: StateScores self.StateScores: cNMF state scores from all cell types compiled (NSample x (NcellxNState))
        :returns: StateLoadings self.StateLoadings: cNMF state loadings from all cell types compiled (NGene x (NcellxNState))
        """
        if celltype == '':
            celltype = self.Celltypes
            K = K * self.Ncell
        # by default, perform State Discovery for all celltypes
        if isinstance(celltype, Iterable):
            State_dict = dict()
            # Iterate over celltypes
            for ct,k in zip(celltype,K):
                # save cNMF models in State dict
                State_dict[ct] = StateDiscovery_FrameWork(self.GEX,self.Omega,self.Fractions,ct,k,n_iter,n_final_iter,min_cophentic,max_clusters,Ncores)
            # save cNMF models as dict
            self.cNMF = State_dict
            # Save State scores as compile pandas.DataFrame
            self.StateScores = pd.concat([pd.DataFrame(cNMF.H,index = self.Samples).add_prefix(ct+'_') for ct,cNMF in State_dict.items()], axis = 1)
            # Save State loadings as compile pandas.DataFrame
            self.StateLoadings =  pd.concat([pd.DataFrame(cNMF.W,index = self.Genes).add_prefix(ct+'_') for ct,cNMF in State_dict.items()], axis = 1)
        # alternatively, perform State discovery for one cell type
        else:
            # save cNMF models as single cNMF instance
            self.cNMF = StateDiscovery_FrameWork(self.GEX,self.Omega,self.Fractions,celltype,K,n_iter,n_final_iter,min_cophentic,max_clusters,Ncores)
            # save State scores and loadings
            self.StateScores = pd.DataFrame(cNMF.H, index = self.Samples).add_prefix(celltype)
            self.StateLoadings =pd.DataFrame(cNMF.W, index = self.Genes).add_prefix(celltype)




#-------------------------------------------------------------------------------
# 1.2  Define Statescope Initialization
#-------------------------------------------------------------------------------


def Check_Bulk_Format(Bulk)


def Initialize_Statescope(Bulk, Signature=None,TumorType='',Ncelltypes=''):
    TumorTypes = ['NSCLC', 'PBMC', 'PDAC']
    if not TumorType  and Signature == None:
        raise AssertionError("Signature is not specified")
    elif Signature:
        pass
    elif not TumorType in TumorTypes and Signature == None: 
        raise AssertionError(f"{TumorType} is not in {TumorTypes}. Specify custom Signature")
    elif TumorType in TumorTypes:
        Signature = pd.read_csv(glob(f'Framework/BLADE_Deconvolution/Signatures/{TumorType}/{TumorType}_Signature_{Ncelltypes}*celltypes.txt')[0],sep ='\t', index_col = 'Gene')

    
    Samples = Bulk.columns.tolist()
    Celltypes =  [col.split('scExp_')[1] for col in Signature.columns if 'scExp_' in col ]
    Genes = [gene for gene in Bulk.index if gene in Signature.index]
    Signature = Signature.loc[Genes,:]
    Bulk = Bulk.loc[Genes,:]
    Markers = Signature[Signature.IsMarker].index.tolist()
    Omega_columns = ['scVar_'+ct for ct in Celltypes]
    Mu_columns = ['scExp_'+ct for ct in Celltypes]
    Statescope_object = Statescope_Object(Bulk,Signature[Mu_columns],Signature[Omega_columns],Samples,Celltypes,Genes,Markers)
    return Statescope_object

    
#-------------------------------------------------------------------------------
# 1.3  Define Statescope plotting functions
#-------------------------------------------------------------------------------
def Heatmap_Fractions(Statescope_model):
    pass
