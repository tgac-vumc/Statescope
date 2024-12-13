from Deconvolution.BLADE import Framework_Iterative
from StateDiscovery.cNMF import cNMF
from StateDiscovery.lib.cNMF
from StateDiscovery.lib import pymf

# Statescope Object
class Statescope:
    def __init__(self, Bulk, Mu,Omega,Samples,Celltypes,Genes,Markers):
        Nsample = len(Samples)
        Ncell = len(Celltypes)
        Ngene_Bulk = len(Genes)
        Nmarkers = len(Markers)

    # Do Deconvolution (FrameWork Iterative)
    def Deconvolution(self, Ind_Marker=None,
                        Alpha=1, Alpha0=1000, Kappa0=1, sY=1,
                        Nrep=10, Njob=10, fsel=0, Update_SigmaY=False, Init_Trust=10,
                        Expectation=None, Temperature=None, IterMax=100):

        Signature = self.Signature
        Bulk = self.Bulk
        # Perare Signature
        scExp_marker = Signature.iloc[self.Markers, :].filter(regex='scExcp').values.iloc[:,self.Celltypes]
        scVar_marker = Signature.iloc[self.Markers, :].filter(regex='scVar').values.iloc[:,self.Celltypes]

        # Prepare Bulk (select/match genes
        Y = Bulk.iloc[self.Samples, self.Markers]
        # Excecute BLADE Deconvolution: FrameWork iterative
        final_obj, best_obj, best_set, outs = Framework_Iterative(scExp_marker, scVar_marker, Y, Ind_Marker,
                        Alpha, Alpha0, Kappa0, sY,
                        Nrep, Njob, fsel, Update_SigmaY, Init_Trust,
                        Expectation, Temperature, IterMax)
        
        self.BLADE = final_obj
        self.Fractions = pd.DataFrame(final_obj.ExpF(final_obj.Beta), index = self.Samples,columns = self.Celltypes)

        
    # Do Gene Expression Refinement
    def Refinement(self, Ncores=10,weight=100):
        Signature = self.Signature
        Bulk = self.Bulk
        # Prepare Signature
        scExp_All = Signature.iloc[self.Markers, :].filter(regex='scExcp').values.iloc[:,self.Celltypes]
        scVar_All = Signature.iloc[self.Markers, :].filter(regex='scVar').values.iloc[:,self.Celltypes]
        # Prepare Bulk (select/match genes
        Y = Bulk.iloc[self.Samples, self.Genes]

        obj = Purify_AllGenes(self.BLADE, scExp_All, scVar_All,Ncores, weight)

        self.BLADE_final = obj
        self.GEX = obj.Mu

    # Perform State Discovery
    def StateDiscovery(self, celltype = self.Celltypes,
                       K = [None]*self.Ncell,weighing = 'Omega',n_iter = 10,n_final_iter = 100,min_cophenetic = 0.9,max_clusters=10,Ncores = 10):

        # by default, perform State Discovery for all celltypes
        if isinstance(celltype, Iterable):
            State_dict = dict()
            for ct,k in zip(celltype,K):
                State_dict[ct] = StateDiscovery_FrameWork(self.GEX,self.Omega,self.Fractions,ct,k,n_iter,n_final_iter,min_cophentic,max_clusters,Ncores)
            self.cNMF = [x for x in State_dict.values()]
            self.StateScores = pd.concat([pd.DataFrame(cNMF.H,index = self.Samples,columns).add_prefix(ct+'_') ct,cNMF in State_dict.keys()], axis = 1)
            self.StateLoadings =  pd.concat([pd.DataFrame(cNMF.W,index = self.Genes,columns).add_prefix(ct+'_') ct,cNMF in State_dict.keys()], axis = 1)
        else:
            self.cNMF = StateDiscovery_FrameWork(self.GEX,self.Omega,self.Fractions,celltype,K,n_iter,n_final_iter,min_cophentic,max_clusters,Ncores)
            self.StateScores = pd.DataFrame(cNMF.H, index = self.Samples).add_prefix(celltype)
            self.StateLoadings =pd.DataFrame(cNMF.W, index = self.Genes).add_prefix(celltype)



def Initialize_Statescope(Bulk, Signature=None,TumorType='',Ncelltypes='default'):
    TumorTypes = ['NSCLC', 'PBMC', 'PDAC']
    if not TumorType in TumorTypes and Signature == None: 
        raise AssertionError(f"{TumorType} is not in {TumorTypes}. Specify custom Signature")

    samples = Bulk.index
    Celltypes =  [col for col in Signature.columns if ]
    Genes = Bulk.columns
    Markers = # Fetch marker IDs
    
    
    Statescope_object = Statescope(Bulk,Mu,Omega,Samples,Celltypes,Genes,Markers)
    return Statescope_object

    
