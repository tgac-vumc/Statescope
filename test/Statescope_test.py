import sys
import os
os.chdir('../')
sys.path.insert(1, 'Framework/')
sys.path.insert(1, 'Framework/StateDiscovery/lib/pymf/')
import Statescope
from Statescope import Initialize_Statescope
import pandas as pd
import pickle

Bulk = pd.read_csv('https://github.com/tgac-vumc/OncoBLADE/raw/refs/heads/main/data/Transcriptome_matrix_subset.txt', sep = '\t', index_col = 'symbol')
Statescope_model = Initialize_Statescope(Bulk,TumorType = 'NSCLC', Ncores = 40)
Statescope_model.Deconvolution()
Statescope_model.Refinement()
Statescope_model.StateDiscovery()
print(Statescope_model.StateScores)
