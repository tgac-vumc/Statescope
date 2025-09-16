#!/bin/bash
#SBATCH --job-name=statescope_cpu
#SBATCH --partition=defq                 # adjust for your cluster 
#SBATCH --cpus-per-task=32              # total CPU cores
#SBATCH --mem=250G                       # RAM                
#SBATCH --chdir=/net/beegfs/users/P094398/Statescopedeployment/Statescope/tutorial         # <- change me
#SBATCH --output=/net/beegfs/users/P094398/Statescopedeployment/Statescope/tutorial/Output/logs/%x_%j.out # <- change me
#SBATCH --error=/net/beegfs/users/P094398/Statescopedeployment/Statescope/tutorial/Output/logs/%x_%j.err  # <- change me
#SBATCH --mail-user=a.bose1@amsterdamumc.nl    # optional
#SBATCH --mail-type=END,FAIL


# Activate environment
source /appdata/users/P094398/miniconda3/etc/profile.d/conda.sh

# activate your env
conda activate /appdata/users/P094398/conda-envs/statescope

# Run (CPU)
srun python /net/beegfs/users/P094398/Statescopedeployment/Statescope/tutorial/run_deconvolution_hpc.py \
  --model-path /net/beegfs/users/P094398/StatescopeDevBranch/src/Test/Data/PseudobulkTest/Test_model1.pkl \
  --backend cpu \
  --nrep 100 --njobs 32 \
  --iter-max 1000 \
  --out-dir /net/beegfs/users/P094398/Statescopedeployment/Statescope/tutorial/Output