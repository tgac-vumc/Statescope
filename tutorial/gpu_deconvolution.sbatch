#!/bin/bash
#SBATCH --job-name=statescope_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2g.24gb:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --chdir=/net/beegfs/users/P094398/Statescopedeployment/Statescope/tutorial
#SBATCH --output=/net/beegfs/users/P094398/Statescopedeployment/Statescope/tutorial/Output/logs/%x_%j.out
#SBATCH --error=/net/beegfs/users/P094398/Statescopedeployment/Statescope/tutorial/Output/logs/%x_%j.err
#SBATCH --mail-user=a.bose1@amsterdamumc.nl
#SBATCH --mail-type=END,FAIL

# Activate environment
source /appdata/users/P094398/miniconda3/etc/profile.d/conda.sh

# activate your env
conda activate /appdata/users/P094398/conda-envs/statescope


# Run (GPU)
srun python /net/beegfs/users/P094398/Statescopedeployment/Statescope/tutorial/run_deconvolution_hpc.py \
  --model-path /net/beegfs/users/P094398/StatescopeDevBranch/src/Test/Data/PseudobulkTest/Test_model1.pkl \
  --backend gpu \
  --nrep 100 --njobs 4 \
  --iter-max 1000 \
  --out-dir /net/beegfs/users/P094398/Statescopedeployment/Statescope/tutorial/Output