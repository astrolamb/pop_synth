#!/bin/bash
#SBATCH --job-name=pop_synth
#SBATCH --output=bootstrap_%a.out
#SBATCH --error=bootstrap_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=5GB
#SBATCH --account=taylor_group
#SBATCH --partition=production
#SBATCH --mail-user=william.g.lamb@vanderbilt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint='zen'
#SBATCH --array=0-99

# Load any necessary modules
module purge
module load Anaconda3/2022.05

# Activate your virtual environment if needed
source activate /home/lambwg/.conda/envs/ceffyl/

cd /home/lambwg/pop_synth

# Run your Python script
python ./scripts/bootstrap.py --real_idx $SLURM_ARRAY_TASK_ID --model_idx 3