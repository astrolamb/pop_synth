#!/bin/bash
#SBATCH --job-name=pop_synth
#SBATCH --output=pop_synth.out
#SBATCH --error=pop_synth.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=20GB
#SBATCH --account=taylor_group
#SBATCH --partition=production
#SBATCH --mail-user=william.g.lamb@vanderbilt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint='zen'
#SBATCH --array=1-5

# Load any necessary modules
module purge
module load Anaconda3/2022.05

# Activate your virtual environment if needed
source activate /home/lambwg/.conda/envs/ceffyl/

cd /home/lambwg/pop_synth

# Run your Python script
python ./scripts/bootstrap.py --model_idx $SLURM_ARRAY_TASK_ID