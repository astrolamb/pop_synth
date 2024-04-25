#!/bin/bash
#SBATCH --job-name=pop_synth
#SBATCH --output=pop_synth.out
#SBATCH --error=pop_synth.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mem=1GB
#SBATCH --account=taylor_group
#SBATCH --partition=production
#SBATCH --mail-user=william.g.lamb@vanderbilt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint='zen|skylake'
#SBATCH --array=0-100

# Load any necessary modules
module purge
module load Anaconda3/2022.05

# Activate your virtual environment if needed
source activate /home/lambwg/.conda/envs/ceffyl/

# Run your Python script
python ./pop_synth.py --Nidx $SLURM_ARRAY_TASK_ID --outdir /data/taylor_group/william_lamb/pop_synth/h2cf/model4/