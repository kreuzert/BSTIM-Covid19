#!/bin/bash -x

# Slurm script to call the specific bash script that samples
# ia effects for a specific 3 week window.

#SBATCH --job-name=134_IA_ROUT
#SBATCH --account=covid19dynstat
#SBATCH --partition=batch
#SBATCH --array=1-100:4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --output=/p/project/covid19dynstat/%u/runs/%A_o.txt
#SBATCH --error=/p/project/covid19dynstat/%u/runs/%A_e.txt
# ATTENTION:
#     1. ensure SLURM output/errors directory exists
#     2. do not switch to zsh in ~/.bashrc because of "jutil env" 

# select project
jutil env activate -p covid19dynstat

# run tasks
srun --exclusive -n ${SLURM_NTASKS} gridjob_sample_ia_window_134.slurm.sh