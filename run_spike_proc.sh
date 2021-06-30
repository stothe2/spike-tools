#!/bin/bash

#SBATCH --job-name=spk_proc
#SBATCH --array=0-287
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --output="/om/user/sachis/spike_proc/slurm-%A_%a.out"
#SBATCH --mem=10000

source /braintree/home/sachis/miniconda3/etc/profile.d/conda.sh
conda activate spike-tools

cd /braintree/data2/active/users/sachis/spike-tools
python spike_tools/spike_proc.py $SLURM_ARRAY_TASK_ID