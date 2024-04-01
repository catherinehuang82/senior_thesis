#!/bin/bash
#SBATCH -J cif100_gs_expl # A single job name for the array
#SBATCH -n 1 # total number of processes
#SBATCH -N 1 # number of nodes
#SBATCH -p seas_compute # Partition
#SBATCH --mem 20000 # Memory request (20 GB)
#SBATCH -t 0-96:00 # (D-HH:MM)
#SBATCH -o /n/holyscratch01/hlakkaraju_lab/Users/czhuangusa/cnn/lira/outputs/cifar100_20k_explanations/%j.out # Standard output
#SBATCH -e /n/holyscratch01/hlakkaraju_lab/Users/czhuangusa/cnn/lira/errors/cifar100_20k_explanations/%j.err # Standard error
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=catherinehuang@college.harvard.edu

# just directly run the script!
conda run -n cnn python3 lira_dpmodel_explanations.py --total_data_examples 20000 --explanation_type $1 --experiment_no $2 --clipping_mode $3 --epsilon $4 --epochs $5 --model $6
