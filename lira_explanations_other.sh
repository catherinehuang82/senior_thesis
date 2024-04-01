#!/bin/bash
#SBATCH -J svhn_dp_moreepochs # A single job name for the array
#SBATCH -n 1 # total number of processes
#SBATCH -N 1 # number of nodes
#SBATCH -p seas_compute,sapphire,shared # Partition
#SBATCH --mem 45000 # Memory request (35 GB)
#SBATCH -t 0-70:00 # (D-HH:MM)
#SBATCH -o /n/holyscratch01/hlakkaraju_lab/Users/czhuangusa/cnn/lira/outputs/other_20k_explanations/%j.out # Standard output
#SBATCH -e /n/holyscratch01/hlakkaraju_lab/Users/czhuangusa/cnn/lira/errors/other_20k_explanations/%j.err # Standard error
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=catherinehuang@college.harvard.edu

# just directly run the script!
conda run -n cnn python3 lira_explanations_other.py --total_data_examples 20000 --model vit_small_patch16_224 --explanation_type $1 --experiment_no $2 --clipping_mode $3 --epsilon $4 --epochs $5 --data $6