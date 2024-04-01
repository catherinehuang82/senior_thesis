#!/bin/bash
#SBATCH -J pred_TESTk100_cif10 # A single job name for the array
#SBATCH -n 1 # total number of processes
#SBATCH -N 1 # number of nodes
#SBATCH -p seas_compute,sapphire,shared # Partition
#SBATCH --mem 50000 # Memory request (50 GB)
#SBATCH -t 0-60:00 # (D-HH:MM)
#SBATCH -o /n/holyscratch01/hlakkaraju_lab/Users/czhuangusa/cnn/lira/outputs/perturbation_gap/%j.out # Standard output
#SBATCH -e /n/holyscratch01/hlakkaraju_lab/Users/czhuangusa/cnn/lira/errors/perturbation_gap/%j.err # Standard error
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=catherinehuang@college.harvard.edu

# just directly run the script!
conda run -n cnn python3 prediction_gap_pred.py --data $1 --epochs $2 --epsilon $3 --clipping_mode $4 --model $5 --explanation_type $6 --sigma $7 --k $8 --m $9