#!/bin/bash
#SBATCH -J gtsrb_trainacc_training # A single job name for the array
#SBATCH --gres=gpu:1
#SBATCH -p seas_gpu,gpu # Partition
#SBATCH --mem 30000 # Memory request (30 GB)
#SBATCH -t 0-25:00 # (D-HH:MM)
#SBATCH -o /n/holyscratch01/hlakkaraju_lab/Users/czhuangusa/cnn/lira/outputs/other_training_20000/%j.out # Standard output
#SBATCH -e /n/holyscratch01/hlakkaraju_lab/Users/czhuangusa/cnn/lira/errors/other_training_20000/%j.err # Standard error
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=catherinehuang@college.harvard.edu

# just directly run the script!
conda run -n cnn python3 lira_training_other.py --total_data_examples 20000 --data $1 --experiment_no $2 --clipping_mode $3 --epsilon $4 --epochs $5 --model $6
