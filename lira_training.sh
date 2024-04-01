#!/bin/bash
#SBATCH -J 0_32_dp_base_fewerepochs_cifar100_model_training # A single job name for the array
#SBATCH --gres=gpu:1
#SBATCH -p seas_gpu # Partition
#SBATCH --mem 30000 # Memory request (30 GB)
#SBATCH -t 0-6:00 # (D-HH:MM)
#SBATCH -o /n/holyscratch01/hlakkaraju_lab/Users/czhuangusa/cnn/lira/outputs/training/%j.out # Standard output
#SBATCH -e /n/holyscratch01/hlakkaraju_lab/Users/czhuangusa/cnn/lira/errors/training/%j.err # Standard error
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=catherinehuang@college.harvard.edu

# just directly run the script!
conda run -n cnn python3 lira_training.py --cifar_data CIFAR100 --experiment_no $1 --clipping_mode $2 --epsilon $3 --epochs $4 --model $5