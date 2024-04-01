#!/bin/bash
#SBATCH -J fid_TEST_kmeans_svhn # A single job name for the array
#SBATCH -n 1 # total number of processes
#SBATCH -N 1 # number of nodes
#SBATCH -p seas_compute,sapphire,shared # Partition
#SBATCH --mem 30000 # Memory request (35 GB)
#SBATCH -t 0-60:00 # (D-HH:MM)
#SBATCH -o /n/holyscratch01/hlakkaraju_lab/Users/czhuangusa/cnn/lira/outputs/fid/%j.out # Standard output
#SBATCH -e /n/holyscratch01/hlakkaraju_lab/Users/czhuangusa/cnn/lira/errors/fid/%j.err # Standard error
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=catherinehuang@college.harvard.edu

# just directly run the script!
conda run -n cnn python3 fidelity.py --n_rows $1 --channel $2 --epsilon $3 --clipping_mode $4 --clustering_method $5 --explanation_type $6 --model $7 --data $8 --epochs $9 --fid_method all_clusters --dry_run True