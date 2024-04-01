#!/bin/bash
# create "outputs" folder if it doesn't exist
if [ ! -d "lira/outputs" ]; then
    mkdir "lira/outputs"
    echo "Created 'outputs' folder."
fi

# create "errors" folder if it doesn't exist
if [ ! -d "lira/errors" ]; then
    mkdir "lira/errors"
    echo "Created 'errors' folder."
fi

# iterate through the settings

# exp_no=0
# eps=2.0
# exp_type="ig"
# clipping_mode="nonDP"
# sbatch --account=hlakkaraju_lab lira_dpmodel_explanations.sh $exp_type $exp_no $clipping_mode $eps

# NEED TO RUN 17..32 LATER
experiment_nos=( {0..32} )
# experiment_nos=(18 31 32)

epsilons=(0.5 1.0 2.0 8.0)

epochs=(9 15 30 50)

# explanation_types=("gs" "ig" "dl" "sl") # ("gs" "ig")
explanation_types=("ixg" "sl" "gs" "ig") # just running ixg for now because some of these failed
# explanation_types=("ixg" "sl" "gs" "ig" "dl")

datasets=("SVHN")
# datasets=("GTSRB")

model_type="vit_small_patch16_224"

# train non-DP models
# for exp_type in "${explanation_types[@]}"; do
#     for exp_no in "${experiment_nos[@]}"; do
#         for data in "${datasets[@]}"; do
#             for epoch_count in "${epochs[@]}"; do
#                 clipping_mode="nonDP"
#                 eps=0.0
#                 variances_dir="lira/variances_${data}_20000/model=${model_type}_mode=${clipping_mode}_type=${exp_type}_nsamples=20_epochs=${epoch_count}/${exp_no}.csv"
#                 echo "Model Directory: $variances_dir"
#                 if [ -e "$variances_dir" ]; then
#                     echo "File exists. Job already ran."
#                 else
#                     sbatch --account=hlakkaraju_lab lira_explanations_other.sh $exp_type $exp_no $clipping_mode $eps $epoch_count $data
#                     sleep 0.5
#                 fi
#             done
#         done
#     done
# done

# sleep 0.5

# train DP models
experiment_nos=( {0..16} )
epochs=(15 30 50)
for exp_no in "${experiment_nos[@]}"; do
    for data in "${datasets[@]}"; do
        for eps in "${epsilons[@]}"; do
            for exp_type in "${explanation_types[@]}"; do
                for epoch_count in "${epochs[@]}"; do
                    clipping_mode="BK-MixOpt"
                     variances_dir="lira/norms_l1_${data}_20000/model=${model_type}_mode=${clipping_mode}_eps=${eps}_type=${exp_type}_nsamples=20_epochs=${epoch_count}/${exp_no}.csv"
                    echo "Model Directory: $variances_dir"
                    if [ -e "$variances_dir" ]; then
                        echo "File exists. Job already ran."
                    else
                        sbatch --account=hlakkaraju_lab lira_explanations_other.sh $exp_type $exp_no $clipping_mode $eps $epoch_count $data
                        sleep 0.5
                    fi
                done
            done
        done
    done
done