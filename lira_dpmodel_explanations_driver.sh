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

# experiment_nos=( {0..32} )
experiment_nos=( {0..32} )

epsilons=(0.5 1.0 2.0 8.0)

# epochs=(7 9 30 50)
epochs=(9)

models=("beit_base_patch16_224.in22k_ft_in22k_in1k")
# models=("vit_large_patch16_224.augreg_in21k_ft_in1k")

explanation_types=("gs") # ("gs" "ig")
# explanation_types=("ixg" "dl" "sl")

# train non-DP models
for exp_no in "${experiment_nos[@]}"; do
    for exp_type in "${explanation_types[@]}"; do
        for epoch_count in "${epochs[@]}"; do
            for model_type in "${models[@]}"; do
                clipping_mode="nonDP"
                eps=0.0
                                   variances_dir="lira/norms_l1_CIFAR100_20000/model=${model_type}_mode=${clipping_mode}_type=${exp_type}_nsamples=20_epochs=${epoch_count}/${exp_no}.csv"
                echo "Model Directory: $variances_dir"
                if [ -e "$variances_dir" ]; then
#                 if [ -e "$variances_dir" ]; then
                    echo "File exists. Job already ran."
                else
                    sbatch --account=hlakkaraju_lab lira_dpmodel_explanations.sh $exp_type $exp_no $clipping_mode $eps $epoch_count $model_type
                    sleep 0.5
                fi
            done
        done
    done
done

# sleep 0.5

# epochs=(7 9)

# train DP models
# experiment_nos=( {0..16} )
# for exp_no in "${experiment_nos[@]}"; do
#     for eps in "${epsilons[@]}"; do
#         for exp_type in "${explanation_types[@]}"; do
#             for epoch_count in "${epochs[@]}"; do
#                 clipping_mode="BK-MixOpt"
#                 variances_dir="lira/norms_l1_CIFAR100_20000/model=${model_type}_mode=${clipping_mode}_eps=${eps}_type=${exp_type}_nsamples=20_epochs=${epoch_count}/${exp_no}.csv"
#                 echo "Model Directory: $variances_dir"
#                 if [ -e "$variances_dir" ]; then
# #                 if [ -e "$variances_dir" ]; then
#                     echo "File exists. Job already ran."
#                 else
#                     sbatch --account=hlakkaraju_lab lira_dpmodel_explanations.sh $exp_type $exp_no $clipping_mode $eps $epoch_count $model_type
#                     sleep 0.5
#                 fi
#             done
#         done
#     done
# done