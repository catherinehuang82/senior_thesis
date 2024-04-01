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

experiment_nos=( {0..16} ) # endpoint is 16 to report mean/sd accuracy over 16 experiments

epsilons=(0.5 1.0 2.0 8.0)

epochs=(50)
# epochs=(5 7 9 30 50)

models=("beit_base_patch16_224.in22k_ft_in22k_in1k")
# models=("vit_base_patch16_224.augreg2_in21k_ft_in1k" "vit_large_patch16_224.augreg_in21k_ft_in1k")

# train non-DP models
for exp_no in "${experiment_nos[@]}"; do
    for epoch_count in "${epochs[@]}"; do
        for model_type in "${models[@]}"; do
            clipping_mode="nonDP"
            eps=0.0
            model_dir="lira/model_state_dicts_TEMP_CIFAR100_20000/model=${model_type}_mode=${clipping_mode}_epochs=${epoch_count}/${exp_no}.pt"
            echo "Model Directory: $model_dir"
            if [ -e "$model_dir" ]; then
                echo "File exists. Job already ran."
            else
                sbatch --account=hlakkaraju_lab lira_training_20000.sh $exp_no $clipping_mode $eps $epoch_count $model_type
                sleep 0.5
            fi
        done
    done
done

sleep 0.5
# why do we only run epochs=3, 5, 7, 9 in the DP case? i imagine accuracy for the models trained on too many epochs is so low
# that these models are no longer useful
epochs=(5 7 9)

# train DP models
# experiment_nos=( {0..16} )
# for exp_no in "${experiment_nos[@]}"; do
#     for eps in "${epsilons[@]}"; do
#         for epoch_count in "${epochs[@]}"; do
#             clipping_mode="BK-MixOpt"
#             model_dir="lira/model_state_dicts_TEMP_CIFAR100_20000/model=${model_type}_mode=${clipping_mode}_eps=${eps}_epochs=${epoch_count}/${exp_no}.pt"
#             echo "Model Directory: $model_dir"
#             if [ -e "$model_dir" ]; then
#                 echo "File exists. Job already ran."
#             else
#                 sbatch lira_training_20000.sh $exp_no $clipping_mode $eps $epoch_count $model_type
#                 sleep 0.5
#             fi
#         done
#     done
# done