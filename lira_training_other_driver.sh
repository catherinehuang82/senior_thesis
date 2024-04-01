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

experiment_nos=( {0..32} )

epsilons=(0.5 1.0 2.0 8.0)

# epochs=(7 9 15 30 50) # uncomment this when doing GTSRB
epochs=(9 15 30 50)

datasets=("GTSRB") # tentatively, to get DP explanations for higher epoch counts, as well as train accuracies

# datasets=("GTSRB" "SVHN")

model_types=("vit_small_patch16_224")

# train non-DP models
for exp_no in "${experiment_nos[@]}"; do
    for data in "${datasets[@]}"; do
        for epoch_count in "${epochs[@]}"; do
            for model_type in "${model_types[@]}"; do
                clipping_mode="nonDP"
                eps=0.0
                model_dir="lira/model_state_dicts_TEMP_${data}/model=${model_type}_mode=${clipping_mode}_epochs=${epoch_count}/${exp_no}.pt"
#                 echo "Model Directory: $model_dir"
                if [ -e "$model_dir" ]; then
                    echo "File exists. Job already ran."
                else
                    sbatch --account=hlakkaraju_lab lira_training_other.sh $data $exp_no $clipping_mode $eps $epoch_count $model_type
                    sleep 0.5
                fi
            done
        done
    done
done

# sleep 0.5

experiment_nos=( {0..16} )
epochs=(15 30 50)
# train DP models
for exp_no in "${experiment_nos[@]}"; do
    for data in "${datasets[@]}"; do
        for eps in "${epsilons[@]}"; do
            for epoch_count in "${epochs[@]}"; do
                for model_type in "${model_types[@]}"; do
                    clipping_mode="BK-MixOpt"
                    model_dir="lira/model_state_dicts_TEMP_${data}/model=${model_type}_mode=${clipping_mode}_eps=${eps}_epochs=${epoch_count}/${exp_no}.pt"
#                     echo "Model Directory: $model_dir"
                    if [ -e "$model_dir" ]; then
                        echo "File exists. Job already ran."
                    else
                        sbatch lira_training_other.sh $data $exp_no $clipping_mode $eps $epoch_count $model_type
                        sleep 0.5
                    fi
                done
            done
        done
    done
done
