#!/bin/bash
# iterate through the settings

nrows_list=(2500)

epsilon_list=(0.5 1.0 2.0 8.0)

explanation_type_list=("sl" "ixg")

model="beit_base_patch16_224.in22k_ft_in22k_in1k" # "vit_small_patch16_224"

data="CIFAR100"

epochs=9

k_list=(50 100 1000)

m_list=(10)

sigma_list=(0.1 0.25)

# train non-DP models
for exp_type in "${explanation_type_list[@]}"; do
    for nrows in "${nrows_list[@]}"; do
        for k in "${k_list[@]}"; do
            for m in "${m_list[@]}"; do
                for sigma in "${sigma_list[@]}"; do
                    clipping_mode="nonDP"
                    eps=0.0
                    fid_dir="perturbation_gap_pred_test/${data}_sigma=${sigma}_k=${k}_m=${m}/mode=${clipping_mode}_nrows=${nrows}_type=${exp_type}_epochs=${epochs}.csv"
                    echo "fidelity directory: $fid_dir"
                    if [ -e "$fid_dir" ]; then
                        echo "File exists. Job already ran."
                    else
                        sbatch --account=hlakkaraju_lab perturbation_gap_pred.sh $data $epochs $eps $clipping_mode $model $exp_type $sigma $k $m
                        sleep 0.5
                    fi
                done
            done
        done
    done
done


# train DP models
for exp_type in "${explanation_type_list[@]}"; do
    for nrows in "${nrows_list[@]}"; do
        for eps in  "${epsilon_list[@]}"; do
            for k in "${k_list[@]}"; do
                for m in  "${m_list[@]}"; do
                    for sigma in "${sigma_list[@]}"; do
                        clipping_mode="BK-MixOpt"
                        fid_dir="perturbation_gap_pred_test/${data}_sigma=${sigma}_k=${k}_m=${m}/mode=${clipping_mode}_nrows=${nrows}_type=${exp_type}_epsilon=${eps}_epochs=${epochs}.csv"
                        echo "fidelity directory: $fid_dir"
                        if [ -e "$fid_dir" ]; then
                            echo "File exists. Job already ran."
                        else
                            sbatch --account=hlakkaraju_lab perturbation_gap_pred.sh $data $epochs $eps $clipping_mode $model $exp_type $sigma $k $m
                            sleep 0.5
                        fi
                    done
                done
            done
        done
    done
done