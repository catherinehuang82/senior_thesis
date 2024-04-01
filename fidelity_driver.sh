#!/bin/bash
# iterate through the settings

nrows_list=(10000)

channel_list=(1 2 3)

epsilon_list=(0.5 1.0 2.0 8.0)

# clustering_method_list=("hierarchical")
clustering_method_list=("kmeans")

# explanation_type_list=("ig")

explanation_type_list=("sl" "ixg" "ig")

model="vit_small_patch16_224" # "beit_base_patch16_224.in22k_ft_in22k_in1k"

data="SVHN"

epochs=50

# train non-DP models
for exp_type in "${explanation_type_list[@]}"; do
    for nrows in "${nrows_list[@]}"; do
        for clustering_method in "${clustering_method_list[@]}"; do
            for channel in "${channel_list[@]}"; do
                clipping_mode="nonDP"
                eps=0.0
                fid_dir="fidelity_test1/fid_${data}_${clustering_method}_method=all_clusters/mode=${clipping_mode}_nrows=${nrows}_type=${exp_type}_epochs=${epochs}_channel=${channel}.csv"
                echo "fidelity directory: $fid_dir"
                if [ -e "$fid_dir" ]; then
                    echo "File exists. Job already ran."
                else
                    sbatch --account=hlakkaraju_lab fidelity.sh $nrows $channel $eps $clipping_mode $clustering_method $exp_type $model $data $epochs
                    sleep 0.7
                fi
            done
        done
    done
done


# train DP models
for exp_type in "${explanation_type_list[@]}"; do
    for nrows in "${nrows_list[@]}"; do
        for clustering_method in "${clustering_method_list[@]}"; do
            for eps in  "${epsilon_list[@]}"; do
                for channel in "${channel_list[@]}"; do
                    clipping_mode="BK-MixOpt"
 fid_dir="fidelity_test1/fid_${data}_${clustering_method}_method=all_clusters/mode=${clipping_mode}_nrows=${nrows}_type=${exp_type}_epsilon=${eps}_epochs=${epochs}_channel=${channel}.csv"
                    echo "fidelity directory: $fid_dir"
                    if [ -e "$fid_dir" ]; then
                        echo "File exists. Job already ran."
                    else
                        sbatch --account=hlakkaraju_lab fidelity.sh $nrows $channel $eps $clipping_mode $clustering_method $exp_type $model $data $epochs
                        sleep 0.7
                    fi
                done
            done
        done
    done
done