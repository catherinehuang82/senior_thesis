import numpy as np
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython import display
import random
import scipy.stats
from sklearn import metrics
from sklearn.metrics import roc_curve
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    print(f'clustering_method: {args.clustering_method}')
    print(f'dataset: {args.data}')
    print(f'explanation_type: {args.explanation_type}')
    print(f'epsilon: {args.epsilon}')
    print(f'channel: {args.channel}')
    print(f'clipping_mode: {args.clipping_mode}')
    if args.clustering_method == 'kmeans':
        df_fid = pd.DataFrame(columns=['experiment_no', 'k', 'consistency', 'num_clusters', 'num_singles', 'epsilon', 'inertia'])
    elif args.clustering_method == 'hierarchical':
        df_fid = pd.DataFrame(columns=['experiment_no', 'max_distance', 'consistency', 'num_clusters', 'num_singles', 'epsilon'])
    
    fid_cluster_dir = f'fidelity_cluster_sizes/{args.data}_{args.clustering_method}_method={args.fid_method}'
    if not os.path.exists(fid_cluster_dir):
        os.makedirs(fid_cluster_dir)
    if 'nonDP' in args.clipping_mode:
        fid_cluster_path = f'{fid_cluster_dir}/type={args.explanation_type}_mode={args.clipping_mode}_epochs={args.epochs}_channel={args.channel}.txt'
    else:
        fid_cluster_path = f'{fid_cluster_dir}/type={args.explanation_type}_mode={args.clipping_mode}_eps={args.epsilon}_epochs={args.epochs}_channel={args.channel}.txt'
        
    ## clear existing contents of the file
    try:
        with open(fid_cluster_path, 'w') as file:
            pass
    except Exception as e:
        print(f"An error occurred: {e}")
        # Continue with the code even if an error occurs
        pass
    
    for experiment_no in tqdm(range(args.num_experiments)):
        exp_no = str(experiment_no)
        try:
            with open(fid_cluster_path, 'a') as file:
                file.write(f'EXPERIMENT NO: {exp_no}\n')
        except Exception as e:
            print(f"An error occurred: {e}")
            # Continue with the code even if an error occurs
            pass
        
        # get file paths for data
        if 'nonDP' in args.clipping_mode:
            df_path = f'lira/attributions_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_type={args.explanation_type}_nsamples=20_epochs={args.epochs}/channel{args.channel}/{exp_no}.csv'
            indices_path = f'lira/indices_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}/{exp_no}.csv'
            preds_path = f'lira/preds_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_type={args.explanation_type}_nsamples=20_epochs={args.epochs}/{exp_no}.csv'
        else:
            df_path = f'lira/attributions_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_type={args.explanation_type}_nsamples=20_epochs={args.epochs}/channel{args.channel}/{exp_no}.csv'
            indices_path = f'lira/indices_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_epochs={args.epochs}/{exp_no}.csv'
            preds_path = f'lira/preds_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_type={args.explanation_type}_nsamples=20_epochs={args.epochs}/{exp_no}.csv'
        
        if not os.path.exists(df_path) or not os.path.exists(preds_path):
            continue
        print('reading in dataframes...')
        print(f'df_path: {df_path}')
        
        # read in the data
        df_preds = pd.read_csv(preds_path,
    #                            skiprows=skip_rows,
    #                            header=None
                               nrows=args.n_rows
                              )
        df_attr = pd.read_csv(df_path,
    #                               skiprows=skip_rows,
    #                               header=None
                                  nrows=args.n_rows
                                 )
        df_indices = pd.read_csv(indices_path,
    #                              skiprows=skip_rows,
    #                              header=None
                                 nrows=args.n_rows
                                )
        df_attr = df_attr[df_indices[exp_no] == 0]
        df_preds = df_preds[df_indices[exp_no] == 0]
        print(f'df_attr size: {df_attr.size}')
        print('read in dataframes')
#         print(df_attr.shape)
        if 'kmeans' in args.clustering_method:
            k_list = [10, 25, 50, 100, 150, 200]
            for k in tqdm(k_list):
#                 df_fid = pd.DataFrame(columns=['experiment_no', 'k', 'consistency', 'num_clusters', 'epsilon'])
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(df_attr)
                clusters = kmeans.labels_
                df_preds[str(k)] = clusters
                consistency=0.0
                total_in_clusters=0.0
                large_cluster_count = 0
                num_singles = 0
                try:
                    with open(fid_cluster_path, 'a') as file:
                        file.write(f'k: {k}\n')
                except Exception as e:
                    print(f"An error occurred: {e}")
                    # Continue with the code even if an error occurs
                    pass
                for c in np.unique(clusters):
                    df_preds_subset = df_preds.loc[df_preds[str(k)] == c]
                    total_cluster_size = len(df_preds_subset)
                    if 'max' in args.fid_method:
                        mode_label = df_preds_subset[exp_no].mode().iloc[0]
                        mode_label_count = len(df_preds_subset.loc[df_preds_subset[exp_no] == mode_label])
                        if total_cluster_size <= 1.0:
                            num_singles += 1
                            continue 
#                         print(total_cluster_size, mode_label_count / total_cluster_size)
                        consistency+=mode_label_count
                        total_in_clusters+=total_cluster_size
                        large_cluster_count += 1
                    elif args.fid_method == 'all_clusters':
                        if total_cluster_size <= 1.0:
                            num_singles += 1
                            continue
                        try:
                            with open(fid_cluster_path, 'a') as file:
                                file.write(str(total_cluster_size) + '\n')
                                print(f"File updated with the cluster count value: {total_cluster_size}")
                        except Exception as e:
                            print(f"An error occurred: {e}")
                            # Continue with the code even if an error occurs
                            pass
                        cluster_counts = df_preds_subset[exp_no].value_counts().tolist()
                        for cluster_count in cluster_counts:
                            consistency += cluster_count * (cluster_count - 1) / (total_cluster_size - 1)
                        total_in_clusters += total_cluster_size
                        large_cluster_count += 1
                if total_in_clusters > 2.0:
                    row = {
                        'experiment_no': experiment_no,
                        'k': k,
                        'consistency': consistency / total_in_clusters,
                        'num_clusters': large_cluster_count,
                        'num_singles': num_singles,
                        'epsilon': args.epsilon,
                        'inertia': kmeans.inertia_
                    }
                    print(f'consistency for exp {experiment_no}, k {k}, num_clusters {large_cluster_count}: {consistency / total_in_clusters}')
                    df_fid.loc[len(df_fid)] = list(row.values())
                try:
                    with open(fid_cluster_path, 'a') as file:
                        file.write('=====\n')
                except Exception as e:
                    print(f"An error occurred: {e}")
                    # Continue with the code even if an error occurs
                    pass
        elif args.clustering_method == 'hierarchical':
#             df_fid = pd.DataFrame(columns=['experiment_no', 'max_distance', 'consistency', 'num_clusters', 'epsilon'])
            if 'nonDP' in args.clipping_mode:
                if args.data == 'SVHN' and args.explanation_type == 'SL':
                    max_distances = [0.2, 0.25, 0.3, 0.35, 0.4]
                elif args.data == 'SVHN' and args.explanation_type == 'IXG':
                    max_distances = [0.06, 0.12, 0.24, 0.3, 0.36]
                elif args.data == 'SVHN':
                    max_distances = [0.2, 0.25, 0.3, 0.35, 0.4]
                elif args.data == 'CIFAR10':
                    max_distances = [0.3, 0.4, 0.5, 0.6, 0.7]
                elif args.data == 'CIFAR100':
                    max_distances = [0.1, 0.2, 0.3, 0.4, 0.5]
            else:
                max_distances = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
            for dist in tqdm(max_distances):
                linkage_matrix = linkage(df_attr, method='complete')
                clusters = fcluster(linkage_matrix, dist, criterion='distance')
                df_preds[str(dist)] = clusters
                consistency=0.0
                total_in_clusters=0.0
                large_cluster_count = 0
                num_singles = 0
                try:
                    with open(fid_cluster_path, 'a') as file:
                        file.write(f'max_distance: {dist}\n')
                except Exception as e:
                    print(f"An error occurred: {e}")
                    # Continue with the code even if an error occurs
                    pass
                for c in np.unique(clusters):
                    df_preds_subset = df_preds.loc[df_preds[str(dist)] == c]
                    total_cluster_size = len(df_preds_subset)
                    
                    if 'max' in args.fid_method:
                        mode_label = df_preds_subset[exp_no].mode().iloc[0]
                        mode_label_count = len(df_preds_subset.loc[df_preds_subset[exp_no] == mode_label])

                        if total_cluster_size <= 1.0:
                            num_singles += 1
                            continue
#                         print(total_cluster_size, mode_label_count / total_cluster_size)
                        consistency+=mode_label_count
                        total_in_clusters+=total_cluster_size
                        large_cluster_count += 1
                    elif args.fid_method == 'all_clusters':
                        if total_cluster_size <= 1.0:
                            num_singles += 1
                            continue
                        try:
                            with open(fid_cluster_path, 'a') as file:
                                file.write(str(total_cluster_size) + '\n')
                                print(f"File updated with the cluster count value: {total_cluster_size}")
                        except Exception as e:
                            print(f"An error occurred: {e}")
                            # Continue with the code even if an error occurs
                            pass
                        cluster_counts = df_preds_subset[exp_no].value_counts().tolist()
                        for cluster_count in cluster_counts:
                            consistency += cluster_count * (cluster_count - 1) / (total_cluster_size - 1)
                        total_in_clusters += total_cluster_size
                        large_cluster_count += 1
                if total_in_clusters > 2.0:
                    row = {
                        'experiment_no': experiment_no,
                        'max_distance': dist,
                        'consistency': consistency / total_in_clusters,
                        'num_clusters': large_cluster_count,
                        'num_singles': num_singles,
                        'epsilon': args.epsilon,
                    }
                    print(f'consistency for exp {experiment_no}, dist {dist}, num_clusters {large_cluster_count}: {consistency / total_in_clusters}')
                    df_fid.loc[len(df_fid)] = list(row.values())
                try:
                    with open(fid_cluster_path, 'a') as file:
                        file.write('=====\n')
                except Exception as e:
                    print(f"An error occurred: {e}")
                    # Continue with the code even if an error occurs
                    pass
    fid_dir = f'fidelity_test/fid_{args.data}_{args.clustering_method}_method={args.fid_method}'
    if not os.path.exists(fid_dir):
        os.makedirs(fid_dir)
    if 'nonDP' in args.clipping_mode:
        df_fid.to_csv(f'{fid_dir}/mode={args.clipping_mode}_nrows={args.n_rows}_type={args.explanation_type}_epochs={args.epochs}_channel={args.channel}.csv', index=False)
    else:
        df_fid.to_csv(f'{fid_dir}/mode={args.clipping_mode}_nrows={args.n_rows}_type={args.explanation_type}_epsilon={args.epsilon}_epochs={args.epochs}_channel={args.channel}.csv', index=False)
            
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='CIFAR10')
    parser.add_argument('--num_experiments', default=5, type=int, help='number of experiments')
    parser.add_argument('--n_rows', default=10000, type=int, help='number of rows to sample attributions')
    parser.add_argument('--channel', default=1, type=int, help='color channel')
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of epochs')
    parser.add_argument('--epsilon', default=2.0, type=float, help='target epsilon')
    parser.add_argument('--clipping_mode', default='nonDP', type=str)
    parser.add_argument('--model', default='vit_small_patch16_224', type=str) # try: resnet18
#     parser.add_argument('--dimension', type=int,default=224)
#     parser.add_argument('--origin_params', nargs='+', default=None)
    parser.add_argument('--explanation_type', default='ixg', choices=['gs', 'ig', 'ixg', 'sl', 'dl'])
#     parser.add_argument('--nsamples', type=int, default=20)
#     parser.add_argument('--experiment_no', type=int)
    parser.add_argument('--dry_run', type=lambda x: x.lower() == 'true', default=False) # whether or not we want to save data
    parser.add_argument('--verbose_flag', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--total_data_examples', type=int, default=20000)
    parser.add_argument('--clustering_method', default='kmeans', choices=['kmeans', 'hierarchical'])
    parser.add_argument('--fid_method', default='all_clusters', choices=['max', 'all_clusters'])
    args = parser.parse_args()
    main(args)