import os
import gc
import numpy as np
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython import display
import random
import scipy.stats
from sklearn import metrics
from sklearn.metrics import roc_curve
import pandas as pd
import matplotlib.pyplot as plt

import timm
import torchvision
from torchvision.transforms.functional import gaussian_blur
from torch.utils.data import Subset
import torch.nn.functional as F

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    df_fid = pd.DataFrame(columns=['experiment_no', 'epsilon', 'sigma', 'k', 'm', 'gap'])
    
    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
    ])
    
    if args.data=='CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='data/', train=True, download=True, transform=transformation)
        testset = torchvision.datasets.CIFAR10(root='data/', train=False, download=True, transform=transformation)
    elif args.data=='CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='data/', train=True, download=True, transform=transformation)
        testset = torchvision.datasets.CIFAR100(root='data/', train=False, download=True, transform=transformation)
    elif args.data=='SVHN':
        trainset = getattr(torchvision.datasets,args.data)(root='data/', split='train', download=True, transform=transformation)
        testset = getattr(torchvision.datasets,args.data)(root='data/', split='test', download=True, transform=transformation)
    else:
        raise ValueError("Must specify datasets as CIFAR10 or CIFAR100")

    # Combine train and test sets
    combinedset = torch.utils.data.ConcatDataset([trainset, testset])

#     subset_indices = list(range(args.n_rows))
#     combinedset = Subset(combinedset, subset_indices)

    if args.data in ['SVHN','CIFAR10']:
        num_classes=10
    elif args.data in ['CIFAR100']:
        num_classes=100
    
    for experiment_no in tqdm(range(args.num_experiments)):
        exp_no = str(experiment_no)
        net = timm.create_model(args.model, num_classes=num_classes)
        
        ##### load model #####
        if 'nonDP' in args.clipping_mode:
            model_dir = f'lira/model_state_dicts_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}'
            indices_path = f'lira/indices_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}/{exp_no}.csv'
        else:
            model_dir = f'lira/model_state_dicts_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_epochs={args.epochs}'
            indices_path = f'lira/indices_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_epochs={args.epochs}/{exp_no}.csv'

        if device.type == 'cuda':
            net.load_state_dict(torch.load(f'{model_dir}/{exp_no}.pt'))
        else:
            net.load_state_dict(torch.load(f'{model_dir}/{exp_no}.pt',  map_location=torch.device('cpu')))

        net = net.eval()
        
        print('loaded model')
        
        ##### load attributions #####
        
        if 'nonDP' in args.clipping_mode:
            df_path1 = f'lira/attributions_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_type={args.explanation_type}_nsamples=20_epochs={args.epochs}/channel1/{exp_no}.csv' 
            df_path2 = f'lira/attributions_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_type={args.explanation_type}_nsamples=20_epochs={args.epochs}/channel2/{exp_no}.csv'
            df_path3 = f'lira/attributions_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_type={args.explanation_type}_nsamples=20_epochs={args.epochs}/channel3/{exp_no}.csv'
        else:
            df_path1 = f'lira/attributions_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_type={args.explanation_type}_nsamples=20_epochs={args.epochs}/channel1/{exp_no}.csv'
            df_path2 = f'lira/attributions_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_type={args.explanation_type}_nsamples=20_epochs={args.epochs}/channel2/{exp_no}.csv'
            df_path3 = f'lira/attributions_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_type={args.explanation_type}_nsamples=20_epochs={args.epochs}/channel3/{exp_no}.csv'
        
        print('reading in dataframes...')
        df_channel1 = pd.read_csv(df_path1,
                          nrows=args.n_rows)
        df_channel2 = pd.read_csv(df_path2,
                                  nrows=args.n_rows)
        df_channel3 = pd.read_csv(df_path3,
                                  nrows=args.n_rows)
        df_indices = pd.read_csv(indices_path, nrows=args.n_rows)
        df_channel1 = df_channel1[df_indices[exp_no] == 0]
        df_channel2 = df_channel2[df_indices[exp_no] == 0]
        df_channel3 = df_channel3[df_indices[exp_no] == 0]
        
        mask = df_indices[exp_no] == 0
        # Use the boolean mask to filter the indices
        subset_indices = mask.index[mask]
        combinedset_test = Subset(combinedset, subset_indices)
        print(f'df_channel1 size: {df_channel1.size}')
        
        print('read in dataframes')
#         print(df_attr.shape)

        original_shape = (1, 224, 224)
        all_norms = []
        for i, (x, y) in tqdm(enumerate(combinedset_test)):
            inp = torch.unsqueeze(x, dim=0) # [1, 3, 224, 224]
        #     print(inp.size())
            inp = inp.float()
            original_output = net.forward(inp)
            original_output_proba = F.softmax(original_output, dim=1)
#             orig_probs.append(torch.max(original_output_proba))
            for j in range(args.m):
                noisy_inp = torch.zeros(inp.size())
                for channel in range(3):
                    if channel == 0:
                        df_channel = df_channel1
                    elif channel == 1:
                        df_channel = df_channel2
                    elif channel == 2:
                        df_channel == df_channel3
                    flat_attr_channel = torch.tensor(df_channel.iloc[i].values)
            #         print(flat_attr_channel.size()) # [50176]
                    top_k_indices = torch.topk(flat_attr_channel.abs(), args.k, dim=0).indices
            #         print(top_k_indices)

                    noise = torch.randn_like(flat_attr_channel) * args.sigma  # Generate noise with mean 0 and standard deviation 1

                    # Set the noise at the indices specified in 'topk_indices' to zero
                    noise[top_k_indices] = 0

                    clean_channel = inp[:, channel, :, :].view(-1)
                    noisy_channel = clean_channel + noise
            #         noisy_channel = torch.clamp(noisy_channel, 0, 1)
                    noisy_channel_reshaped = noisy_channel.view(original_shape)
            #         print(f'flat_attr_channel: {clean_channel}')
            #         print(f'noisy_channel: {noisy_channel}')

                    noisy_inp[:, channel, :, :] = noisy_channel_reshaped
                noisy_output = net.forward(noisy_inp)
                noisy_output_proba = F.softmax(noisy_output, dim=1)
                all_norms = np.append(all_norms, torch.norm(original_output_proba - noisy_output_proba, p=1).item())
                del df_channel

        row = {
            'experiment_no': experiment_no,
            'epsilon': args.epsilon,
            'sigma': args.sigma,
            'k': args.k,
            'm': args.m,
            'gap': np.mean(all_norms)
        }
        df_fid.loc[len(df_fid)] = list(row.values())
        del net
        gc.collect()
        torch.cuda.empty_cache()
        
    fid_dir = f'perturbation_gap_test/{args.data}_sigma={args.sigma}_k={args.k}_m={args.m}'
    if not os.path.exists(fid_dir):
        os.makedirs(fid_dir)
    if not args.dry_run:
        if 'nonDP' in args.clipping_mode:
            df_fid.to_csv(f'{fid_dir}/mode={args.clipping_mode}_nrows={args.n_rows}_type={args.explanation_type}_epochs={args.epochs}.csv', index=False)
        else:
            df_fid.to_csv(f'{fid_dir}/mode={args.clipping_mode}_nrows={args.n_rows}_type={args.explanation_type}_epsilon={args.epsilon}_epochs={args.epochs}.csv', index=False)
            
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='CIFAR10')
    parser.add_argument('--num_experiments', default=5, type=int, help='number of experiments')
    parser.add_argument('--n_rows', default=5000, type=int, help='number of rows to sample attributions')
    parser.add_argument('--epochs', default=30, type=int,
                        help='number of epochs')
    parser.add_argument('--epsilon', default=2.0, type=float, help='target epsilon')
    parser.add_argument('--clipping_mode', default='nonDP', type=str)
    parser.add_argument('--model', default='vit_small_patch16_224', type=str) # try: resnet18
    parser.add_argument('--explanation_type', default='ixg', choices=['gs', 'ig', 'ixg', 'sl', 'dl'])
    parser.add_argument('--dry_run', type=lambda x: x.lower() == 'true', default=False) # whether or not we want to save data
    parser.add_argument('--verbose_flag', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--total_data_examples', type=int, default=20000)
    parser.add_argument('--sigma', type=float, default=0.1) # 0.25, 0.5
    parser.add_argument('--k', type=int, default=50) # number of most important features [50, 100]
    parser.add_argument('--m', type=int, default=10) # number of runs per example [5, 10]
    
    args = parser.parse_args()
    main(args)