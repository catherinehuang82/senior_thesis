# in directory: cnn

import argparse
import os
import sys
import multiprocessing
sys.path.append('captumdp/')  # Add the directory containing captumdp to the Python path

# Get the absolute path to the directory containing this script
# script_directory = os.path.abspath(os.path.dirname(__file__))

# # Append the parent directory to sys.path
# parent_directory = os.path.join(script_directory, '..')
# sys.path.append(parent_directory)

print(sys.path)
from captumdp import captum as ct

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from IPython import display
import copy

import pandas as pd
from sklearn.preprocessing import normalize

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Subset
from torch import linalg as LA
from statistics import mean

import random
import importlib

def main(args):
    # only do this for experiments NOT in the first 10
    # check if explanations for this configuration have already been generated
    if True:
#     if args.experiment_no > 4:
        if 'nonDP' not in args.clipping_mode:
            variances_dir = f'lira/variances_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}'
        else:
            variances_dir = f'lira/variances_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}'

        if os.path.exists(f'{variances_dir}/{args.experiment_no}.csv'):
            print(f"The explanation variances file already exists. Terminating job.")
            return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Data
    print('==> Preparing data..')
    
    try:
        if not args.dry_run:
            debug_dir = f'lira/debug_{args.data}_{args.total_data_examples}'
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
                print(f"Created directory: {debug_dir}")
            if 'nonDP' in args.clipping_mode:
                debug_file_path = f'{debug_dir}/model={args.model}_mode={args.clipping_mode}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}.txt'
            else:
                debug_file_path = f'{debug_dir}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}.txt'
            with open(debug_file_path, 'a') as file:
                file.write(f"Preparing data for experiment {args.experiment_no}:" + "\n")
    except Exception as e:
        print(f"An error occurred: {e}")
        # Continue with the code even if an error occurs
        pass
    
    if args.data in ['SVHN','CIFAR10']:
        num_classes=10
    elif args.data in ['CIFAR100','FGVCAircraft']:
        num_classes=100
    elif args.data in ['Food101']:
        num_classes=101
    elif args.data in ['GTSRB']:
        num_classes=43
    elif args.data in ['CelebA']:
        num_classes=40
    elif args.data in ['Places365']:
        num_classes=365
    elif args.data in ['ImageNet']:
        num_classes=1000
    elif args.data in ['INaturalist']:
        num_classes=10000
    elif args.data in ['EuroSAT']:
        num_classes=10
                
    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
    ])

    if args.data in ['SVHN','Food101','GTSRB','FGVCAircraft']:
        trainset = getattr(torchvision.datasets,args.data)(root='data/', split='train', download=True, transform=transformation)
        testset = getattr(torchvision.datasets,args.data)(root='data/', split='test', download=True, transform=transformation)
    elif args.data in ['CIFAR10','CIFAR100']:
        trainset = getattr(torchvision.datasets,args.data)(root='data/', train=True, download=True, transform=transformation)
        testset = getattr(torchvision.datasets,args.data)(root='data/', train=False, download=True, transform=transformation)
    elif args.data=='CelebA':
        trainset = getattr(torchvision.datasets,args.data)(root='data/', split='train', download=False, target_type='attr', transform=transformation)
        testset = getattr(torchvision.datasets,args.data)(root='data/', split='test', download=False, target_type='attr',transform=transformation)
    elif args.data=='Places365':
        trainset = getattr(torchvision.datasets,args.data)(root='data/', split='train-standard', small=True, download=False, transform=transformation)
        testset = getattr(torchvision.datasets,args.data)(root='data/', split='val', small=True, download=False, transform=transformation)
    elif args.data=='INaturalist':
        trainset = getattr(torchvision.datasets,args.data)(root='data/', version='2021_train_mini', download=False, transform=transformation)
        testset = getattr(torchvision.datasets,args.data)(root='data/', version='2021_valid', download=False, transform=transformation)
    elif args.data=='ImageNet':
        trainset = getattr(torchvision.datasets,args.data)(root='data/', split='train', transform=transformation)
        testset = getattr(torchvision.datasets,args.data)(root='data/', split='val', transform=transformation)
        
    # Combine train and test sets
    combinedset = torch.utils.data.ConcatDataset([trainset, testset])
    
    subset_indices = list(range(args.total_data_examples))
    combinedset = Subset(combinedset, subset_indices)
    
    try:
        if not args.dry_run:
            debug_dir = f'lira/debug_{args.data}_{args.total_data_examples}'
            if 'nonDP' in args.clipping_mode:
                debug_file_path = f'{debug_dir}/model={args.model}_mode={args.clipping_mode}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}.txt'
            else:
                debug_file_path = f'{debug_dir}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}.txt'
            with open(debug_file_path, 'a') as file:
                file.write(f"combinedset size: {len(combinedset)}\n")
    except Exception as e:
        print(f"An error occurred: {e}")
        # Continue with the code even if an error occurs
        pass
    
    model = timm.create_model(args.model, num_classes=num_classes)
    
    if 'nonDP' in args.clipping_mode:
        model_dir = f'lira/model_state_dicts_{args.data}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}'
    else:
        model_dir = f'lira/model_state_dicts_{args.data}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_epochs={args.epochs}'

    if device.type == 'cuda':
        model.load_state_dict(torch.load(f'{model_dir}/{args.experiment_no}.pt'))
    else:
        model.load_state_dict(torch.load(f'{model_dir}/{args.experiment_no}.pt',  map_location=torch.device('cpu')))

    model = model.eval()
    
    if args.explanation_type == 'gs':
        explainer = ct.attr.GradientShap(model)
    elif args.explanation_type == 'ig':
        explainer = ct.attr.IntegratedGradients(model)
    elif args.explanation_type == 'ixg':
        explainer = ct.attr.InputXGradient(model)
    elif args.explanation_type == 'lrp':
        explainer = ct.attr.LRP(model)
    elif args.explanation_type == 'dl':
        explainer = ct.attr.DeepLift(model)
    elif args.explanation_type == 'sl':
        explainer = ct.attr.Saliency(model)
    elif args.explanation_type == 'gb':
        explainer = ct.attr.GuidedBackprop(model)
        
    channel1_df = pd.DataFrame(columns=range(args.dimension**2))
    channel2_df = pd.DataFrame(columns=range(args.dimension**2))
    channel3_df = pd.DataFrame(columns=range(args.dimension**2))

    if args.verbose_flag:
        print(f'Starting explanations for type = {args.explanation_type}, clipping mode = {args.clipping_mode}, eps = {args.epsilon}')
        
    try:
        if not args.dry_run:
            debug_dir = f'lira/debug_{args.data}_{args.total_data_examples}'
            if 'nonDP' in args.clipping_mode:
                debug_file_path = f'{debug_dir}/model={args.model}_mode={args.clipping_mode}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}.txt'
            else:
                debug_file_path = f'{debug_dir}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}.txt'
            with open(debug_file_path, 'a') as file:
                file.write(f"Starting explanations for experiment {args.experiment_no}, type = {args.explanation_type}, clipping mode = {args.clipping_mode}, eps = {args.epsilon}:" + "\n")
    except Exception as e:
        print(f"An error occurred: {e}")
        # Continue with the code even if an error occurs
        pass
        
    scores = np.array([])
    norms_l1 = np.array([])
    norms_l2 = np.array([])
    preds = np.array([])
    if args.explanation_type not in ['ixg', 'sl', 'gb']:
        deltas = np.array([])
            
    for i, (x, y) in tqdm(enumerate(combinedset)):
        if i % 100 == 0:
            try:
                if not args.dry_run:
                    debug_dir = f'lira/debug_{args.data}_{args.total_data_examples}'
                    if 'nonDP' in args.clipping_mode:
                        debug_file_path = f'{debug_dir}/model={args.model}_mode={args.clipping_mode}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}.txt'
                    else:
                        debug_file_path = f'{debug_dir}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}.txt'
                    with open(debug_file_path, 'a') as file:
                        file.write(f"Experiment {args.experiment_no}: {i} explanations done" + "\n")
            # slurm job ID to python code, so we have unique file names
            except Exception as e:
                print(f"An error occurred: {e}")
                # Continue with the code even if an error occurs
                pass
        inp = torch.unsqueeze(x, dim=0) # [1, 3, 224, 224]
        inp = inp.float()
        output = model.forward(inp)
        pred = torch.argmax(output, axis=-1).item()
        if args.explanation_type == 'gs':
            baseline_dist = torch.randn(inp.size()) * 0.001
            attributions, delta = explainer.attribute(inp, n_samples=5, baselines=baseline_dist, target=pred,
                           return_convergence_delta=True)
        elif args.explanation_type == 'ig':
            baseline = torch.zeros(inp.size())
            attributions, delta = explainer.attribute(inp, baseline, n_steps=args.nsamples,
                                                      target=pred, return_convergence_delta=True)
        elif args.explanation_type == 'dl':
            baseline = torch.zeros(inp.size())
            attributions, delta = explainer.attribute(inp, baseline,
                                                      target=pred, return_convergence_delta=True)
        elif args.explanation_type == 'lrp':
            baseline = torch.zeros(inp.size())
            attributions, delta = explainer.attribute(inp, target=pred, return_convergence_delta=True)
        elif args.explanation_type in ['ixg', 'sl', 'gb']:
            attributions = explainer.attribute(inp, target=pred)
#             print(f'attributions shape: {np.shape(attributions)}') # torch.Size([1, 3, 224, 224]

        if args.explanation_type not in ['ixg', 'sl', 'gb']:
            deltas = np.append(deltas, torch.mean(torch.abs(delta)).item())
        if args.experiment_no <= 4:
            channel_1 = attributions[:, 0, :, :]
            channel_2 = attributions[:, 1, :, :]
            channel_3 = attributions[:, 2, :, :]
            channel1_df.loc[len(channel1_df)] = channel_1.view(-1).detach().numpy()
            channel2_df.loc[len(channel2_df)] = channel_2.view(-1).detach().numpy()
            channel3_df.loc[len(channel3_df)] = channel_3.view(-1).detach().numpy()
            preds = np.append(preds, pred)
        scores = np.append(scores, np.sum(torch.var(attributions, dim=(2, 3), unbiased=True).detach().numpy()))
        norms_l1 = np.append(norms_l1, torch.norm(attributions, p=1).detach().item())
        norms_l2 = np.append(norms_l2, torch.norm(attributions, p=2).detach().item())

    if args.verbose_flag:
        print(f'Explanations done for type = {args.explanation_type}, clipping mode = {args.clipping_mode}, eps = {args.epsilon}, type={args.explanation_type}, nsamples={args.nsamples}')
    
    if not args.dry_run:
        scores_df = pd.DataFrame({f'{args.experiment_no}': scores})
        norms_l1_df = pd.DataFrame({f'{args.experiment_no}': norms_l1})
        norms_l2_df = pd.DataFrame({f'{args.experiment_no}': norms_l2})
        preds_df = pd.DataFrame({f'{args.experiment_no}': preds})
        if args.explanation_type not in ['ixg', 'sl', 'gb']:
            deltas_df = pd.DataFrame({f'{args.experiment_no}': deltas})
        if 'nonDP' not in args.clipping_mode:
            variances_dir = f'lira/variances_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}'
            norms_l1_dir = f'lira/norms_l1_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}'
            norms_l2_dir = f'lira/norms_l2_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}'
            deltas_dir = f'lira/deltas_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}'
            attributions_dir = f'lira/attributions_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}'
            preds_dir = f'lira/preds_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}'
        else:
            variances_dir = f'lira/variances_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}'
            norms_l1_dir = f'lira/norms_l1_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}'
            norms_l2_dir = f'lira/norms_l2_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}'
            deltas_dir = f'lira/deltas_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}'
            attributions_dir = f'lira/attributions_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}'
            preds_dir = f'lira/preds_{args.data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}'
        if not os.path.exists(variances_dir):
            os.makedirs(variances_dir)
            print(f'Created directory: {variances_dir}')
        if not os.path.exists(norms_l1_dir):
            os.makedirs(norms_l1_dir)
            print(f'Created directory: {norms_l1_dir}')
        if not os.path.exists(norms_l2_dir):
            os.makedirs(norms_l2_dir)
            print(f'Created directory: {norms_l2_dir}')
        if args.explanation_type not in ['ixg', 'sl', 'gb'] and not os.path.exists(deltas_dir):
            os.makedirs(deltas_dir)
            print(f'Created directory: {deltas_dir}')
        if not os.path.exists(attributions_dir):
            os.makedirs(attributions_dir)
            print(f'Created directory: {attributions_dir}')
        if not os.path.exists(preds_dir):
            os.makedirs(preds_dir)
            print(f'Created directory: {preds_dir}')
        if args.experiment_no <= 4:
            if not os.path.exists(f'{attributions_dir}/channel1'):
                os.makedirs(f'{attributions_dir}/channel1')
            if not os.path.exists(f'{attributions_dir}/channel2'):
                os.makedirs(f'{attributions_dir}/channel2')
            if not os.path.exists(f'{attributions_dir}/channel3'):
                os.makedirs(f'{attributions_dir}/channel3')
        scores_df.to_csv(f'{variances_dir}/{args.experiment_no}.csv', index=False)
        norms_l1_df.to_csv(f'{norms_l1_dir}/{args.experiment_no}.csv', index=False)
        norms_l2_df.to_csv(f'{norms_l2_dir}/{args.experiment_no}.csv', index=False)
        # make sure we only save data for the DP CIFAR-10 explanations
#         if args.experiment_no <= 4:
        if args.experiment_no <= 4:
            preds_df.to_csv(f'{preds_dir}/{args.experiment_no}.csv', index=False)
            channel1_df = channel1_df.round(5)
            channel2_df = channel2_df.round(5)
            channel3_df = channel3_df.round(5)
            channel1_df.to_csv(f'{attributions_dir}/channel1/{args.experiment_no}.csv', index=False)
            channel2_df.to_csv(f'{attributions_dir}/channel2/{args.experiment_no}.csv', index=False)
            channel3_df.to_csv(f'{attributions_dir}/channel3/{args.experiment_no}.csv', index=False)
        if args.explanation_type not in ['ixg', 'sl', 'gb']:
            deltas_df.to_csv(f'{deltas_dir}/{args.experiment_no}.csv', index=False)
         
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='GSTRB')
    parser.add_argument('--bs', default=1000, type=int, help='batch size')
    parser.add_argument('--epochs', default=5, type=int,
                        help='number of epochs')
    parser.add_argument('--mini_bs', type=int, default=50)
    parser.add_argument('--epsilon', default=2.0, type=float, help='target epsilon')
    parser.add_argument('--clipping_mode', default='BK-MixOpt', type=str)
    parser.add_argument('--model', default='vit_small_patch16_224', type=str) # try: resnet18
    parser.add_argument('--dimension', type=int,default=224)
    parser.add_argument('--origin_params', nargs='+', default=None)
    
    parser.add_argument('--explanation_type', default='ixg', choices=['gs', 'ig', 'ixg', 'lrp', 'dl', 'sl', 'gb'])
    parser.add_argument('--nsamples', type=int, default=20)
    parser.add_argument('--experiment_no', type=int)
    parser.add_argument('--dry_run', type=lambda x: x.lower() == 'true', default=False) # whether or not we want to save data
    parser.add_argument('--verbose_flag', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--total_data_examples', type=int, default=20000)
    args = parser.parse_args()
    main(args)
    
 