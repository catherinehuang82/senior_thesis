from fastDP import PrivacyEngine

import os
import gc
import numpy as np
import pandas as pd
import torch
import torchvision
torch.manual_seed(2)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Dataset, TensorDataset, Subset
import timm
from opacus.validators import ModuleValidator
from opacus.accountants.utils import get_noise_multiplier
from tqdm import tqdm
from statistics import mean
import warnings; warnings.filterwarnings("ignore")


'''Train CIFAR10/CIFAR100 with PyTorch.'''
def main(args):
    if 'nonDP' in args.clipping_mode:
        model_dir = f'lira/model_state_dicts_{args.cifar_data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}'
    else:
        model_dir = f'lira/model_state_dicts_{args.cifar_data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_epochs={args.epochs}'
    if os.path.isfile(f'{model_dir}/{args.experiment_no}.pt'):
        print("We already ran this job.")
        return None
            
    if args.clipping_mode not in ['nonDP','BK-ghost', 'BK-MixGhostClip', 'BK-MixOpt','nonDP-BiTFiT','BiTFiT']:
        print("Mode must be one of 'nonDP','BK-ghost', 'BK-MixGhostClip', 'BK-MixOpt','nonDP-BiTFiT','BiTFiT'")
        return None

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    ####### data ######
    print(f'Epsilon = {args.epsilon}')
    print('==> Preparing data..')

    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.dimension),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
    ])

    if args.cifar_data=='CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='data/', train=True, download=True, transform=transformation)
        testset = torchvision.datasets.CIFAR10(root='data/', train=False, download=True, transform=transformation)
    elif args.cifar_data=='CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='data/', train=True, download=True, transform=transformation)
        testset = torchvision.datasets.CIFAR100(root='data/', train=False, download=True, transform=transformation)
    else:
        return "Must specify datasets as CIFAR10 or CIFAR100"
         
    print('==> Splitting data..')
    
    # Combine train and test sets
    combinedset = torch.utils.data.ConcatDataset([trainset, testset])
    
    subset_indices = list(range(args.total_data_examples))
    combinedset = Subset(combinedset, subset_indices)

    # Get the total number of examples
    total_examples = len(combinedset)

    # Calculate the number of examples you want (half in this case)
    half_examples = total_examples // 2

    # Use random_split to split the dataset into two parts (half and the rest)
    inset, outset = random_split(combinedset, [half_examples, total_examples - half_examples])

    # Get the indices of examples in each half
    in_indices = inset.indices
    out_indices = [i for i in range(total_examples) if i not in in_indices]

    binary_vector = np.zeros(total_examples)
    binary_vector[in_indices] = 1

    trainloader = torch.utils.data.DataLoader(
        inset, batch_size=args.mini_bs, shuffle=True, num_workers=4)

    testloader = torch.utils.data.DataLoader(
        outset, batch_size=100, shuffle=False, num_workers=4)
    
    ####### uncomment this part after we get train accuracies ########
    if 'nonDP' not in args.clipping_mode:
        lira_dir = f'lira/indices_{args.cifar_data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_epochs={args.epochs}'
    else:
        lira_dir = f'lira/indices_{args.cifar_data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}'
    df = pd.DataFrame({f'{args.experiment_no}': np.array(binary_vector)})
    if not os.path.exists(lira_dir):
        # If it doesn't exist, create the directory
        os.makedirs(lira_dir)
    df.to_csv(f'{lira_dir}/{args.experiment_no}.csv', index=False)
    ###################################################################
    
    ####### training #######

    n_acc_steps = args.bs // args.mini_bs # gradient accumulation steps

    # Model
    print('==> Building model..', args.model,'; BatchNorm is replaced by GroupNorm. Mode: ', args.clipping_mode)
    net = timm.create_model(args.model,pretrained=True,num_classes=int(args.cifar_data[5:]))    
    net = ModuleValidator.fix(net); net=net.to(device)
    
    # Save the pre-trained model
#     if not args.dry_run:
#         model_state_dict = net.state_dict()
#         for key in list(model_state_dict.keys()):
#             model_state_dict[key.replace('_module.', '')] = model_state_dict.pop(key)
#         model_dir = f'model_state_dicts_fastdp_{args.cifar_data}'
#         if not os.path.exists(model_dir):
#             # If it doesn't exist, create the directory
#             os.makedirs(model_dir)
#             print(f"Created directory: {model_dir}")
#         torch.save(model_state_dict, f'{model_dir}/pretrained={args.model}.pt')

    print('Number of total parameters: ', sum([p.numel() for p in net.parameters()]))
    print('Number of trainable parameters: ', sum([p.numel() for p in net.parameters() if p.requires_grad]))
    
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    if 'BiTFiT' in args.clipping_mode: # not needed for DP-BiTFiT but use here for safety
        for name,param in net.named_parameters():
            if '.bias' not in name:
                param.requires_grad_(False)

    # Privacy engine
    if 'nonDP' not in args.clipping_mode:
        sigma=get_noise_multiplier(
                target_epsilon = args.epsilon,
                target_delta = 1e-5,
                sample_rate = args.bs/len(trainset),
                epochs = args.epochs,
            )

        if 'BK' in args.clipping_mode:
            clipping_mode=args.clipping_mode[3:]
        else:
            clipping_mode='ghost'
        privacy_engine = PrivacyEngine(
            net,
            batch_size=args.bs,
            sample_size=len(trainset),
            noise_multiplier=sigma,
            epochs=args.epochs,
            clipping_mode=clipping_mode,
            origin_params=args.origin_params,#['patch_embed.proj.bias'],
        )
        privacy_engine.attach(optimizer)        

        
    def train(epoch):

        net.train()
        train_loss = 0
        correct = 0
        total = 0

   
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
                optimizer.step()
                optimizer.zero_grad()
                
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Epoch: ', epoch, len(trainloader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if epoch == args.epochs - 1:
            if not args.dry_run:
                accuracies_dir = f'lira/accuracies_train_{args.cifar_data}_{args.total_data_examples}'
                if not os.path.exists(accuracies_dir):
                    os.makedirs(accuracies_dir)
                    print(f"Created directory: {accuracies_dir}")
                if 'nonDP' in args.clipping_mode:
                    accuracies_file_path = f'{accuracies_dir}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}.txt'
                else:
                    accuracies_file_path = f'{accuracies_dir}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_epochs={args.epochs}.txt'
                with open(accuracies_file_path, 'a') as file:
                    file.write(str(100.*correct/total) + '\n')
                    print(f"File updated with the integer value: {100.*correct/total}")

    def test(epoch):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print('Epoch: ', epoch, len(testloader), 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            ####### uncomment this part after we get train accuracies ########
            if epoch == args.epochs - 1:
                if not args.dry_run:
                    accuracies_dir = f'lira/accuracies_{args.cifar_data}_{args.total_data_examples}'
                    if not os.path.exists(accuracies_dir):
                        os.makedirs(accuracies_dir)
                        print(f"Created directory: {accuracies_dir}")
                    if 'nonDP' in args.clipping_mode:
                        accuracies_file_path = f'{accuracies_dir}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}.txt'
                    else:
                        accuracies_file_path = f'{accuracies_dir}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_epochs={args.epochs}.txt'
                    with open(accuracies_file_path, 'a') as file:
                        file.write(str(100.*correct/total) + '\n')
                        print(f"File updated with the integer value: {100.*correct/total}")
            #####################################################################

    for epoch in range(args.epochs):
        train(epoch)
        test(epoch)
        
    if not args.dry_run:
        model_state_dict = net.state_dict()
        for key in list(model_state_dict.keys()):
            model_state_dict[key.replace('_module.', '')] = model_state_dict.pop(key)
        if 'nonDP' in args.clipping_mode:
            model_dir = f'lira/model_state_dicts_{args.cifar_data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}'
        else:
            model_dir = f'lira/model_state_dicts_{args.cifar_data}_{args.total_data_examples}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_epochs={args.epochs}'
        if not os.path.exists(model_dir):
            # If it doesn't exist, create the directory
            os.makedirs(model_dir)
            print(f"Created directory: {model_dir}")
        torch.save(model_state_dict, f'{model_dir}/{args.experiment_no}.pt')
    # garbage collection
    net.cpu()
    del net
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--epochs', default=5, type=int,
                        help='number of epochs')
    parser.add_argument('--bs', default=1000, type=int, help='batch size')
    parser.add_argument('--mini_bs', type=int, default=50)
    parser.add_argument('--epsilon', default=2.0, type=float, help='target epsilon')
    parser.add_argument('--clipping_mode', default='BK-MixOpt', type=str)
    parser.add_argument('--model', default='beit_base_patch16_224.in22k_ft_in22k_in1k', type=str) # try: vit_small_patch16_224
    parser.add_argument('--cifar_data', type=str, default='CIFAR100')
    parser.add_argument('--dimension', type=int,default=224)
    parser.add_argument('--origin_params', nargs='+', default=None)
    parser.add_argument('--dry_run', default=False)
    parser.add_argument('--experiment_no', type=int)
    parser.add_argument('--total_data_examples', type=int, default=20000)

    args = parser.parse_args()
    
    torch.manual_seed(args.experiment_no)

    main(args)
