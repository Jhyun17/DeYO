# from https://github.com/reiinakano/invariant-risk-minimization/blob/master/invariant_risk_minimization_colored_mnist.ipynb
# Example
# python pretrain_ColoredMNIST.py --root_dir ./data --dset_dir ColoredMNIST
import os
import argparse
parser = argparse.ArgumentParser(description='Waterbirds pretrain')
parser.add_argument('--root_dir', default=None, help='path to data')
parser.add_argument('--dset_dir', default='ColoredMNIST', help='name of dataset directory')
parser.add_argument('--gpu', default='0', type=str, help='gpu index for training.')
parser.add_argument('--seed', default=2024, type=int, help='seed for initializing training.')
parser.add_argument('--batch_size', default=64, type=int, help='batch_size for training.')
parser.add_argument('--test_batch_size', default=1000, type=int, help='batch_size for test.')
parser.add_argument('--workers', default=2, type=int, help='num_workers for train loader.')
parser.add_argument('--if_shuffle', default=1, type=int, help='shuffle for training.')
parser.add_argument('--max_epochs', default=20, type=int, help='epochs for training.')
parser.add_argument('--interval', default=10, type=int, help='intervals for saving.')
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import pickle

import matplotlib.pyplot as plt

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
import torchvision
from torchvision import transforms
import torchvision.datasets.utils as dataset_utils

from dataset.ColoredMNIST_dataset import ColoredMNIST

def test_model(model, device, test_loader, set_name="test set"):
    model.eval()
    CELoss = torch.nn.CrossEntropyLoss()
    test_loss = 0
    correct_count = torch.tensor([0,0,0,0])
    total_count = torch.tensor([0,0,0,0])
    with torch.no_grad():
        for data, target, color in test_loader:
            data, target, color = data.to(device), target.to(device), color.to(device)
            group = 2*target + color
            output = model(data)
            test_loss += CELoss(output, target).sum().item()  # sum up batch loss
            TFtensor = (output.argmax(dim=1)==target)
            for group_idx in range(4):
                correct_count[group_idx] += TFtensor[group_idx==group].sum().item()
                total_count[group_idx] += len(TFtensor[group_idx==group])

    test_loss /= len(test_loader.dataset)
    accs = correct_count / total_count * 100

    print('\nPerformance on {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
      set_name, test_loss, correct_count.sum().item(), total_count.sum().item(),
      correct_count.sum().item()/total_count.sum().item()*100))
    print('Group accuracy  => RSmall: {:.2f}, GSmall: {:.2f}, RBig: {:.2f}, GBig: {:.2f}'.format(accs[0].item(), 
                                                                            accs[1].item(), 
                                                                            accs[2].item(), 
                                                                            accs[3].item()))
    print('Detailed counts => RSmall: {}/{}, GSmall: {}/{}, RBig: {}/{}, GBig: {}/{}'.format(
        correct_count[0], total_count[0], 
        correct_count[1], total_count[1], 
        correct_count[2], total_count[2], 
        correct_count[3], total_count[3]))

    return correct_count, total_count


def erm_train(model, device, train_loader, optimizer, epoch, args):
    model.train()
    CELoss = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target, group) in enumerate(train_loader):
        data, target, group = data.to(device), target.to(device), group.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = CELoss(output, target)
        loss.backward()
        optimizer.step()
    print('Train Epoch: {}\tLoss: {:.6f}'.format(
        epoch, loss.item()))


def train_and_test_erm(model, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}
    all_train_loader = torch.utils.data.DataLoader(
      ColoredMNIST(root=args.root_dir, env='all_train',# flip=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                     ])),
      batch_size=args.batch_size, shuffle=args.if_shuffle, **kwargs)

    test_loader = torch.utils.data.DataLoader(
      ColoredMNIST(root=args.root_dir, env='test',# flip=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                   ])),
      batch_size=args.test_batch_size, shuffle=False, **kwargs)

    if not os.path.exists(os.path.join(args.root_dir, 'ColoredMNIST_model.pickle')):
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

        for epoch in range(args.max_epochs):
            erm_train(model, device, all_train_loader, optimizer, epoch, args)
            if epoch % args.interval == 0 or epoch==args.max_epochs-1:
                test_model(model, device, all_train_loader, set_name='train set')
                test_model(model, device, test_loader)

if __name__=='__main__':    
    assert args.root_dir is not None
    assert args.dset_dir is not None
    
    args.root_dir = os.path.join(args.root_dir, args.dset_dir)
    
    SEED = args.seed
    deterministic=True

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False        
    
    model = torchvision.models.resnet18(pretrained=True)
    num_classes=2
    model.fc=nn.Linear(model.fc.in_features,num_classes,bias=True)
    model.eval()
    train_and_test_erm(model, args)

    if not os.path.exists(os.path.join(args.root_dir, 'ColoredMNIST_model.pickle')):
        with open(file=os.path.join(args.root_dir, 'ColoredMNIST_model.pickle'), mode='wb') as f:
            pickle.dump(model, f)
    else:
        print('Pretrained model already exists.')