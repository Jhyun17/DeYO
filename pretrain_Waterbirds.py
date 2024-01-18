# Example
# python pretrain_Waterbirds.py --root_dir ./data --dset_dir Waterbirds

import os
import argparse
parser = argparse.ArgumentParser(description='Waterbirds pretrain')
parser.add_argument('--root_dir', default=None, help='path to data')
parser.add_argument('--dset_dir', default=None, help='name of dataset directory')
parser.add_argument('--gpu', default='0', type=str, help='gpu index for training.')
parser.add_argument('--seed', default=2024, type=int, help='seed for initializing training.')
parser.add_argument('--batch_size', default=64, type=int, help='batch_size for training.')
parser.add_argument('--test_batch_size', default=256, type=int, help='batch_size for test.')
parser.add_argument('--workers', default=2, type=int, help='num_workers for train loader.')
parser.add_argument('--if_shuffle', default=1, type=int, help='shuffle for training.')
parser.add_argument('--max_epochs', default=200, type=int, help='epochs for training.')
parser.add_argument('--interval', default=10, type=int, help='intervals for saving.')
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

SEED = args.seed
deterministic=True

import random
import torch
import numpy as np

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
        
import wget
import tarfile
import h5py
import csv
from dataset.waterbirds_dataset import WaterbirdsDataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pickle
import models.Res as Resnet
from tqdm import tqdm

def download_dataset(root_dir):
    # url from Official github https://github.com/kohpangwei/group_DRO
    url = 'https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz'
    filename = os.path.join(root_dir, 'waterbird_complete95_forest2water2.tar.gz')
    if not os.path.isfile(filename):
        print('Downloading the tar file')
        wget.download(url, out=root_dir)
    else:
        print(filename, 'already exists')


def make_dataset(dset_dir):
    print('Dataset path:', dset_dir)
    meta_path = os.path.join(dset_dir, 'metadata.csv')
    print('Metadata path:', meta_path)

    hdf5_file = os.path.join(dset_dir, 'waterbirds_dataset.h5py')

    split_cnt=[-1,-1,-1]

    with h5py.File(hdf5_file, 'w') as hf:

        f = open(meta_path, 'r')
        rdr = csv.reader(f)
        cnt = 0
        for line in tqdm(rdr):
            # skip line 1
            if cnt == 0:
                cnt += 1
                continue

            file_path = os.path.join(dset_dir, line[1])
            x = Image.open(file_path).convert('RGB')
            x = x.resize((224, 224))
            split_val = int(line[3])
            if split_val==0:
                split = 'train'
            elif split_val==1:
                split = 'val'
            elif split_val==2:
                split = 'test'
            else:
                raise Exception("Not accurate split")

            split_cnt[split_val] += 1

            h5py_path = os.path.join('Waterbirds', split, str(split_cnt[split_val]))
            hf[h5py_path] = x
            hf[h5py_path].attrs['img_id'] = int(line[0])
            hf[h5py_path].attrs['img_filename'] = line[1]
            hf[h5py_path].attrs['y'] = int(line[2])
            hf[h5py_path].attrs['split'] = split_val
            hf[h5py_path].attrs['place'] = int(line[4])
            hf[h5py_path].attrs['place_filename'] = line[5]

            cnt += 1
        f.close()

def eval_waterbirds(net, val_loader, epoch_cnt):
    correct_count = 0
    total_count = 0
    for labeled_batch in (val_loader):
        data = labeled_batch
        x, y = data[0], data[1]
        x = x.cuda()
        y = y.cuda()
        
        logits = net(x)
        
        correct_count += (logits.argmax(dim=1) == y).sum().item()
        total_count += len(logits)
    print('Acc at epoch {}: {:.2f}%'.format(epoch_cnt, correct_count/total_count*100))
    return 0


def test_waterbirds(net, test_loader, epoch_cnt):
    correct_count = [0,0,0,0]
    total_count = [0,0,0,0]
    for labeled_batch in (test_loader):
        data = labeled_batch
        x, y = data[0], data[1]
        place = data[2]['place']
        x = x.cuda()
        y = y.cuda()
        place = place.cuda()
        
        logits = net(x)
        
        group = 2*y + place #0: land+land, 1: land+sea, 2: sea+land, 3: sea+sea
        TFtensor = (logits.argmax(dim=1) == y)
        
        for group_idx in range(4):
            correct_count[group_idx] += TFtensor[group==group_idx].sum().item()
            total_count[group_idx] += len(TFtensor[group==group_idx])
        
    print('Acc at epoch {}: LL: {:.2f}%, LS: {:.2f}%, SL: {:.2f}%, SS: {:.2f}%,'.format(
        epoch_cnt, 
        correct_count[0]/total_count[0]*100,
        correct_count[1]/total_count[1]*100,
        correct_count[2]/total_count[2]*100,
        correct_count[3]/total_count[3]*100
    )
         )
    return 0

if __name__ == '__main__':
    assert args.root_dir is not None
    assert args.dset_dir is not None

    transform=transforms.Compose(
        [transforms.ToTensor(), 
         # transforms.Resize((224, 224)),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    root_dir = args.root_dir
    dset_dir = os.path.join(root_dir, args.dset_dir)
    filename = os.path.join(root_dir, 'waterbird_complete95_forest2water2.tar.gz')
    if not os.path.isfile(os.path.join(dset_dir, 'metadata.csv')):
        download_dataset(root_dir)
        print('Extracting data from tar file')
        ap = tarfile.open(filename)
        ap.extractall(root_dir)
        ap.close()
        os.rename(os.path.join(root_dir, 'waterbird_complete95_forest2water2'), dset_dir)
        
    assert os.path.isfile(os.path.join(dset_dir, 'metadata.csv'))==True
        
    dataset_file = os.path.join(dset_dir, 'waterbirds_dataset.h5py')
    if not os.path.isfile(dataset_file):
        make_dataset(dset_dir)
    else:
        print('Dataset file already exists')

    train_dataset = WaterbirdsDataset(dataset_file, 'train', transform)
    val_dataset = WaterbirdsDataset(dataset_file, 'val', transform)
    test_dataset = WaterbirdsDataset(dataset_file, 'test', transform)
    
    
    if not os.path.exists(os.path.join(dset_dir, 'waterbirds_pretrained_model.pickle')):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                                   shuffle=args.if_shuffle, num_workers=args.workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size,
                                                 shuffle=False, num_workers=args.workers,
                                                 pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, 
                                                  shuffle=False, num_workers=args.workers,
                                                  pin_memory=True)


        net = Resnet.__dict__['resnet50'](pretrained=True)
        net.fc = torch.nn.Linear(in_features=net.fc.in_features, out_features=2)
        net = net.cuda()
        net.train()

        # Train loop
        CEloss = torch.nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
        optimizer.zero_grad()
        # net.train()
        for epochs in tqdm(range(args.max_epochs)):
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                                       shuffle=True, num_workers=2,
                                                       pin_memory=True)
            for _, labeled_batch in enumerate((train_loader)):
                data = labeled_batch
                x, y = data[0], data[1]
                x = x.cuda()
                y = y.cuda()

                logits = net(x)

                loss = CEloss(logits, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if epochs % args.interval == 0:
                with torch.no_grad():
                    net.eval()
                    test_waterbirds(net, test_loader, epochs)
                    net.train()

        with torch.no_grad():
            print('Final accuracy!')
            net.eval()
            with open(file=os.path.join(dset_dir, 'waterbirds_pretrained_model.pickle'), mode='wb') as f:
                pickle.dump(net, f)
            test_waterbirds(net, test_loader, epochs)
    else:
        print('Pretrained model already exists.')
