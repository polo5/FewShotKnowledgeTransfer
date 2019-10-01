import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os

def get_loaders(args, indices):

    if args.dataset == 'SVHN':
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        train_dataset = datasets.SVHN(args.dataset_path, split='train', download=True, transform=transform_train)
        test_dataset = datasets.SVHN(args.dataset_path, split='test', download=True, transform=transform_test)

    elif args.dataset == 'CIFAR10':
        mean = (0.4914, 0.4822, 0.4465) if args.dataset == 'CIFAR10' else (0.5071, 0.4867, 0.4408)
        std = (0.2023, 0.1994, 0.2010) if args.dataset == 'CIFAR10' else (0.2675, 0.2565, 0.2761)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        if args.dataset == 'CIFAR10':
            train_dataset = datasets.CIFAR10(args.dataset_path, train=True, download=True, transform=transform_train)
            test_dataset = datasets.CIFAR10(args.dataset_path, train=False, download=True, transform=transform_test)
        elif args.dataset == 'CIFAR100':
            train_dataset = datasets.CIFAR100(args.dataset_path, train=True, download=True, transform=transform_train)
            test_dataset = datasets.CIFAR100(args.dataset_path, train=False, download=True, transform=transform_test)

    else:
        raise NotImplementedError

    if args.n_images_per_class < 0:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
            drop_last=False, num_workers=args.workers, pin_memory=False)
    else:
        if indices is None:
            indices = get_indices_for_n_images_per_class(n=args.n_images_per_class, dataset=args.dataset, datasets_path=args.datasets_path)
        else:
            print('\n{} indices for {} images per class already provided, sum={}'.format(len(indices), args.n_images_per_class, sum(indices)))

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(indices),
            drop_last=False, num_workers=args.workers, pin_memory=False)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=args.workers, pin_memory=False)

    return train_loader, test_loader, indices


class DatasetWithIdx(Dataset):
    def __init__(self, dataset, datasets_path):
        dataset_path = os.path.join(datasets_path, dataset)
        if dataset == 'SVHN':
            self.dataset = datasets.SVHN(root=dataset_path, split='train', download=True)
        elif dataset == 'CIFAR10':
            self.dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True)
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        _, target = self.dataset[index]
        return target, index

    def __len__(self):
        return len(self.dataset)


def get_indices_for_n_images_per_class(n, dataset, datasets_path):
    n_classes = 1000 if dataset=='ImageNet' else (100 if dataset=='CIFAR100' else 10)
    dataset = DatasetWithIdx(dataset, datasets_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=True) # get a different subset for each seed
    indices = []
    counts = np.zeros(n_classes)

    print('\n Fetching indices for {} images per class...\n'.format(n))

    for target, index in loader:
        if counts[int(target)] < n:
            indices.append(int(index))
            counts[int(target)] += 1
        if np.all(counts == n):
            break

    print('\n{} indices fetched! sum = {}\n'.format(len(indices), sum(indices)))

    return indices


if __name__=='__main__':
    import os
    from utils.helpers import *
    import math
    from time import time
    from torchvision.utils import make_grid

    datasets_path = "/home/paul/Datasets/Pytorch"
    dataset = 'CIFAR100'
    dataset_path = os.path.join(datasets_path, dataset)

    ## get_CIFAR10_indices_for_n_images_per_class
    n_classes = 10 if dataset == 'CIFAR10' else 100
    from torch.utils.data.sampler import SubsetRandomSampler
    t0 = time()
    indices = get_indices_for_n_images_per_class(n=100, dataset=dataset, datasets_path=datasets_path)
    print('Time taken: {} s'.format(time()-t0))
    print('Length of indices: ', len(indices))

    # check for duplicates
    if len(indices) != len(set(indices)):
        print('\nTHERE ARE DUPLICATES!\n')
    else:
        print('\nNo duplicates found in indices\n')


    ######## Subsample those indices

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    if dataset=='CIFAR10':
        train_dataset = datasets.CIFAR10(dataset_path, train=True, download=True, transform=transform_train)
    elif dataset == 'CIFAR100':
        train_dataset = datasets.CIFAR100(dataset_path, train=True, download=True, transform=transform_train)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, sampler=SubsetRandomSampler(indices),
        batch_size=100, drop_last=False, num_workers=4)


    totals = np.zeros(n_classes)

    for x,y in train_loader:
        totals += np.array([len(y[y==i]) for i in range(n_classes)])

        #grid = make_grid(x, nrow=10, normalize=True)
        #plot_image(grid)

    print(totals)