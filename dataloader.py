###########################################################################
# Computer vision - Binary neural networks demo software by HyperbeeAI.   #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. main@shallow.ai #
###########################################################################
import os
import torch
import torchvision
from torchvision import transforms
import random
import numpy as np

class ai85_normalize:
    def __init__(self, act_8b_mode):
        self.act_8b_mode = act_8b_mode

    def __call__(self, img):
        if(self.act_8b_mode):
            return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127)
        return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127).div(128.)
        
def load_cifar100(batch_size=128, num_workers=1, shuffle=True, act_8b_mode=False):
    """
    Maxim's data augmentation: 4 pixels are padded on each side, and a 32x32 crop is randomly sampled
    from the padded image or its horizontal flip.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ai85_normalize(act_8b_mode=act_8b_mode)
    ])

    train_dataset = torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        ai85_normalize(act_8b_mode=act_8b_mode)
    ])

    test_dataset = torchvision.datasets.CIFAR100(root='data', train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, test_loader

def load_cifar100_p(batch_size=128, num_workers=1, shuffle=True, act_8b_mode=False, partial=20.0):
    """
    Maxim's data augmentation: 4 pixels are padded on each side, and a 32x32 crop is randomly sampled
    from the padded image or its horizontal flip.
    """
    dataset_size=50000
    # Do an error check since we added the parameter, it's not relayed to torch or something
    if((partial > 100.0) or (partial < 0.0)):
        print('')
        print('Argument partial can only be between 0 and 100')
        print('Exiting.')
        print('')
        sys.exit()

    # Train dataset transform # disabled augmentation to compare optimization performance
    train_transform = transforms.Compose([
#        transforms.RandomCrop(32, padding=4),
#        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ai85_normalize(act_8b_mode=act_8b_mode)
    ])

    # Load complete training dataset to use as a base for partial dataset
    train_dataset = torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=train_transform)
    # Get subset of training dataset
    num_elements_to_load = np.round(50000*partial/100.0)
    indices_from_dataset = createRandomSortedList(int(num_elements_to_load), 0, dataset_size)
    partial_dataset        = torch.utils.data.Subset(train_dataset, indices_from_dataset)
    print('Loaded',partial,'% of the training dataset, corresponding to', len(indices_from_dataset) ,'image/label tuples')
    batch_loader = torch.utils.data.DataLoader(partial_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return batch_loader

def createRandomSortedList(num, start = 1, end = 100):
    arr = []
    tmp = random.randint(start, end)
    for x in range(num):
        while tmp in arr:
            tmp = random.randint(start, end)
        arr.append(tmp)
    arr.sort()
    return arr
