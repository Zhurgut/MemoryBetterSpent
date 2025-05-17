
import os

import torch, torchvision
from torch import nn
from torchvision.transforms import v2

from torcheval.metrics.functional import multiclass_accuracy as accuracy

import numpy as np

"""
the provided metric function needs to be some kind of score, I mean something that's increasing...
"""



def cifar10(training_batch_size, max_batch_size):

    # for model evaluation, load the unaugmented dataset

    to_tensor = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])

    cifar10_train = torchvision.datasets.CIFAR10(os.path.dirname(__file__), train = True,  transform=to_tensor, download=True)
    cifar10_test  = torchvision.datasets.CIFAR10(os.path.dirname(__file__), train = False, transform=to_tensor, download=True)

    train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=min(max_batch_size, 50000))
    test_loader  = torch.utils.data.DataLoader(cifar10_test,  batch_size=min(max_batch_size, 10000))
    
    train_batches = [train_x for (train_x, _) in train_loader]
    test_batches  = [test_x  for (test_x, _)  in test_loader]

    _, train_labels = next(iter(torch.utils.data.DataLoader(cifar10_train, batch_size=50000))) 
    _, test_labels = next(iter(torch.utils.data.DataLoader(cifar10_test, batch_size=10000)))
    




    # for training, load the datasets with precomputed augmentations

    wd = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    os.chdir("../..")

    # TODO call it myself..., automagically detect length, not hardcode 20
    "make sure you have called generate_augmented_dataset.py before"

    training_datasets = [torch.utils.data.TensorDataset(torch.load(f"augmented_datasets/data{i}.pt"), torch.load(f"augmented_datasets/labels{i}.pt")) for i in range(20)]

    os.chdir(wd)




    # function that returns a dataset to be used for training in the next epoch

    def training_loader():
        return torch.utils.data.DataLoader(training_datasets[np.random.randint(0, len(training_datasets))], batch_size=training_batch_size, shuffle=True, num_workers=0, pin_memory=True)




    in_dim, out_dim = 32*32*3, 10
    
    return (
        # input and output dimensions of the model
        in_dim, out_dim,
        
        # evaluation data
        train_batches, test_batches,
        train_labels, test_labels,

        # training dataset generator
        training_loader,

        # loss, eval metric
        nn.CrossEntropyLoss(), accuracy
    )