
import os
import csv

import torch, torchvision
from torch import nn
from torchvision.transforms import v2
from torch.utils.data import TensorDataset, DataLoader

from datasets import load_dataset
import transformers
from transformers import GPT2TokenizerFast, GPT2Tokenizer

from torcheval.metrics.functional import multiclass_accuracy as accuracy
from torcheval.metrics import Perplexity

import numpy as np



"""
the provided metric function needs to be some kind of score, I mean something that's increasing...
"""


def simple(training_batch_size, max_batch_size):
    

    folder_path = os.path.join(os.path.dirname(__file__), "..", "..", "datasets")

    X_train = []
    y_train = []
    with open(os.path.join(folder_path, "simple_train.csv"), newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            x_val, y_val, label = float(row[0]), float(row[1]), int(float(row[2]))
            X_train.append([x_val, y_val])
            y_train.append(label)

    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.long)
    
    X_test = []
    y_test = []
    with open(os.path.join(folder_path, "simple_test.csv"), newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            x_val, y_val, label = float(row[0]), float(row[1]), int(float(row[2]))
            X_test.append([x_val, y_val])
            y_test.append(label)

    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.long)
    

    train_ds = TensorDataset(X_train, y_train)
    test_ds  = TensorDataset(X_test, y_test)

    def training_loader():
        return DataLoader(train_ds, batch_size=X_train.shape[0], num_workers=0, pin_memory=True)


    in_dim, out_dim = 2, 4
    train_batches, test_batches = [X_train], [X_test]
    train_labels,  test_labels  = y_train, y_test
    
    return (
        # input and output dimensions of the model
        in_dim, out_dim, 
        
        -1, # image dim, not used for this dataset
        
        # evaluation data
        train_batches, test_batches,
        train_labels, test_labels,

        # training dataset generator
        training_loader,

        # loss, eval metric
        nn.CrossEntropyLoss(), accuracy
    )


def cifar10(training_batch_size, max_batch_size):

    # for model evaluation, load the unaugmented dataset

    to_tensor = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])

    cifar10_train = torchvision.datasets.CIFAR10(os.path.dirname(__file__), train = True,  transform=to_tensor, download=True)
    cifar10_test  = torchvision.datasets.CIFAR10(os.path.dirname(__file__), train = False, transform=to_tensor, download=True)

    train_loader = DataLoader(cifar10_train, batch_size=min(max_batch_size, 50000))
    test_loader  = DataLoader(cifar10_test,  batch_size=min(max_batch_size, 10000))
    
    train_batches = [train_x for (train_x, _) in train_loader]
    test_batches  = [test_x  for (test_x, _)  in test_loader]

    _, train_labels = next(iter(DataLoader(cifar10_train, batch_size=50000))) 
    _, test_labels = next(iter(DataLoader(cifar10_test, batch_size=10000)))
    




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
        return DataLoader(training_datasets[np.random.randint(0, len(training_datasets))], batch_size=training_batch_size, shuffle=True, num_workers=0, pin_memory=True)




    in_dim, out_dim = 32*32*3, 10
    
    return (
        # input and output dimensions of the model
        in_dim, out_dim, 32, 
        
        # evaluation data
        train_batches, test_batches,
        train_labels, test_labels,

        # training dataset generator
        training_loader,

        # loss, eval metric
        nn.CrossEntropyLoss(), accuracy
    )



# for tiny imagenet, top level definitions so everything is available to all worker processes
mixup200 = v2.MixUp(num_classes=200)
cutmix200 = v2.CutMix(num_classes=200)
cutmix_or_mixup200 = v2.RandomChoice([mixup200, cutmix200])

def to_tuple(batch):
    X = torch.stack([item["tensor"] for item in batch])
    y = torch.tensor([item["label"] for item in batch])
    return X, y

def to_tuple_with_mixup(batch):
    # X = torch.stack([item["tensor"] for item in batch])
    X = torch.stack([item["tensor"] for item in batch])
    y = torch.tensor([item["label"] for item in batch])
    X, y = cutmix_or_mixup200(X, y)
    return X, y

def tiny_imagenet(training_batch_size, max_batch_size):

    ds_folder = os.path.join(os.path.dirname(__file__), "../..", "tiny_imagenet")
    os.makedirs(ds_folder, exist_ok=True)
    
    raw_datasets = load_dataset("zh-plus/tiny-imagenet", cache_dir=ds_folder)

    to_tensor = v2.Compose([
        v2.ToImage(),
        v2.RGB(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    augment = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RGB(),
        v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.RandomErasing()
    ])

    def apply_tf(x):
        x["tensor"] = to_tensor(x["image"])
        return x

    def apply_augment(x):
        x["tensor"] = [augment(img) for img in x["image"]]
        return x

    
    train_data = raw_datasets["train"].with_transform(apply_tf, output_all_columns=True)
    test_data  = raw_datasets["valid"].with_transform(apply_tf, output_all_columns=True)

    train_augmented = raw_datasets["train"].with_transform(apply_augment, output_all_columns=True)

    # train_data.set_format("torch", ["tensor", "label"])
    # test_data.set_format("torch", ["tensor", "label"])
    # train_augmented.set_format("torch", ["tensor", "label"])

    num_workers = 8

    train_loader = DataLoader(train_data, batch_size=min(max_batch_size, 100000), shuffle=False, collate_fn=to_tuple, num_workers=num_workers, pin_memory=False)
    test_loader =  DataLoader(test_data,  batch_size=min(max_batch_size, 10000),  shuffle=False, collate_fn=to_tuple, num_workers=num_workers, pin_memory=False)

    train_batches = [train_x for (train_x, _) in train_loader]
    test_batches  = [test_x  for (test_x, _)  in test_loader]

    train_labels = torch.cat([  train_label for (_, train_label) in DataLoader(train_data, shuffle=False, collate_fn=to_tuple, batch_size=1000, num_workers=num_workers) ], dim=0)
    _, test_labels  = next(iter(DataLoader(test_data,  shuffle=False, collate_fn=to_tuple, batch_size=10000)))

    def new_train_loader():
        return DataLoader(train_augmented, batch_size=training_batch_size, shuffle=True, collate_fn=to_tuple_with_mixup, num_workers=num_workers, pin_memory=True)


    in_dim, out_dim = 64*64*3, 200
    

    return (
        # input and output dimensions of the model
        in_dim, out_dim, 64,
        
        # evaluation data
        train_batches, test_batches,
        train_labels, test_labels,

        # training dataset generator
        new_train_loader,

        # loss, eval metric
        nn.CrossEntropyLoss(label_smoothing=0.1), accuracy
    )






def wikitext2(training_batch_size, max_batch_size):

    ds_folder = os.path.join(os.path.dirname(__file__), "../..", "wikitext2")
    os.makedirs(ds_folder, exist_ok=True)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = 10**10

    train_ds = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=ds_folder, split="train")
    train_ds = train_ds.filter(lambda x: x["text"].strip() != "")
    all_train_text = "\n\n".join(train_ds["text"])

    train_ids = tokenizer(all_train_text, return_tensors="pt").input_ids.squeeze()  # 2.4M int64 


    test_ds = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=ds_folder, split="test")
    test_ds = test_ds.filter(lambda x: x["text"].strip() != "")
    all_test_text = "\n\n".join(test_ds["text"])

    test_ids = tokenizer(all_test_text, return_tensors="pt").input_ids.squeeze()  # 250k int64 


    def new_train_loader():
        sample_len = 128
        samples = []
        i = torch.randint(0, sample_len, (1,)).item()
        while i - sample_len < train_ids.size(0):
            samples.append(train_ids[i:i+sample_len])
            i += sample_len + torch.randint(-4, 4, (1,)).item()
        
        all_samples = torch.stack(samples)
        ds = TensorDataset(all_samples)
        
        return DataLoader(ds, batch_size=training_batch_size, shuffle=True)
    
    # to compute perplexity, use sliding window with size 1024 and stride _
    stride = 512
    samples = []
    i = 0
    while i + 1024 < train_ids.size(0):
        samples.append(train_ids[i:i+1024])
        i += stride
    all_samples = torch.stack(samples)
    train_batches = [batch[0] for batch in DataLoader(TensorDataset(all_samples), batch_size=max_batch_size)]

    samples = []
    i = 0
    while i + 1024 < test_ids.size(0):
        samples.append(test_ids[i:i+1024])
        i += stride
    all_samples = torch.stack(samples)
    test_batches = [batch[0] for batch in DataLoader(TensorDataset(all_samples), batch_size=max_batch_size)]


    return (
        None, None, None,

        train_batches, test_batches, 
        None, None,

        new_train_loader,

        stride, Perplexity(ignore_index=-100)
    )

