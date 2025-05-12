

import torch
from torch import nn 
import numpy as np

from torch.profiler import record_function



import torchvision, os
from torchvision.transforms import v2

vanilla = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
])

cifar10_train = torchvision.datasets.CIFAR10(os.path.dirname(__file__), train = True,  transform=vanilla, download=True)
cifar10_test  = torchvision.datasets.CIFAR10(os.path.dirname(__file__), train = False, transform=vanilla, download=True)

wd = os.getcwd()
os.chdir(os.path.dirname(__file__))
os.chdir("../..")
training_datasets = [torch.utils.data.TensorDataset(torch.load(f"augmented_datasets/data{i}.pt"), torch.load(f"augmented_datasets/labels{i}.pt")) for i in range(20)]
os.chdir(wd)

from torcheval.metrics.functional import multiclass_accuracy as accuracy
# mixup = v2.MixUp(num_classes=10)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# print("using device: ", device)






def train_epoch(model, batch_size, opt, epoch):

    model.train(True)
    
    training_loader = torch.utils.data.DataLoader(training_datasets[np.random.randint(0, len(training_datasets))], batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    # loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss_fn = nn.CrossEntropyLoss()
    
    train_loss = 0.0
    
    for X, labels in training_loader:

        X, labels = X.to(device).float(), labels.to(device).float()
        opt.zero_grad()
        
        with record_function("forward"):
            logits = model(X)
            loss = loss_fn(logits, labels)
        
        with record_function("backward"):
            loss.backward()
            
        opt.step()
        
        train_loss += loss.item()
    
    return train_loss / len(training_loader)
    


def initialize_batches(batch_size):
    train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=min(batch_size, 50000))
    test_loader  = torch.utils.data.DataLoader(cifar10_test,  batch_size=min(batch_size, 10000))
    
    train_batches = [train_x for (train_x, _) in train_loader]
    test_batches  = [test_x  for (test_x, _)  in test_loader]

    return train_batches, test_batches


def all_logits(model, data):
    
    all_logits = []
    
    for X in data:
        all_logits.append(model(X.to(device)))
    
    return torch.cat(all_logits)
    
    
def evaluate(model, train_labels, test_labels, batch_size):
    if not hasattr(evaluate, "train_batches"):
        evaluate.train_batches, evaluate.test_batches = initialize_batches(batch_size)
    
    train_batches = evaluate.train_batches
    test_batches = evaluate.test_batches
    
    loss = nn.CrossEntropyLoss()
    
    logits = all_logits(model, train_batches)
    train_accuracy = accuracy(logits, train_labels)
    avg_train_loss = loss(logits, train_labels)

    logits = all_logits(model, test_batches)
    test_accuracy = accuracy(logits, test_labels)
    avg_test_loss = loss(logits, test_labels)
    
    return train_accuracy, avg_train_loss, test_accuracy, avg_test_loss
    
    

# def evaluate(model, train_X, test_X, train_labels, test_labels):
#     logits = model(train_X)
#     train_accuracy = accuracy(logits, train_labels)
#     avg_train_loss = nn.CrossEntropyLoss()(logits, train_labels)

#     logits = model(test_X)
#     test_accuracy = accuracy(logits, test_labels)
#     avg_test_loss = nn.CrossEntropyLoss()(logits, test_labels)
    
#     return train_accuracy, avg_train_loss, test_accuracy, avg_test_loss
    
    
    
    
def train(model, nr_epochs, lr, batch_size, weight_decay, max_bs):
    
    with record_function("setup"):
    
        model = model.to(device)
        
        _, test_labels = next(iter(torch.utils.data.DataLoader(cifar10_test, batch_size=10000)))
        test_labels = test_labels.to(device)
        
        _, train_labels = next(iter(torch.utils.data.DataLoader(cifar10_train, batch_size=50000)))
        train_labels = train_labels.to(device)
        
        
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=nr_epochs, eta_min=0)
        
        # early stopping criterium
        window_size = 100 # if no significant improvement over this many epochs
        significant_improvement = 0.03 # improvement is significant if (new-old)/old > 0.03
        
        training_losses = []
        training_accuracies = []
        test_losses = []
        test_accuracies = []
        
        model.train(False)
        with torch.no_grad():
            train_accuracy, avg_train_loss, test_accuracy, avg_test_loss = evaluate(model, train_labels, test_labels, max_bs)
            
        training_losses.append(avg_train_loss.item())
        training_accuracies.append(train_accuracy.item())
        test_losses.append(avg_test_loss.item())
        test_accuracies.append(test_accuracy.item())


    with record_function("training"):

        for epoch in range(nr_epochs):
            
            with record_function("Train"):
                model.train(True)
                avg_train_loss = train_epoch(model, batch_size, opt, epoch)
                scheduler.step()
            
            with record_function("Eval"):
                model.train(False)
                with torch.no_grad():
                    train_accuracy, _, test_accuracy, avg_test_loss =  evaluate(model, train_labels, test_labels, max_bs)
            
            # print(epoch, ": ", train_accuracy.item(), ", ", test_accuracy.item())
            
            training_losses.append(avg_train_loss)
            training_accuracies.append(train_accuracy.item())
            test_losses.append(avg_test_loss.item())
            test_accuracies.append(test_accuracy.item())
            
            # # check for early stopping due to convergence
            # if epoch > 2*window_size:
            #     old = max(training_accuracies[:-window_size])
            #     new = max(training_accuracies[-window_size:])
            #     if new/old - 1 < significant_improvement:
            #         break
            
            # stop if training has collapsed (probably learning rate too high...)
            if epoch > window_size and (training_accuracies[-1] < 0.5*max(training_accuracies) or training_accuracies[-1] < 0.2):
                break
            

    return training_losses, training_accuracies, test_losses, test_accuracies
        # print(epoch, ": avg training loss: ", avg_train_loss, "train accuracy: ", round(train_accuracy.item(), 4), " avg test loss: ", round(avg_test_loss.item(), 4), " test accuracy: ", round(test_accuracy.item(), 4))
    


