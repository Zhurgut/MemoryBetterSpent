

import torch
from torch import nn 
import numpy as np


import time


device = torch.device("cuda") # else ERROR
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# print("using device: ", device)






def train_epoch(model, opt, loss_fn, new_training_loader):

    model.train(True)
    
    training_loader = new_training_loader()

    for X, labels in training_loader:

        X, labels = X.to(device).float(), labels.to(device) # .float() ????
        opt.zero_grad()

        logits = model(X)
        loss = loss_fn(logits, labels)

        loss.backward()
            
        opt.step()

    


def all_logits(model, data):
    
    all_logits = []
    
    for X in data:
        all_logits.append(model(X.to(device)))
    
    return torch.cat(all_logits)
    
    


def evaluate(model, train_batches, test_batches, train_labels, test_labels, loss_fn, metric_fn):
    
    logits = all_logits(model, train_batches)
    train_accuracy = metric_fn(logits, train_labels)
    train_loss = loss_fn(logits, train_labels)

    logits = all_logits(model, test_batches)
    test_accuracy = metric_fn(logits, test_labels)
    test_loss = loss_fn(logits, test_labels)
    
    return train_accuracy, train_loss, test_accuracy, test_loss
    
    
    
    
def train(model, dataset, nr_epochs, lr, weight_decay, early_stopping, lr_decay):

    _, _, _, train_batches, test_batches, train_labels, test_labels, training_loader_fn, loss_fn, metric_fn = dataset

    model = model.to(device)
    
    test_labels = test_labels.to(device)
    train_labels = train_labels.to(device)


    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=nr_epochs, eta_min=0)
    

    # early stopping criterium
    window_size = 100 # if no significant improvement over this many epochs
    significant_improvement = 0.01 # improvement is significant if (new-old)/old > 0.01
    

    training_losses = []
    training_accuracies = []
    test_losses = []
    test_accuracies = []
    times = []

    def evaluate_model():
        return evaluate(model, train_batches, test_batches, train_labels, test_labels, loss_fn, metric_fn)

    def record_training_stats():
        model.train(False)
        with torch.no_grad():
            train_accuracy, train_loss, test_accuracy, test_loss = evaluate_model()
            
        training_losses.append(train_loss.item())
        training_accuracies.append(train_accuracy.item())
        test_losses.append(test_loss.item())
        test_accuracies.append(test_accuracy.item())
        times.append(time.time())

    #     print(training_losses, ", ", training_accuracies, ", ", test_losses, ", ", test_accuracies)

    # print("rec", time.time())

    record_training_stats() # first entry before training
    # print("*")

    for epoch in range(nr_epochs):
        # print("training", time.time())

        train_epoch(model, opt, loss_fn, training_loader_fn)
        # print("recording", time.time())

        record_training_stats()
        # print("*")

        # check for early stopping due to convergence
        if early_stopping and epoch > 2*window_size:
            old = max(training_accuracies[:-window_size])
            new = max(training_accuracies[-window_size:])
            if new/old - 1 < significant_improvement:
                break
        
        if lr_decay:
            lr_scheduler.step()


    return training_losses, training_accuracies, test_losses, test_accuracies, times
        # print(epoch, ": avg training loss: ", avg_train_loss, "train accuracy: ", round(train_accuracy.item(), 4), " avg test loss: ", round(avg_test_loss.item(), 4), " test accuracy: ", round(test_accuracy.item(), 4))
    


