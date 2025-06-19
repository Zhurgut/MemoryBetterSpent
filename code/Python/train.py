

import torch
from torch import nn 
import numpy as np

from torcheval.metrics import Perplexity
import time
import math


device = torch.device("cuda") # else ERROR
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# print("using device: ", device)






def train_epoch(model, opt, loss_fn, new_training_loader, is_gpt2):

    model.train(True)
    
    training_loader = new_training_loader()
    total_loss = 0.0
    i = 0

    if is_gpt2:

        for batch in training_loader:

            opt.zero_grad()

            input_ids = batch[0].to(device)

            outputs = model(input_ids, labels=input_ids.clone())
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
                
            opt.step()

            i += 1
    else:

        for X, labels in training_loader:

            X, labels = X.to(device).float(), labels.to(device) # .float() ????
            opt.zero_grad()

            logits = model(X)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            loss.backward()
                
            opt.step()

            i += 1

    return total_loss / i
    


def all_logits(model, data):
    
    all_logits = []
    
    for X in data:
        all_logits.append(model(X.to(device)))
    
    return torch.cat(all_logits)
    
    


def evaluate(model, train_batches, test_batches, train_labels, test_labels, loss_fn, metric_fn, compute_train_accuracy, is_gpt2):

    if is_gpt2:

        stride = loss_fn # hack, need to get the stride here all the way from dataset-loading
        train_ppl = 1000

        if compute_train_accuracy:

            metric_fn.reset()

            for batch in train_batches:

                input_ids = batch
                labels = input_ids.clone()
                labels[:, :-stride] = -100

                outputs = model(input_ids, labels=labels)

                logits = outputs.logits

                metric_fn.update(logits[:, :-1, :], labels[:, 1:])
            
            train_ppl = metric_fn.compute()

        metric_fn.reset()
        test_loss = 0.0

        for batch in test_batches:

            input_ids = batch
            labels = input_ids.clone()
            labels[:, :-stride] = -100

            outputs = model(input_ids, labels=labels)
            test_loss += outputs.loss.item()

            logits = outputs.logits

            metric_fn.update(logits[:, :-1, :], labels[:, 1:])

        avg_test_loss = test_loss / len(test_batches)
        test_ppl = metric_fn.compute()

        return torch.Tensor([train_ppl]), torch.Tensor([test_ppl]), torch.Tensor([avg_test_loss])

    else:
        train_accuracy = torch.Tensor([-1])
        if compute_train_accuracy:
            logits = all_logits(model, train_batches)
            train_accuracy = metric_fn(logits, train_labels)

        logits = all_logits(model, test_batches)
        test_accuracy = metric_fn(logits, test_labels)
        test_loss = loss_fn(logits, test_labels)
        
        return train_accuracy, test_accuracy, test_loss
    
    
    
    
def train(model, dataset, nr_epochs, lr, weight_decay, early_stopping, lr_decay, is_gpt2=False):

    _, _, _, train_batches, test_batches, train_labels, test_labels, training_loader_fn, loss_fn, metric_fn = dataset

    model = model.to(device)
    
    if not is_gpt2:
        test_labels = test_labels.to(device)
        train_labels = train_labels.to(device)
    if is_gpt2:
        metric_fn = metric_fn.to(device)
        test_batches  = [t.to(device) for t in test_batches]
        train_batches = [t.to(device) for t in train_batches]

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

    def evaluate_model(compute_train_accuracy):
        return evaluate(model, train_batches, test_batches, train_labels, test_labels, loss_fn, metric_fn, compute_train_accuracy, is_gpt2)

    def record_training_stats(epoch, train_loss):
        model.train(False)
        with torch.no_grad():
            train_accuracy, test_accuracy, test_loss = evaluate_model(epoch % 10 == 0)
            
        training_losses.append(train_loss)
        training_accuracies.append(train_accuracy.item())
        test_losses.append(test_loss.item())
        test_accuracies.append(test_accuracy.item())
        times.append(time.time())

    #     print(training_losses, ", ", training_accuracies, ", ", test_losses, ", ", test_accuracies)

    # print("rec", time.time())

    record_training_stats(-1, 0.0) # first entry before training
    # print("*")

    for epoch in range(1, nr_epochs+1):
        # print("training", time.time())

        train_loss = train_epoch(model, opt, loss_fn, training_loader_fn, is_gpt2)

        if math.isnan(train_loss):
            times[0] = -1 # little incidator to say that there was a nan, probably shoul introduce a proper flag or something for this sort of thing... TODO
            break
        # print("recording", time.time())

        record_training_stats(epoch, train_loss)
        # print("*")

        # check for early stopping due to convergence
        if early_stopping and epoch > 2*window_size and not is_gpt2: # gpt2 measures perplexity, which is decreasing as performance improves, as opposed to accuracy... TODO fix later
            old = max(training_accuracies[:-window_size])
            new = max(training_accuracies[-window_size:])
            if new/old - 1 < significant_improvement:
                break
        
        if lr_decay:
            lr_scheduler.step()


    return training_losses, training_accuracies, test_losses, test_accuracies, times
        # print(epoch, ": avg training loss: ", avg_train_loss, "train accuracy: ", round(train_accuracy.item(), 4), " avg test loss: ", round(avg_test_loss.item(), 4), " test accuracy: ", round(test_accuracy.item(), 4))
    


