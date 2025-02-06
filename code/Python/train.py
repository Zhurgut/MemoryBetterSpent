

import torch
from torch import nn 



import torchvision, os
from torchvision.transforms import v2

vanilla = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
])

cifar10_train = torchvision.datasets.CIFAR10(os.path.dirname(__file__), train = True,  transform=vanilla, download=True)
cifar10_test  = torchvision.datasets.CIFAR10(os.path.dirname(__file__), train = False, transform=vanilla, download=True)


training_datasets = [torch.utils.data.TensorDataset(torch.load(f"augmented_datasets/data{i}.pt"), torch.load(f"augmented_datasets/labels{i}.pt")) for i in range(10)]


from torcheval.metrics.functional import multiclass_accuracy as accuracy
# mixup = v2.MixUp(num_classes=10)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("using device: ", device)






def train_epoch(model, batch_size, opt, epoch):
    print(epoch)

    model.train(True)
    
    training_loader = torch.utils.data.DataLoader(training_datasets[epoch % 10], batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    
    loss_fn = nn.CrossEntropyLoss()
    
    train_loss = 0.0
    
    for X, labels in training_loader:

        X, labels = X.to(device).float(), labels.to(device).float()
        
        opt.zero_grad()
        
        logits = model(X)
        loss = loss_fn(logits, labels)
        
        loss.backward()
        opt.step()
        
        train_loss += loss.item()
    
    return train_loss / len(training_loader)
    
    
    
def train(model, nr_epochs, lr, batch_size):
    
    model = model.to(device)
    
    test_X, test_labels = next(iter(torch.utils.data.DataLoader(cifar10_test, batch_size=10000)))
    test_X, test_labels = test_X.to(device), test_labels.to(device)
    
    train_X, train_labels = next(iter(torch.utils.data.DataLoader(cifar10_train, batch_size=50000)))
    train_X, train_labels = train_X.to(device), train_labels.to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # opt  = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    
    training_losses = []
    training_accuracies = []
    test_losses = []
    test_accuracies = []
    
    model.train(False)
    with torch.no_grad():
    
        logits = model(train_X)
        train_accuracy = accuracy(logits, train_labels)
        avg_train_loss = nn.CrossEntropyLoss()(logits, train_labels)

        logits = model(test_X)
        test_accuracy = accuracy(logits, test_labels)
        avg_test_loss = nn.CrossEntropyLoss()(logits, test_labels)
        
    training_losses.append(avg_train_loss.item())
    training_accuracies.append(train_accuracy.item())
    test_losses.append(avg_test_loss.item())
    test_accuracies.append(test_accuracy.item())

    for epoch in range(nr_epochs):
        
        model.train(True)
        avg_train_loss = train_epoch(model, batch_size, opt, epoch)
        
        model.train(False)
        with torch.no_grad():
            
            logits = model(train_X)
            train_accuracy = accuracy(logits, train_labels)
        
            logits = model(test_X)
            test_accuracy = accuracy(logits, test_labels)
            
            avg_test_loss = nn.CrossEntropyLoss()(logits, test_labels)
        
        training_losses.append(avg_train_loss)
        training_accuracies.append(train_accuracy.item())
        test_losses.append(avg_test_loss.item())
        test_accuracies.append(test_accuracy.item())
        
    return training_losses, training_accuracies, test_losses, test_accuracies
        # print(epoch, ": avg training loss: ", avg_train_loss, "train accuracy: ", round(train_accuracy.item(), 4), " avg test loss: ", round(avg_test_loss.item(), 4), " test accuracy: ", round(test_accuracy.item(), 4))
    


