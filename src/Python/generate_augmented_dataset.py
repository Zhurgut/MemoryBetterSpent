
import torch
import torchvision, os
from torchvision.transforms import v2


augment = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float16, scale=True),
    v2.RandomResizedCrop((32, 32), scale=(0.7, 1.0), ratio=(0.8, 1.2)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
    
])

cifar10_train_augmented = torchvision.datasets.CIFAR10(os.path.dirname(__file__), train = True,  transform=augment, download=True)

mixup = v2.MixUp(num_classes=10)

training_loader = torch.utils.data.DataLoader(cifar10_train_augmented, batch_size=50000, shuffle=True, num_workers=4, persistent_workers=True)



wd = os.getcwd()
os.chdir(os.path.dirname(__file__))
os.chdir("../..")
os.makedirs("augmented_datasets", exist_ok=True)

for i in range(20):
    print(i, "/ ", 20)
    for X, labels in training_loader:
        X, labels = mixup(X, labels)
        torch.save(X, f"augmented_datasets/data{i}.pt")
        torch.save(labels, f"augmented_datasets/labels{i}.pt")

os.chdir(wd)