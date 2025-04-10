import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from medmnist import INFO
from medmnist.dataset import PathMNIST

def get_medmnist_loaders(batch_size=64):
    data_flag = 'pathmnist'
    info = INFO[data_flag]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    train_dataset = PathMNIST(split='train', transform=transform, download=True)
    test_dataset = PathMNIST(split='test', transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
