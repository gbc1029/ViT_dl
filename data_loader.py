"""Data loading utilities for CIFAR-10"""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10


def get_cifar10_dataloaders(batch_size=64, num_workers=4, data_dir='./data'):
    """
    Get CIFAR-10 data loaders with appropriate transforms.
    
    Args:
        batch_size: Batch size for training and testing
        num_workers: Number of workers for data loading
        data_dir: Directory to store/load CIFAR-10 dataset
    
    Returns:
        train_loader, test_loader: Training and testing data loaders
    """
    
    # Mean and std for CIFAR-10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Download and load datasets
    train_dataset = CIFAR10(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    test_dataset = CIFAR10(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=16)
    
    # Check a batch
    images, labels = next(iter(train_loader))
    print(f"Train batch - Images shape: {images.shape}, Labels shape: {labels.shape}")
    
    # Dataset info
    print(f"\nTrain samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Classes: {train_loader.dataset.classes}")
