"""Data loading utilities for CIFAR-10/CIFAR-100 with advanced augmentation"""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
import random


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class MixUp(object):
    """Apply MixUp augmentation to a batch of images and labels.
    
    Args:
        alpha (float): Beta distribution parameter for MixUp.
        num_classes (int): Number of classes for one-hot encoding.
    """
    def __init__(self, alpha=0.2, num_classes=10):
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, batch_imgs, batch_labels):
        """
        Args:
            batch_imgs (Tensor): Batch of images, shape (B, C, H, W).
            batch_labels (Tensor): Batch of labels, shape (B,).
        Returns:
            mixed_imgs, mixed_labels: Mixed batch.
        """
        if self.alpha <= 0:
            return batch_imgs, batch_labels
            
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch_imgs.size(0)
        index = torch.randperm(batch_size)
        
        mixed_imgs = lam * batch_imgs + (1 - lam) * batch_imgs[index, :]
        
        # One-hot encode labels
        labels_onehot = torch.zeros(batch_size, self.num_classes, device=batch_imgs.device)
        labels_onehot.scatter_(1, batch_labels.unsqueeze(1), 1)
        
        mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[index, :]
        
        return mixed_imgs, mixed_labels


def get_train_transform(dataset='cifar10', randaug_enabled=False, randaug_n=2, randaug_m=9,
                        use_cutout=False, cutout_length=16, use_color_jitter=False,
                        color_jitter_brightness=0.2, color_jitter_contrast=0.2,
                        color_jitter_saturation=0.2, color_jitter_hue=0.1,
                        use_random_rotation=False, rotation_degrees=15,
                        use_random_affine=False, affine_translate=0.1):
    """
    Build advanced train transform pipeline for CIFAR datasets.
    
    Args:
        dataset: 'cifar10' or 'cifar100'
        randaug_enabled: Enable RandAugment
        randaug_n: RandAugment number of transformations
        randaug_m: RandAugment magnitude (1-10)
        use_cutout: Enable Cutout augmentation
        cutout_length: Cutout patch length (pixels)
        use_color_jitter: Enable ColorJitter
        color_jitter_brightness: Brightness jitter range
        color_jitter_contrast: Contrast jitter range
        color_jitter_saturation: Saturation jitter range
        color_jitter_hue: Hue jitter range
        use_random_rotation: Enable random rotation
        rotation_degrees: Max rotation degrees
        use_random_affine: Enable random affine
        affine_translate: Max translation (fraction of image size)
    
    Returns:
        transforms.Compose: Training transform pipeline
    """
    # Get dataset-specific normalization
    if dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
    else:  # cifar100
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    
    transform_list = []
    
    # 1. Basic geometric augmentation (always applied)
    transform_list.append(transforms.RandomCrop(32, padding=4))
    transform_list.append(transforms.RandomHorizontalFlip())
    
    # 2. Random rotation (optional)
    if use_random_rotation:
        transform_list.append(transforms.RandomRotation(rotation_degrees))
    
    # 3. Random affine (optional, includes rotation + translation + scaling)
    if use_random_affine:
        transform_list.append(transforms.RandomAffine(
            degrees=rotation_degrees if not use_random_rotation else 0,
            translate=(affine_translate, affine_translate)
        ))
    
    # 4. RandAugment (optional, replaces other color augmentations)
    if randaug_enabled:
        # Check if RandAugment is available (torchvision >= 0.9)
        rand_augment = getattr(transforms, 'RandAugment', None)
        auto_augment = getattr(transforms, 'AutoAugment', None)
        
        if rand_augment is not None:
            transform_list.append(rand_augment(num_ops=randaug_n, magnitude=randaug_m))
        elif auto_augment is not None:
            # Fallback to AutoAugment if RandAugment not available
            transform_list.append(auto_augment(transforms.AutoAugmentPolicy.CIFAR10))
        else:
            print("Warning: RandAugment/AutoAugment not available, falling back to ColorJitter")
            use_color_jitter = True
    else:
        # 5. Color Jitter (optional, not used if RandAugment enabled)
        if use_color_jitter:
            transform_list.append(transforms.ColorJitter(
                brightness=color_jitter_brightness,
                contrast=color_jitter_contrast,
                saturation=color_jitter_saturation,
                hue=color_jitter_hue
            ))
    
    # 6. Convert to tensor and normalize
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean, std))
    
    # 7. Cutout/RandomErasing (optional, happens after normalization)
    if use_cutout:
        # Try RandomErasing first (more recent)
        random_erasing = getattr(transforms, 'RandomErasing', None)
        if random_erasing is not None:
            transform_list.append(RandomErasing(
                p=0.5,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
                value=0
            ))
        else:
            # Fallback to Cutout
            transform_list.append(Cutout(n_holes=1, length=cutout_length))
    
    return transforms.Compose(transform_list)


def get_test_transform(dataset='cifar10'):
    """Build standard test transform (no augmentation)."""
    if dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
    else:  # cifar100
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def get_cifar10_dataloaders(batch_size=64, num_workers=4, data_dir='./data',
                             randaug_enabled=False, randaug_n=2, randaug_m=9,
                             use_cutout=False, **kwargs):
    """Get CIFAR-10 data loaders with advanced transforms."""
    
    train_transform = get_train_transform(
        dataset='cifar10',
        randaug_enabled=randaug_enabled,
        randaug_n=randaug_n,
        randaug_m=randaug_m,
        use_cutout=use_cutout,
        **kwargs
    )
    
    test_transform = get_test_transform(dataset='cifar10')
    
    train_dataset = CIFAR10(
        root=data_dir, 
        train=True, 
        download=False, 
        transform=train_transform
    )
    
    test_dataset = CIFAR10(
        root=data_dir, 
        train=False, 
        download=False, 
        transform=test_transform
    )
    
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


def get_cifar100_dataloaders(batch_size=64, num_workers=4, data_dir='./data',
                              randaug_enabled=False, randaug_n=2, randaug_m=9,
                              use_cutout=False, **kwargs):
    """Get CIFAR-100 data loaders with advanced transforms."""
    
    train_transform = get_train_transform(
        dataset='cifar100',
        randaug_enabled=randaug_enabled,
        randaug_n=randaug_n,
        randaug_m=randaug_m,
        use_cutout=use_cutout,
        **kwargs
    )
    
    test_transform = get_test_transform(dataset='cifar100')
    
    train_dataset = CIFAR100(
        root=data_dir, 
        train=True, 
        download=False, 
        transform=train_transform
    )
    
    test_dataset = CIFAR100(
        root=data_dir, 
        train=False, 
        download=False, 
        transform=test_transform
    )
    
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


def get_dataloaders(dataset='cifar10', batch_size=64, num_workers=4, data_dir='./data',
                    **kwargs):
    """
    Get data loaders for the specified dataset with all augmentation options.
    
    Args:
        dataset: 'cifar10' or 'cifar100'
        batch_size: Batch size for training and testing
        num_workers: Number of workers for data loading
        data_dir: Directory to store/load dataset
        **kwargs: Additional augmentation parameters passed to get_train_transform
    
    Returns:
        train_loader, test_loader: Training and testing data loaders
    """
    if dataset == 'cifar10':
        return get_cifar10_dataloaders(
            batch_size=batch_size, 
            num_workers=num_workers, 
            data_dir=data_dir,
            **kwargs
        )
    elif dataset == 'cifar100':
        return get_cifar100_dataloaders(
            batch_size=batch_size, 
            num_workers=num_workers, 
            data_dir=data_dir,
            **kwargs
        )
    else:
        raise ValueError(f"Invalid dataset: {dataset}. Must be 'cifar10' or 'cifar100'")


if __name__ == "__main__":
    print("Testing advanced data augmentation...")
    
    # Test baseline
    print("\n1. Testing baseline (RandomCrop + RandomFlip):")
    train_loader, test_loader = get_dataloaders('cifar10', batch_size=16)
    images, labels = next(iter(train_loader))
    print(f"  Train batch - Images shape: {images.shape}, Labels shape: {labels.shape}")
    
    # Test with RandAugment
    print("\n2. Testing with RandAugment:")
    train_loader, test_loader = get_dataloaders(
        'cifar10', batch_size=16, 
        randaug_enabled=True, randaug_n=2, randaug_m=9
    )
    images, labels = next(iter(train_loader))
    print(f"  Train batch - Images shape: {images.shape}, Labels shape: {labels.shape}")
    
    # Test with Cutout
    print("\n3. Testing with Cutout:")
    train_loader, test_loader = get_dataloaders(
        'cifar10', batch_size=16,
        use_cutout=True, cutout_length=16
    )
    images, labels = next(iter(train_loader))
    print(f"  Train batch - Images shape: {images.shape}, Labels shape: {labels.shape}")
    
    # Test with ColorJitter
    print("\n4. Testing with ColorJitter:")
    train_loader, test_loader = get_dataloaders(
        'cifar10', batch_size=16,
        use_color_jitter=True
    )
    images, labels = next(iter(train_loader))
    print(f"  Train batch - Images shape: {images.shape}, Labels shape: {labels.shape}")
    
    # Test with RandAugment + Cutout
    print("\n5. Testing with RandAugment + Cutout:")
    train_loader, test_loader = get_dataloaders(
        'cifar10', batch_size=16,
        randaug_enabled=True, randaug_n=2, randaug_m=9,
        use_cutout=True, cutout_length=16
    )
    images, labels = next(iter(train_loader))
    print(f"  Train batch - Images shape: {images.shape}, Labels shape: {labels.shape}")
    
    print("\n✅ All augmentation tests completed successfully!")