"""Data loading utilities for ResNet on CIFAR-10/CIFAR-100 (matching ViT pipeline)"""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
import random

# Optional Kornia import for GPU augmentations
try:
    import kornia
    import kornia.augmentation as K
    import kornia.augmentation.auto as auto
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False


class Cutout(object):
    """Randomly mask out one or more patches from an image."""
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
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
    """Apply MixUp augmentation to a batch of images and labels."""
    def __init__(self, alpha=0.2, num_classes=10):
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, batch_imgs, batch_labels):
        if self.alpha <= 0:
            return batch_imgs, batch_labels
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch_imgs.size(0)
        index = torch.randperm(batch_size)
        mixed_imgs = lam * batch_imgs + (1 - lam) * batch_imgs[index, :]
        labels_onehot = torch.zeros(batch_size, self.num_classes, device=batch_imgs.device)
        labels_onehot.scatter_(1, batch_labels.unsqueeze(1), 1)
        mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[index, :]
        return mixed_imgs, mixed_labels


class KorniaAugmentationPipeline:
    """GPU-accelerated augmentation pipeline using Kornia."""
    
    def __init__(self, use_flip_crop=True, randaug_enabled=False, randaug_n=2,
                 randaug_m=9, use_cutout=False, device='cuda'):
        self.device = device
        if not KORNIA_AVAILABLE:
            raise ImportError("Kornia required. pip install kornia")
        
        self.flip_aug = K.RandomHorizontalFlip(same_on_batch=False, p=0.5)
        
        if randaug_enabled:
            self.randaug = auto.RandAugment(n=randaug_n, m=randaug_m)
        else:
            self.randaug = None
        
        if use_cutout:
            self.erasing = K.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), same_on_batch=False)
        else:
            self.erasing = None
        
        self.flip_aug = self.flip_aug.to(device)
        if self.randaug:
            self.randaug = self.randaug.to(device)
        if self.erasing:
            self.erasing = self.erasing.to(device)
    
    def random_crop_with_padding(self, batch_imgs, padding=4):
        b, c, h, w = batch_imgs.shape
        batch_imgs = torch.nn.functional.pad(batch_imgs, (padding, padding, padding, padding), mode='reflect')
        crop_size = h
        top = torch.randint(0, 2 * padding + 1, (1,), device=self.device).item()
        left = torch.randint(0, 2 * padding + 1, (1,), device=self.device).item()
        return batch_imgs[:, :, top:top+crop_size, left:left+crop_size]
    
    def __call__(self, batch_imgs, batch_labels=None):
        batch_imgs = batch_imgs.to(self.device)
        batch_imgs = self.random_crop_with_padding(batch_imgs, padding=4)
        batch_imgs = self.flip_aug(batch_imgs)

        if self.randaug:
            batch_imgs = self.randaug(batch_imgs)

        if self.erasing:
            type = batch_imgs.dtype
            if batch_imgs.dtype != torch.float32:
                print(f"Converting from {batch_imgs} to float32")
                batch_imgs = batch_imgs.float()
            batch_imgs = self.erasing(batch_imgs)
            batch_imgs = batch_imgs.to(type)
        if batch_labels is not None:
            return batch_imgs, batch_labels
        return batch_imgs


def get_train_transform(dataset='cifar10', randaug_enabled=False, randaug_n=2, randaug_m=9,
                        use_cutout=False, cutout_length=16, use_color_jitter=False,
                        color_jitter_brightness=0.2, color_jitter_contrast=0.2,
                        color_jitter_saturation=0.2, color_jitter_hue=0.1,
                        use_random_rotation=False, rotation_degrees=15,
                        use_random_affine=False, affine_translate=0.1):
    """Build advanced train transform pipeline for CIFAR datasets."""
    if dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
    else:
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    
    transform_list = []
    transform_list.append(transforms.RandomCrop(32, padding=4))
    transform_list.append(transforms.RandomHorizontalFlip())
    
    if use_random_rotation:
        transform_list.append(transforms.RandomRotation(rotation_degrees))
    if use_random_affine:
        transform_list.append(transforms.RandomAffine(
            degrees=rotation_degrees if not use_random_rotation else 0,
            translate=(affine_translate, affine_translate)
        ))
    if randaug_enabled:
        rand_augment = getattr(transforms, 'RandAugment', None)
        if rand_augment is not None:
            transform_list.append(rand_augment(num_ops=randaug_n, magnitude=randaug_m))
        else:
            print("Warning: RandAugment not available, falling back to ColorJitter")
            use_color_jitter = True
    else:
        if use_color_jitter:
            transform_list.append(transforms.ColorJitter(
                brightness=color_jitter_brightness, contrast=color_jitter_contrast,
                saturation=color_jitter_saturation, hue=color_jitter_hue
            ))
    
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean, std))
    
    if use_cutout:
        random_erasing = getattr(transforms, 'RandomErasing', None)
        if random_erasing is not None:
            transform_list.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0))
        else:
            transform_list.append(Cutout(n_holes=1, length=cutout_length))
    
    return transforms.Compose(transform_list)


def get_test_transform(dataset='cifar10'):
    """Build standard test transform (no augmentation)."""
    if dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
    else:
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def get_dataloaders(dataset='cifar10', batch_size=64, num_workers=4, data_dir='./data',
                    **kwargs):
    """Get data loaders with specified augmentations."""
    train_transform = get_train_transform(dataset=dataset, **kwargs)
    test_transform = get_test_transform(dataset=dataset)
    
    if dataset == 'cifar10':
        DatasetClass = CIFAR10
    elif dataset == 'cifar100':
        DatasetClass = CIFAR100
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    
    train_dataset = DatasetClass(root=data_dir, train=True, download=False, transform=train_transform)
    test_dataset = DatasetClass(root=data_dir, train=False, download=False, transform=test_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=True, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=True, pin_memory=True
    )
    return train_loader, test_loader


class KorniaDatasetWrapper(torch.utils.data.Dataset):
    """Dataset wrapper that applies Kornia augmentations via collate function.
    
    The collate_fn as a bound method is pickleable (unlike closures),
    making it compatible with Windows multiprocessing.
    """
    def __init__(self, dataset, kornia_pipeline):
        self.dataset = dataset
        self.kornia_pipeline = kornia_pipeline
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label
    
    def kornia_collate_fn(self, batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs)
        labels = torch.tensor(labels)
        imgs, labels = self.kornia_pipeline(imgs, labels)
        return imgs, labels


def get_dataloaders_with_kornia(dataset='cifar10', batch_size=64, num_workers=0,
                                 data_dir='./data', randaug_enabled=False, randaug_n=2,
                                 randaug_m=9, use_cutout=False, device='cuda', **kwargs):
    """Get data loaders with GPU-based Kornia augmentations.
    
    Note: num_workers must be 0 when using GPU augmentations, since
    the collate_fn runs on the main process to access CUDA.
    """
    if not KORNIA_AVAILABLE:
        raise ImportError("Kornia required. pip install kornia")
    
    if dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
    else:
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    if dataset == 'cifar10':
        DatasetClass = CIFAR10
    elif dataset == 'cifar100':
        DatasetClass = CIFAR100
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    
    train_dataset = DatasetClass(root=data_dir, train=True, download=False, transform=base_transform)
    test_dataset = DatasetClass(root=data_dir, train=False, download=False, transform=base_transform)
    
    kornia_pipeline = KorniaAugmentationPipeline(
        use_flip_crop=True, randaug_enabled=randaug_enabled,
        randaug_n=randaug_n, randaug_m=randaug_m,
        use_cutout=use_cutout, device=device
    )
    
    kornia_wrapper = KorniaDatasetWrapper(train_dataset, kornia_pipeline)
    
    train_loader = DataLoader(
        kornia_wrapper, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
        collate_fn=kornia_wrapper.kornia_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )
    return train_loader, test_loader


if __name__ == "__main__":
    print("Testing data loaders...")
    for ds in ['cifar10', 'cifar100']:
        train_loader, test_loader = get_dataloaders(dataset=ds, batch_size=16, num_workers=0)
        images, labels = next(iter(train_loader))
        print(f"  {ds}: {images.shape}, {labels.shape}")
    print("Done.")
