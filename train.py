"""Training script for ResNet on CIFAR-10/CIFAR-100 (matching ViT training pipeline)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm
import argparse
import os
from datetime import datetime
import numpy as np
import random
import inspect

from model import resnet_tiny, resnet_small
from data_loader import get_dataloaders, get_dataloaders_with_kornia, MixUp
from config import Config

# ── Logging helpers ──────────────────────────────────────────────

def _log_info(msg):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - {msg}")

def _log_warning(msg):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - WARNING - {msg}")

def _log_error(msg):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {msg}")


# ── Checkpoint helpers ───────────────────────────────────────────

def get_checkpoint_dir(config):
    return f'checkpoints/resnet'

def get_timestamp_filename(base_name, ext='pth'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base_name}_{timestamp}.{ext}"

def save_checkpoint(model, optimizer, scheduler, history, config, scaler=None,
                    ema_model=None, filename='checkpoint.pth'):
    save_dir = get_checkpoint_dir(config)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, get_timestamp_filename(filename.replace('.pth', '')))

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'history': history,
        'config': config.__dict__,
        'random_states': {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'np': np.random.get_state(),
            'random': random.getstate(),
        },
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    if ema_model is not None:
        checkpoint['ema_state_dict'] = ema_model.state_dict()

    torch.save(checkpoint, save_path)
    _log_info(f"Checkpoint saved to: {save_path}")
    return save_path

def load_checkpoint(model, checkpoint_path, config=None, optimizer=None,
                    scheduler=None, device=torch.device('cuda'), scaler=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    _log_info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    if config and 'config' in checkpoint:
        saved = checkpoint['config']
        for k, v in saved.items():
            if hasattr(config, k):
                setattr(config, k, v)
    return checkpoint.get('history', {})

def get_latest_checkpoint(config):
    ckpt_dir = get_checkpoint_dir(config)
    if not os.path.exists(ckpt_dir):
        return None
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    if not ckpts:
        return None
    ckpts.sort(reverse=True)
    return os.path.join(ckpt_dir, ckpts[0])


# ── Model factory ────────────────────────────────────────────────

def create_model(config):
    if config.model_type == 'resnet_tiny':
        return resnet_tiny(num_classes=config.num_classes)
    elif config.model_type == 'resnet_small':
        return resnet_small(num_classes=config.num_classes)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")


# ── Trainer ──────────────────────────────────────────────────────

class ResNetTrainer:
    """Trainer for ResNet on CIFAR-10/100, matching ViT training pipeline."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu')

        # Handle resume mode
        if config.mode == 'resume':
            if config.checkpoint_path is None:
                config.checkpoint_path = get_latest_checkpoint(config)
            if config.checkpoint_path:
                _log_info(f"Loading checkpoint config from: {config.checkpoint_path}")
                checkpoint = torch.load(config.checkpoint_path, map_location='cpu')
                saved_config = checkpoint.get('config', {})
                if saved_config:
                    for key, value in saved_config.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)

        # Re-set num_classes based on dataset
        if config.dataset == 'cifar10':
            config.num_classes = 10
        elif config.dataset == 'cifar100':
            config.num_classes = 100
        _log_info(f"Dataset: {config.dataset} -> num_classes={config.num_classes}")

        # Save random states
        self.random_states = {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'np': np.random.get_state(),
            'random': random.getstate(),
        }

        # Model
        _log_info(f"Creating ResNet model: {config.model_type}")
        self.model = create_model(config).to(self.device)
        _log_info(f"  Parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")

        # Data loaders
        _log_info(f"Creating data loaders for {config.dataset}")

        if getattr(config, 'use_kornia', False):
            _log_info("Using GPU-accelerated Kornia augmentations")
            self.train_loader, self.test_loader = get_dataloaders_with_kornia(
                dataset=config.dataset, batch_size=config.batch_size,
                num_workers=config.num_workers, data_dir=config.data_dir,
                randaug_enabled=config.randaug_enabled, randaug_n=config.randaug_n,
                randaug_m=config.randaug_m, use_cutout=config.use_cutout,
                device=self.device)
        else:
            _log_info("Using standard torchvision (CPU) augmentations")
            self.train_loader, self.test_loader = get_dataloaders(
                dataset=config.dataset, batch_size=config.batch_size,
                num_workers=config.num_workers, data_dir=config.data_dir,
                randaug_enabled=config.randaug_enabled, randaug_n=config.randaug_n,
                randaug_m=config.randaug_m, use_cutout=config.use_cutout,
                cutout_length=config.cutout_length,
                use_color_jitter=config.use_color_jitter,
                color_jitter_brightness=config.color_jitter_brightness,
                color_jitter_contrast=config.color_jitter_contrast,
                color_jitter_saturation=config.color_jitter_saturation,
                color_jitter_hue=config.color_jitter_hue,
                use_random_rotation=config.use_random_rotation,
                rotation_degrees=config.rotation_degrees,
                use_random_affine=config.use_random_affine,
                affine_translate=config.affine_translate)

        # Loss with optional label smoothing
        self.label_smoothing = config.label_smoothing
        kwargs = {}
        if self.label_smoothing > 0:
            try:
                sig = inspect.signature(nn.CrossEntropyLoss.__init__)
                if 'label_smoothing' in sig.parameters:
                    test_criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
                    kwargs['label_smoothing'] = self.label_smoothing
                    _log_info(f"Label smoothing enabled: {self.label_smoothing}")
                else:
                    _log_warning("Label smoothing not available, disabling")
                    self.label_smoothing = 0.0
            except Exception as e:
                _log_warning(f"Label smoothing init failed: {e}, disabling")
                self.label_smoothing = 0.0
        else:
            _log_info("Label smoothing disabled")
        self.criterion = nn.CrossEntropyLoss(**kwargs)

        # Optimizer (SGD for ResNet)
        _log_info(f"Creating SGD optimizer: lr={config.learning_rate}, momentum={config.momentum}, weight_decay={config.weight_decay}")
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True)

        # LR scheduler (MultiStep for ResNet)
        _log_info(f"MultiStepLR scheduler: milestones={config.lr_milestones}, gamma={config.lr_gamma}")
        self.scheduler = MultiStepLR(self.optimizer, milestones=config.lr_milestones, gamma=config.lr_gamma)

        # AMP
        self.use_amp = config.use_amp
        if hasattr(torch.amp, 'GradScaler'):
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp and torch.cuda.is_available() and str(self.device) != 'cpu')
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and torch.cuda.is_available() and str(self.device) != 'cpu')
        if self.use_amp:
            _log_info("AMP enabled")
        else:
            _log_info("AMP disabled")

        # EMA
        self.ema_decay = getattr(config, 'ema_decay', 0.0)
        self.ema_model = None
        if self.ema_decay > 0:
            self.ema_model = create_model(config).to(self.device)
            self.ema_model.load_state_dict(self.model.state_dict())
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
            _log_info(f"EMA enabled: decay={self.ema_decay}")
        else:
            _log_info("EMA disabled")

        # MixUp
        if getattr(config, 'use_mixup', False):
            alpha = getattr(config, 'mixup_alpha', 0.2)
            prob = getattr(config, 'mixup_prob', 0.5)
            self.mixup = MixUp(alpha=alpha, num_classes=config.num_classes)
            self.mixup_prob = prob
            _log_info(f"MixUp enabled: alpha={alpha}, prob={prob}")
        else:
            self.mixup = None
            self.mixup_prob = 0.0
            _log_info("MixUp disabled")

        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'test_loss': [], 'test_acc': []
        }

        # Resume mode: load model, optimizer, scheduler states
        if config.mode == 'resume' and config.checkpoint_path:
            history = load_checkpoint(
                self.model, config.checkpoint_path, config,
                self.optimizer, self.scheduler, self.device,
                scaler=self.scaler if self.use_amp else None)
            self.history.update(history)
            _log_info(f"Resumed from epoch {len(self.history['train_loss'])}")
            # Also restore EMA if it exists
            if self.ema_model is not None and config.checkpoint_path:
                ckpt = torch.load(config.checkpoint_path, map_location=self.device)
                if 'ema_state_dict' in ckpt:
                    self.ema_model.load_state_dict(ckpt['ema_state_dict'])
                    _log_info("EMA states restored")

    # ── EMA update ──
    def _ema_update(self):
        if self.ema_model is None:
            return
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.data.mul_(self.ema_decay).add_(model_p.data, alpha=1 - self.ema_decay)

    # ── Dry Run ──
    def dry_run(self):
        print("Starting dry run...")
        self.model.train()
        images, labels = next(iter(self.train_loader))
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print(f"Dry run successful!")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Output shape: {outputs.shape}")
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images)
            _, predicted = outputs.max(1)
            acc = (predicted == labels).sum().item() / labels.size(0)
            print(f"  Accuracy: {100 * acc:.2f}%")

    # ── Train Epoch ──
    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
        for images, labels in pbar:
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            mixup_applied = False

            if self.mixup is not None and random.random() < self.mixup_prob:
                images, labels = self.mixup(images, labels)
                mixup_applied = True

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                if mixup_applied:
                    loss = F.kl_div(F.log_softmax(outputs, dim=1), labels, reduction='batchmean')
                else:
                    loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            if getattr(self.config, 'grad_clip', 0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # EMA update per step
            self._ema_update()

            train_loss += loss.item()
            if mixup_applied:
                _, pred_labels = labels.max(1)
            else:
                pred_labels = labels
            _, predicted = outputs.max(1)
            total += pred_labels.size(0)
            correct += predicted.eq(pred_labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

        avg_loss = train_loss / len(self.train_loader)
        avg_acc = 100. * correct / total
        self.history['train_loss'].append(avg_loss)
        self.history['train_acc'].append(avg_acc)
        return avg_loss, avg_acc

    # ── Evaluate ──
    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        # Use EMA model for evaluation if available
        eval_model = self.ema_model if self.ema_model is not None else self.model
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = eval_model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        avg_loss = test_loss / len(self.test_loader)
        avg_acc = 100. * correct / total
        self.history['test_loss'].append(avg_loss)
        self.history['test_acc'].append(avg_acc)
        return avg_loss, avg_acc

    # ── Train Loop ──
    def train(self):
        print(f"\nTraining on device: {self.device}")
        print(f"Model: {self.config.model_type}")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        print(f"Number of epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print("─" * 50)

        best_acc = 0.0
        start_epoch = len(self.history['train_loss'])

        for epoch in range(start_epoch, self.config.num_epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            self.scheduler.step()
            test_loss, test_acc = self.evaluate()

            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            print("─" * 50)

            if test_acc > best_acc:
                best_acc = test_acc
                self.save_model(f'best_resnet.pth')

        self.save_model(f'final_resnet.pth')
        print(f"\nTraining completed! Best test accuracy: {best_acc:.2f}%")
        return self.history

    def save_model(self, filename='checkpoint.pth'):
        return save_checkpoint(
            self.model, self.optimizer, self.scheduler, self.history,
            self.config, scaler=self.scaler if self.use_amp else None,
            ema_model=self.ema_model, filename=filename)


# ── Test Mode ────────────────────────────────────────────────────

def test_mode(config):
    print("Testing mode...")
    if not config.checkpoint_path:
        print("Error: checkpoint_path must be specified")
        return

    device = torch.device('cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu')
    if config.dataset == 'cifar10':
        config.num_classes = 10
    else:
        config.num_classes = 100

    model = create_model(config).to(device)
    load_checkpoint(model, config.checkpoint_path, device=device)

    _, test_loader = get_dataloaders(
        dataset=config.dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, data_dir=config.data_dir)

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss / len(test_loader):.4f}")
    print(f"  Accuracy: {100. * correct / total:.2f}%")

# ── CLI ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet on CIFAR-10/100 (matching ViT pipeline)')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['dry_run', 'train', 'test', 'resume'],
                       help='Training mode')
    parser.add_argument('--model', type=str, default='resnet_tiny',
                       choices=['resnet_tiny', 'resnet_small'],
                       help='Model type: resnet_tiny (~1.64M) or resnet_small (~9.66M)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for test/resume')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--dataset', type=str, default=None, choices=['cifar10', 'cifar100'])
    parser.add_argument('--label-smoothing', type=float, default=None)

    # AMP
    parser.add_argument('--amp', action='store_true', default=None)

    # EMA
    parser.add_argument('--ema-decay', type=float, default=None)

    # Gradient Clipping
    parser.add_argument('--grad-clip', type=float, default=None)

    # Data Augmentation
    parser.add_argument('--randaug', action='store_true', default=None)
    parser.add_argument('--randaug-n', type=int, default=None)
    parser.add_argument('--randaug-m', type=int, default=None)
    parser.add_argument('--cutout', action='store_true', default=None)
    parser.add_argument('--cutout-length', type=int, default=None)
    parser.add_argument('--color-jitter', action='store_true', default=None)
    parser.add_argument('--rotation', action='store_true', default=None)
    parser.add_argument('--rotation-degrees', type=float, default=None)
    parser.add_argument('--affine', action='store_true', default=None)

    # MixUp
    parser.add_argument('--mixup', action='store_true', default=None)
    parser.add_argument('--mixup-alpha', type=float, default=None)
    parser.add_argument('--mixup-prob', type=float, default=None)

    # Other
    parser.add_argument('--verbose', action='store_true', default=None)
    parser.add_argument('--debug', action='store_true', default=None)

    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    config = Config(dataset=args.dataset, model_type=args.model)

    # Override config from args
    overrides = {
        'mode': args.mode,
        'checkpoint_path': args.checkpoint,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'label_smoothing': args.label_smoothing,
        'use_amp': args.amp,
        'ema_decay': args.ema_decay,
        'grad_clip': args.grad_clip,
        'randaug_enabled': args.randaug,
        'randaug_n': args.randaug_n,
        'randaug_m': args.randaug_m,
        'use_cutout': args.cutout,
        'cutout_length': args.cutout_length,
        'use_color_jitter': args.color_jitter,
        'use_random_rotation': args.rotation,
        'rotation_degrees': args.rotation_degrees,
        'use_random_affine': args.affine,
        'use_mixup': args.mixup,
        'mixup_alpha': args.mixup_alpha,
        'mixup_prob': args.mixup_prob,
        'verbose': args.verbose,
        'debug': args.debug,
    }
    for key, val in overrides.items():
        if val is not None and hasattr(config, key):
            setattr(config, key, val)

    config.num_classes = 10 if config.dataset == 'cifar10' else 100

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nConfiguration:")
    print(f"  Model: {config.model_type}")
    print(f"  Dataset: {config.dataset} -> num_classes={config.num_classes}")
    print(f"  Mode: {config.mode}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  AMP: {config.use_amp}")
    print(f"  EMA decay: {config.ema_decay}")
    print(f"  Grad clip: {config.grad_clip}")
    print(f"  Label smoothing: {config.label_smoothing}")
    print(f"  RandAugment: {config.randaug_enabled}")
    print(f"  Cutout: {config.use_cutout}")
    print(f"  MixUp: {config.use_mixup}")

    if config.mode == 'test':
        test_mode(config)
        exit(0)

    trainer = ResNetTrainer(config)

    if config.mode == 'dry_run':
        trainer.dry_run()
    else:
        trainer.train()
