"""Training script for Vision Transformer on CIFAR-10 with multiple modes"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from tqdm import tqdm
import argparse
import os
from datetime import datetime

from model import VisionTransformer
from data_loader import get_cifar10_dataloaders
from config import Config


def save_checkpoint(model, optimizer, scheduler, history, config, filename='checkpoint'):
    """Save model checkpoint with timestamp."""
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f"{filename}_{timestamp}.pth")
    
    # Prepare checkpoint dict
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': config.__dict__
    }
    
    # Add scheduler if exists
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to: {save_path}")
    return save_path


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None, device='cuda'):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded")
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Scheduler state loaded")
    
    history = checkpoint.get('history', {})
    config = checkpoint.get('config', {})
    
    print(f"Checkpoint loaded from: {checkpoint_path}")
    return {'history': history, 'config': config}


class Trainer:
    """Main trainer class for ViT on CIFAR-10."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model
        self.model = VisionTransformer(
            img_size=config.image_size,
            patch_size=config.patch_size,
            num_classes=config.num_classes,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
            emb_dropout=config.emb_dropout
        ).to(self.device)
        
        # Data loaders
        self.train_loader, self.test_loader = get_cifar10_dataloaders(
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler with optional warmup
        n_batches = len(self.train_loader)

        # If warmup_epochs is positive and strictly less than total epochs, use SequentialLR
        if getattr(config, 'warmup_epochs', 0) and config.warmup_epochs < config.num_epochs:
            warmup_iters = n_batches * config.warmup_epochs
            cosine_iters = n_batches * (config.num_epochs - config.warmup_epochs)

            # Ensure cosine_iters is at least 1 to avoid ZeroDivisionError in CosineAnnealingLR
            if cosine_iters <= 0:
                cosine_iters = 1

            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_iters
            )

            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=cosine_iters
            )

            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_iters]
            )
        else:
            # No warmup (either warmup_epochs == 0 or >= num_epochs): use single cosine scheduler
            if getattr(config, 'warmup_epochs', 0) >= config.num_epochs and config.num_epochs > 0:
                print("Warning: warmup_epochs >= num_epochs — disabling warmup and using cosine schedule for all epochs.")

            total_iters = n_batches * max(1, config.num_epochs)
            # Ensure T_max >= 1
            T_max = max(1, total_iters)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=T_max)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
        # Resume from checkpoint if specified
        if config.checkpoint_path and config.mode == 'resume':
            loaded_data = load_checkpoint(
                self.model, config.checkpoint_path,
                self.optimizer, self.scheduler, self.device
            )
            self.history = loaded_data.get('history', self.history)
            print(f"Resumed from epoch {len(self.history['train_loss'])}")
    
    def dry_run(self):
        """Dry run: quick test to verify everything works."""
        print("Starting dry run...")
        self.model.train()
        
        # Test one batch
        images, labels = next(iter(self.train_loader))
        images, labels = images.to(self.device), labels.to(self.device)
        print(f"Batch loaded successfully! Images shape: {images.shape}, Labels shape: {labels.shape}")
        
        # Forward pass
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        print(f"Forward pass successful! Loss: {loss.item():.4f}")
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print("Backward pass successful! Weights updated.")
        
        # Test one forward pass
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images)
            _, predicted = outputs.max(1)
            accuracy = (predicted == labels).sum().item() / labels.size(0)
            print(f"eval pass successful! Accuracy on batch: {100 * accuracy:.2f}%")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
        
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%',
                'lr': f'{current_lr:.2e}'
            })
        
        avg_loss = train_loss / len(self.train_loader)
        avg_acc = 100. * correct / total
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_acc'].append(avg_acc)
        
        return avg_loss, avg_acc
    
    def evaluate(self):
        """Evaluate on test set."""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
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

    
    def train(self):
        """Main training loop."""
        print(f"Training on device: {self.device}")
        print(f"Model: Vision Transformer")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        
        best_acc = 0.0
        
        # If resuming, start from current epoch
        start_epoch = len(self.history['train_loss'])
        
        for epoch in range(start_epoch, self.config.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Evaluate
            test_loss, test_acc = self.evaluate()
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            print("-" * 50)
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.history,
                    self.config,
                    filename='vit_cifar10_best'
                )
        
        # Save final model
        save_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            self.history,
            self.config,
            filename='vit_cifar10_final'
        )
        
        print(f"\nTraining completed! Best test accuracy: {best_acc:.2f}%")


def test_mode(config):
    """Test mode: evaluate a saved model."""
    print("Testing mode...")
    
    if not config.checkpoint_path:
        print("Error: checkpoint_path must be specified in test mode")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = VisionTransformer(
        img_size=config.image_size,
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        mlp_dim=config.mlp_dim,
        dropout=config.dropout,
        emb_dropout=config.emb_dropout
    ).to(device)
    
    # Load checkpoint
    load_checkpoint(model, config.checkpoint_path, device=device)
    
    # Get data loaders
    _, test_loader = get_cifar10_dataloaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    # Evaluate
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
    print(f"  Correct: {correct}/{total}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train or test ViT on CIFAR-10')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['dry_run', 'train', 'test', 'resume'],
                       help='Training mode: dry_run, train, test, resume')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file for test or resume mode')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    return parser.parse_args()


if __name__ == "__main__":

    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
        
    args = parse_args()
    config = Config()
    
    # Override config with command line arguments
    if args.mode:
        config.mode = args.mode
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    
    trainer = Trainer(config)
    
    if config.mode == 'dry_run':
        trainer.dry_run()
    elif config.mode == 'test':
        test_mode(config)
    else:  # train or resume
        trainer.train()
