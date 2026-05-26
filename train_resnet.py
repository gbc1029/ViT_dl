"""Training script for ResNet on CIFAR-10 with multiple modes"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import argparse
import os
from datetime import datetime

from cnn_model import ResNet18, ResNet34
from data_loader import get_cifar10_dataloaders
from config_resnet import ResNetConfig


def get_timestamp_filename(base_name, ext='pth'):
    """Generate filename with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base_name}_{timestamp}.{ext}"


def save_checkpoint(model, optimizer, scheduler, history, config, filename='checkpoint.pth'):
    """Save model checkpoint with timestamp."""
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    # Add timestamp to filename
    save_path = os.path.join(save_dir, get_timestamp_filename(filename.replace('.pth', '')))
    
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


class ResNetTrainer:
    """Trainer class for ResNet on CIFAR-10."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu'
        )
        
        # Model
        if config.model_type == 'resnet18':
            self.model = ResNet18(num_classes=config.num_classes)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
        
        self.model = self.model.to(self.device)
        
        # Data loaders
        self.train_loader, self.test_loader = get_cifar10_dataloaders(
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = MultiStepLR(
            self.optimizer,
            milestones=config.lr_milestones,
            gamma=config.lr_gamma
        )
        
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
        
        # Forward pass
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        print(f"Dry run successful!")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Image shape: {images.shape}")
        print(f"  Output shape: {outputs.shape}")
        
        # Test one forward pass
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images)
            _, predicted = outputs.max(1)
            accuracy = (predicted == labels).sum().item() / labels.size(0)
            print(f"  Accuracy: {100 * accuracy:.2f}%")
        
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
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%',
                'lr': f'{current_lr:.4f}'
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
    
    def save_model(self, path='resnet18.pth', save_final=True):
        """Save model checkpoint."""
        path = save_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            self.history,
            self.config,
            filename=path
        )
        return path
    
    def train(self):
        """Main training loop."""
        print(f"Training on device: {self.device}")
        print(f"Model: {self.config.model_type}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        
        best_acc = 0.0
        
        # If resuming, start from current epoch
        start_epoch = len(self.history['train_loss'])
        
        for epoch in range(start_epoch, self.config.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Step scheduler
            self.scheduler.step()
            
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
                self.save_model('resnet18_best.pth', save_final=False)
        
        # Save final model
        self.save_model('resnet18_final.pth')
        
        print(f"\nTraining completed! Best test accuracy: {best_acc:.2f}%")


def test_mode(config):
    """Test mode: evaluate a saved model."""
    print("Testing mode...")
    
    if not config.checkpoint_path:
        print("Error: checkpoint_path must be specified in test mode")
        return
    
    device = torch.device(
        'cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu'
    )
    
    # Load model
    if config.model_type == 'resnet18':
        model = ResNet18(num_classes=config.num_classes)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    model = model.to(device)
    
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
    parser = argparse.ArgumentParser(description='Train or test ResNet on CIFAR-10')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['dry_run', 'train', 'test', 'resume'],
                       help='Training mode: dry_run, train, test, resume')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34'],
                       help='Model type: resnet18 or resnet34')
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
    args = parse_args()
    config = ResNetConfig()
    
    # Override config with command line arguments
    if args.mode:
        config.mode = args.mode
    if args.model:
        config.model_type = args.model
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    
    trainer = ResNetTrainer(config)
    
    if config.mode == 'dry_run':
        trainer.dry_run()
    elif config.mode == 'test':
        test_mode(config)
    else:  # train or resume
        trainer.train()
