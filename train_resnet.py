"""Training script for ResNet on CIFAR-10"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import time

from cnn_model import ResNet18
from data_loader import get_cifar10_dataloaders
from config import Config


class ResNetTrainer:
    """Trainer class for ResNet on CIFAR-10."""
    
    def __init__(self, config, model_type='resnet18'):
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Model
        if model_type == 'resnet18':
            self.model = ResNet18(num_classes=config.num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
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
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4
        )
        
        # Learning rate scheduler
        self.scheduler = MultiStepLR(
            self.optimizer,
            milestones=[100, 150],
            gamma=0.1
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
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
    
    def save_model(self, path='resnet18_cifar10.pth'):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }, path)
        print(f"Model saved to {path}")
    
    def train(self):
        """Main training loop."""
        print(f"Training on device: {self.device}")
        print(f"Model: ResNet-18")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        
        best_acc = 0.0
        
        # Train for 200 epochs for ResNet
        num_epochs = max(self.config.num_epochs, 200)
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Step scheduler
            self.scheduler.step()
            
            # Evaluate
            test_loss, test_acc = self.evaluate()
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            print("-" * 50)
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                self.save_model('resnet18_cifar10_best.pth')
        
        # Save final model
        self.save_model('resnet18_cifar10_final.pth')
        
        print(f"\nTraining completed! Best test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    config = Config()
    trainer = ResNetTrainer(config, model_type='resnet18')
    trainer.train()
