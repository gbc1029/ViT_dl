"""Training script for Vision Transformer on CIFAR-10 with multiple modes"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from tqdm import tqdm
import argparse
import os
from datetime import datetime
from glob import glob
import numpy as np
import random
from model import VisionTransformer
from data_loader import get_cifar10_dataloaders
from config import Config
from utils import plot_results, count_parameters, visualize_patches
from utils import _log_info, _log_debug, _log_warning, _log_error, init_logging, get_logger
import logging

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
        'config': config.__dict__,
        'random_states': {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'np': np.random.get_state(),
            'random': random.getstate(),
        },
        'saved_fields': [k for k in config.__dict__.keys() if k in ['warmup_epochs', 'device', 'batch_size', 'learning_rate', 'weight_decay', 'num_epochs', 'num_workers']],
    }
    
    # Add scheduler if exists
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    _log_info(f"Checkpoint saved to: {save_path}")
    return save_path


def get_latest_checkpoint():
    """Get the latest checkpoint file."""
    save_dir = 'checkpoints'
    if not os.path.exists(save_dir):
        return None
    
    checkpoints = glob.glob(f"{save_dir}/*.pth")
    if not checkpoints:
        return None
    
    # Sort by modification time (descending)
    checkpoints = sorted(checkpoints, key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None, device=torch.device('cuda')):
    """Load model checkpoint."""
    # Auto-select latest if path is None
    if checkpoint_path is None:
        latest = get_latest_checkpoint()
        if latest:
            _log_info(f"No checkpoint path specified, using latest: {latest}")
            checkpoint_path = latest
        else:
            raise FileNotFoundError("No checkpoint found in checkpoints directory")
    
    # Check if file exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    _log_info(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        _log_info("Optimizer state loaded")
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        _log_info("Scheduler state loaded")
    
    # Restore random states
    if 'random_states' in checkpoint:
        random_states = checkpoint['random_states']
        torch.set_rng_state(random_states['torch'])
        if random_states['cuda'] is not None:
            torch.cuda.set_rng_state_all(random_states['cuda'])
        np.random.set_state(random_states['np'])
        random.setstate(random_states['random'])
        _log_info("Random states restored")
    
    history = checkpoint.get('history', {})
    config_dict = checkpoint.get('config', {})
    
    # Show saved fields
    saved_fields = checkpoint.get('saved_fields', [])
    _log_info(f"Checkpoint contains {len(saved_fields)} saved fields: {saved_fields}")
    
    return {'history': history, 'config': config_dict}


class Trainer:
    """Main trainer class for ViT on CIFAR-10."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu')
        
        # Handle resume mode: restore configuration from checkpoint FIRST
        if config.mode == 'resume':
            # Determine checkpoint path
            if config.checkpoint_path is None:
                config.checkpoint_path = get_latest_checkpoint()
                if config.checkpoint_path:
                    _log_info(f"No checkpoint path specified, using latest: {config.checkpoint_path}")
            
            if config.checkpoint_path:
                # Load checkpoint to get saved configuration (without creating model yet)
                _log_info(f"Loading checkpoint configuration from: {config.checkpoint_path}")
                checkpoint = torch.load(config.checkpoint_path, map_location='cpu')
                saved_config_dict = checkpoint.get('config', {})
                
                # Restore saved configuration
                if saved_config_dict:
                    _log_info("Restoring configuration from checkpoint:")
                    restored_fields = []
                    conflicting_fields = []
                    
                    for key, value in saved_config_dict.items():
                        if hasattr(self.config, key):
                            old_value = getattr(self.config, key)
                            if old_value != value:
                                conflicting_fields.append(f"    {key}: {old_value} -> {value}")
                            else:
                                restored_fields.append(f"    {key}: {value}")
                            setattr(self.config, key, value)
                    
                    # Log restored configurations
                    if restored_fields:
                        _log_info("  Restored (unchanged from current):")
                        for field in restored_fields[:10]:  # Limit output
                            _log_info(field)
                        if len(restored_fields) > 10:
                            _log_info(f"    ... and {len(restored_fields) - 10} more")
                    
                    if conflicting_fields:
                        _log_warning("  Overridden by checkpoint (conflicts with current config):")
                        for field in conflicting_fields:
                            _log_warning(field)
                    
                    # Special handling for num_epochs (allow extension but not reduction)
                    if 'num_epochs' in saved_config_dict:
                        current_epochs = len(checkpoint.get('history', {}).get('train_loss', []))
                        saved_epochs = saved_config_dict.get('num_epochs', 0)
                        if self.config.num_epochs < saved_epochs:
                            _log_warning(f"Current num_epochs ({self.config.num_epochs}) is less than checkpoint's ({saved_epochs})")
                            _log_warning(f"Keeping checkpoint value: {saved_epochs}")
                            setattr(self.config, 'num_epochs', saved_epochs)
                        elif self.config.num_epochs > saved_epochs:
                            _log_info(f"Extending training: {saved_epochs} -> {self.config.num_epochs} epochs")
                            # Keep the extended value
                        else:
                            _log_info(f"Training duration unchanged: {self.config.num_epochs} epochs")
                else:
                    _log_warning("No configuration found in checkpoint, using current config")
            else:
                raise RuntimeError("Resume mode requires checkpoint, but none found")
        
        # Save random states before creating data loaders
        self.random_states = {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'np': np.random.get_state(),
            'random': random.getstate(),
        }
        
        # Model (using restored config values)
        _log_info(f"Creating Vision Transformer model with config:")
        _log_info(f"  image_size: {self.config.image_size}")
        _log_info(f"  patch_size: {self.config.patch_size}")
        _log_info(f"  dim: {self.config.dim}")
        _log_info(f"  depth: {self.config.depth}")
        _log_info(f"  heads: {self.config.heads}")
        
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
        
        # Data loaders (using restored config values)
        _log_info(f"Creating data loaders with batch_size={self.config.batch_size}")
        self.train_loader, self.test_loader = get_cifar10_dataloaders(
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        
        # Loss and optimizer (using restored config values)
        self.criterion = nn.CrossEntropyLoss()
        _log_info(f"Creating optimizer with lr={self.config.learning_rate}, weight_decay={self.config.weight_decay}")
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler with optional warmup (using restored config values)
        n_batches = len(self.train_loader)
        
        # If warmup_epochs is positive and strictly less than total epochs, use SequentialLR
        if getattr(config, 'warmup_epochs', 0) and config.warmup_epochs < config.num_epochs:
            warmup_iters = n_batches * config.warmup_epochs
            cosine_iters = n_batches * (config.num_epochs - config.warmup_epochs)
            
            # Ensure cosine_iters is at least 1 to avoid ZeroDivisionError in CosineAnnealingLR
            if cosine_iters <= 0:
                cosine_iters = 1
            
            _log_info(f"Creating SequentialLR scheduler with warmup_epochs={config.warmup_epochs}")
            
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
                _log_info(f"Warning: warmup_epochs ({config.warmup_epochs}) >= num_epochs ({config.num_epochs}) — disabling warmup and using cosine schedule for all epochs.")
            
            total_iters = n_batches * max(1, config.num_epochs)
            # Ensure T_max >= 1
            T_max = max(1, total_iters)
            _log_info(f"Creating CosineAnnealingLR scheduler (no warmup)")
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=T_max)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
        # Resume: load model, optimizer, scheduler states AFTER they are created
        if config.mode == 'resume' and config.checkpoint_path:
            _log_info(f"Loading model and training states from: {config.checkpoint_path}")
            
            # Restore random states from checkpoint
            if 'random_states' in checkpoint:
                random_states = checkpoint['random_states']
                torch.set_rng_state(random_states['torch'])
                if random_states['cuda'] is not None:
                    torch.cuda.set_rng_state_all(random_states['cuda'])
                np.random.set_state(random_states['np'])
                random.setstate(random_states['random'])
                _log_info("Random states restored from checkpoint")
            
            # Load model, optimizer, scheduler states
            loaded_data = load_checkpoint(
                self.model, config.checkpoint_path,
                self.optimizer, self.scheduler, self.device
            )
            
            # Restore history
            self.history = loaded_data.get('history', self.history)
            
            # Log resume information
            completed_epochs = len(self.history['train_loss'])
            _log_info(f"Successfully resumed from checkpoint")
            _log_info(f"  Completed epochs: {completed_epochs}")
            _log_info(f"  Remaining epochs: {max(0, self.config.num_epochs - completed_epochs)}")
            _log_info(f"  Best test accuracy so far: {max(self.history['test_acc']) if self.history['test_acc'] else 0:.2f}%")
            
            # Verify configuration consistency
            saved_config_dict = checkpoint.get('config', {})
            critical_fields = ['image_size', 'patch_size', 'dim', 'depth', 'heads', 'mlp_dim', 'num_classes']
            inconsistencies = []
            
            for field in critical_fields:
                if field in saved_config_dict and hasattr(self.config, field):
                    if getattr(self.config, field) != saved_config_dict[field]:
                        inconsistencies.append(field)
            
            if inconsistencies:
                _log_error(f"CRITICAL: Model architecture fields differ from checkpoint: {inconsistencies}")
                _log_error("This will likely cause errors during training!")
                raise RuntimeError(f"Model architecture mismatch: {inconsistencies}")
            else:
                _log_info("Model architecture verification passed")
        
        # Log final configuration summary
        _log_info("=" * 60)
        _log_info("Final configuration for training:")
        _log_info(f"  Mode: {config.mode}")
        _log_info(f"  Device: {self.device}")
        _log_info(f"  Epochs: {config.num_epochs}")
        _log_info(f"  Batch size: {config.batch_size}")
        _log_info(f"  Learning rate: {config.learning_rate}")
        _log_info(f"  Warmup epochs: {config.warmup_epochs}")
        _log_info(f"  Weight decay: {config.weight_decay}")
        _log_info(f"  Convergence patience: {config.convergence_patience}")
        _log_info("=" * 60)
    def check_convergence(self, epoch, test_acc):
        """Double convergence check: accuracy + parameter changes."""
        # Check 1: Accuracy improvement
        if len(self.history['test_acc']) >= self.config.convergence_patience:
            recent_acc = self.history['test_acc'][-self.config.convergence_patience:]
            improvement = max(recent_acc) - min(recent_acc)
            
            if improvement < self.config.convergence_threshold:
                _log_info(f"Accuracy convergence detected: {improvement:.4f} < {self.config.convergence_threshold}")
                _log_info(f"Last {self.config.convergence_patience} epochs accuracy: min={min(recent_acc):.2f}%, max={max(recent_acc):.2f}%")
                return True
        
        # Check 2: Parameter change (using parameter L2 distance)
        if len(self.history['train_loss']) > 1:
            prev_loss = self.history['train_loss'][-2]
            curr_loss = self.history['train_loss'][-1]
            loss_change = abs(curr_loss - prev_loss)
            
            if loss_change < self.config.param_change_threshold:
                _log_info(f"Loss convergence detected: {loss_change:.6f} < {self.config.param_change_threshold}")
                _log_info(f"Previous loss: {prev_loss:.4f}, Current loss: {curr_loss:.4f}")
                return True
        
        return False
    
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
        _log_info(f"Training on device: {self.device}")
        _log_info(f"Model: Vision Transformer")
        _log_info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        _log_info(f"Training samples: {len(self.train_loader.dataset)}")
        _log_info(f"Test samples: {len(self.test_loader.dataset)}")
        _log_info(f"Initial learning rate: {self.config.learning_rate}")
        _log_info(f"Batch size: {self.config.batch_size}")
        _log_info(f"Warmup epochs: {self.config.warmup_epochs}")
        
        best_acc = 0.0
        patience_counter = 0
        
        # If resuming, start from current epoch
        start_epoch = len(self.history['train_loss'])
        
        _log_info(f"Starting training from epoch {start_epoch + 1} to {self.config.num_epochs}")
        
        for epoch in range(start_epoch, self.config.num_epochs):
            _log_info(f"\n{'='*60}")
            _log_info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            _log_info(f"{'='*60}")
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Evaluate
            test_loss, test_acc = self.evaluate()
            
            _log_info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            _log_info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            
            # Check convergence
            if self.check_convergence(epoch, test_acc):
                _log_info(f"Early stopping: Training converged at epoch {epoch+1}")
                _log_info(f"Final test accuracy: {test_acc:.2f}%")
                _log_info(f"Best test accuracy: {best_acc:.2f}%")
                break
            
            # Check patience
            if test_acc > best_acc:
                patience_counter = 0
            else:
                patience_counter += 1
                _log_info(f"Convergence patience counter: {patience_counter}/{self.config.convergence_patience}")
            
            if patience_counter >= self.config.convergence_patience:
                _log_info(f"Early stopping: No improvement for {self.config.convergence_patience} epochs")
                _log_info(f"Final test accuracy: {test_acc:.2f}%")
                _log_info(f"Best test accuracy: {best_acc:.2f}%")
                break
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                _log_info(f"New best model! Test accuracy: {best_acc:.2f}%")
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.history,
                    self.config,
                    filename='vit_cifar10_best'
                )
        
        # Save final model
        _log_info(f"\n{'='*60}")
        _log_info("Training completed or stopped early")
        _log_info(f"Best test accuracy: {best_acc:.2f}%")
        save_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            self.history,
            self.config,
            filename='vit_cifar10_final'
        )


def test_mode(config):
    """Test mode: evaluate a saved model."""
    _log_info("Testing mode...")
    
    # Auto-select latest checkpoint if not specified
    if config.checkpoint_path is None:
        config.checkpoint_path = get_latest_checkpoint()
    
    if config.checkpoint_path is None:
        raise RuntimeError("Test mode requires checkpoint, but none found")
    
    device = torch.device('cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu')
    
    # Load model (create new instance with same architecture)
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
    
    # Use training time's test_loader
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
    _log_info(f"Test completed! Accuracy: {100. * correct / total:.2f}%")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train or test ViT on CIFAR-10')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['dry_run', 'train', 'test', 'resume'],
                       help='Training mode: dry_run, train, test, resume')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for test or resume mode')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--warmup-epochs', type=int, default=None,
                       help='Number of warmup epochs')
    parser.add_argument('--weight-decay', type=float, default=None,
                       help='Weight decay')
    parser.add_argument('--verbose', action='store_true', default=None,
                       help='Enable verbose logging')
    parser.add_argument('--debug', action='store_true', default=None,
                       help='Enable debug logging')
    return parser.parse_args()


if __name__ == "__main__":
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
        
    args = parse_args()
    config = Config()
    
    # Sync command line arguments to Config
    # Priority: command line > config.py defaults
    if args.mode is not None:
        config.mode = args.mode
    if args.checkpoint is not None:
        config.checkpoint_path = args.checkpoint
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.warmup_epochs is not None:
        config.warmup_epochs = args.warmup_epochs
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.verbose is not None:
        config.verbose = args.verbose
    if args.debug is not None:
        config.debug = args.debug
    
    # Initialize logging BEFORE creating trainer
    # Console only shows WARNING and above (INFO/DEBUG go to file only)
    force_new_log = (config.mode != 'resume')
    
    # Set console level: WARNING by default, can be overridden
    console_level = logging.WARNING
    if args.verbose or args.debug:
        # If verbose/debug, still keep console at WARNING to avoid clutter
        console_level = logging.WARNING
    
    init_logging(
        verbose=config.verbose, 
        debug=config.debug, 
        log_to_file=True,
        console_output=True,
        console_level=console_level,
        force_new_log=force_new_log
    )
    
    # These will go to file only (console won't show them)
    _log_info("=" * 60)
    _log_info("Program started")
    _log_info(f"Mode: {config.mode}")
    _log_info(f"Command line arguments: {vars(args)}")
    _log_info(f"Verbose: {config.verbose}, Debug: {config.debug}")
    _log_info("=" * 60)
    
    trainer = Trainer(config)
    
    try:
        if config.mode == 'dry_run':
            trainer.dry_run()
        elif config.mode == 'test':
            if config.checkpoint_path is None:
                config.checkpoint_path = get_latest_checkpoint()
            
            if config.checkpoint_path is None:
                raise RuntimeError("Test mode requires checkpoint, but none found")
            
            test_mode(config)
        else:  # train or resume
            trainer.train()
    except Exception as e:
        _log_error(f"Training failed with error: {e}")
        import traceback
        traceback_str = traceback.format_exc()
        _log_error(traceback_str)
        raise
    finally:
        _log_info("Program finished")
        # Final flush
        for handler in get_logger().handlers:
            handler.flush()
