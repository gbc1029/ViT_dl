"""Training script for Vision Transformer on CIFAR-10 with multiple modes"""

"""
example
python train.py'
 --mode train --dataset cifar100 --model-type hybrid '
 --epochs 100 --batch-size 64 --lr 3e-4 '
 --weight-decay 0.05 --warmup-epochs 5 '
 --randaug --mixup --drop-path 0.1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from tqdm import tqdm
import argparse
import os
from datetime import datetime
from glob import glob
from data_loader import get_dataloaders, get_dataloaders_with_kornia,KorniaDatasetWrapper,MixUp, KorniaAugmentationPipeline
import numpy as np
import random
from model import VisionTransformer,HybridVisionTransformer
from config import Config
from utils import plot_results, count_parameters, visualize_patches
from utils import _log_info, _log_debug, _log_warning, _log_error, init_logging, get_logger
import logging
import sys


def get_size_code(model_size):
    """Get size code for checkpoint naming."""
    if model_size == 'tiny':
        return 't'
    elif model_size == 'small':
        return 's'
    elif model_size == 'base':
        return 'b'
    else:
        raise ValueError(f"Invalid model_size: {model_size}")


def get_dataset_code(dataset):
    """Get dataset code for checkpoint naming."""
    if dataset == 'cifar10':
        return '10'
    elif dataset == 'cifar100':
        return '100'
    else:
        raise ValueError(f"Invalid dataset: {dataset}")


def get_checkpoint_dir(config):
    """Get checkpoint directory based on model type."""
    if hasattr(config, 'model_type'):
        return f"checkpoints/{config.model_type}"
    else:
        #fallback直接在主目录下保存
        _log_error("cannot get model_type,save to main catalogue")
        return 'checkpoints'

def save_checkpoint(model, optimizer, scheduler, history, config, filename_prefix='', scaler=None):
    """Save model checkpoint with timestamp."""
    # Get save directory based on model type
    save_dir = get_checkpoint_dir(config)
    os.makedirs(save_dir, exist_ok=True)
    
    # Get size and dataset codes
    size_code = get_size_code(config.model_size)
    dataset_code = get_dataset_code(config.dataset)
    
    # Build filename: {size}_{dataset}_{timestamp}.pth
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if filename_prefix:
        filename = f"{filename_prefix}_{size_code}_{dataset_code}_{timestamp}.pth"
    else:
        filename = f"{size_code}_{dataset_code}_{timestamp}.pth"
    
    save_path = os.path.join(save_dir, filename)
    
    # Prepare checkpoint dict with dataset and model size info
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': config.__dict__,
        'dataset': config.dataset,
        'model_size': config.model_size,
        'random_states': {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'np': np.random.get_state(),
            'random': random.getstate(),
        },
         'saved_fields': [k for k in config.__dict__.keys() if k in ['warmup_epochs', 'device', 'batch_size', 'learning_rate', 'weight_decay', 'num_epochs', 'num_workers', 'dataset', 'model_size', 'label_smoothing', 'use_amp', 'ema_decay', 'grad_clip', 'randaug_enabled', 'randaug_n', 'randaug_m', 'use_cutout', 'use_mixup', 'mixup_alpha', 'mixup_prob', 'drop_path_rate', 'use_kornia']],
    }
    
    # Add scheduler if exists
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Add AMP scaler state if provided
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    torch.save(checkpoint, save_path)
    _log_info(f"Checkpoint saved to: {save_path}")
    return save_path


def get_latest_checkpoint(config):
    """Get the latest checkpoint file."""
    save_dir = get_checkpoint_dir(config)
    if not os.path.exists(save_dir):
        return None
    
    checkpoints = glob(f"{save_dir}/*.pth")
    if not checkpoints:
        return None
    
    # Sort by modification time (descending)
    checkpoints = sorted(checkpoints, key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def load_checkpoint(model, checkpoint_path, config=None, optimizer=None, scheduler=None, device=torch.device('cuda'), scaler=None):
    """Load model checkpoint."""
    # Auto-select latest if path is None
    if checkpoint_path is None:
        if config is None:
            raise ValueError("config must be provided when checkpoint_path is None")
        latest = get_latest_checkpoint(config)
        if latest:
            _log_info(f"No checkpoint path specified, using latest: {latest}")
            checkpoint_path = latest
        else:
            raise FileNotFoundError(f"No checkpoint found in {get_checkpoint_dir(config)} directory")
    
    # Check if file exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    _log_info(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
    
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
    
    # Load AMP scaler state if provided
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        _log_info("AMP scaler state loaded")
    
    # Restore random states
    if 'random_states' in checkpoint:
        random_states = checkpoint['random_states']
        
        # 修复 torch RNG state
        if 'torch' in random_states:
            torch_state = random_states['torch']
            # 检查并转换类型
            if not isinstance(torch_state, torch.ByteTensor):
                print(f"Warning: torch RNG state is not ByteTensor (type={type(torch_state)}), attempting to convert")
                try:
                    # 尝试转换为 ByteTensor
                    if isinstance(torch_state, (list, tuple)):
                        torch_state = torch.ByteTensor(torch_state)
                        print("Successfully converted torch RNG state from list/tuple to ByteTensor")
                    elif isinstance(torch_state, torch.Tensor):
                        torch_state = torch_state.byte()
                        print("Successfully converted torch RNG state from Tensor to ByteTensor")
                    else:
                        _log_warning(f"Unknown torch RNG state type: {type(torch_state)}, skipping restore")
                        torch_state = None
                    # 确保在CPU上（RNG状态总是在CPU上）
                    if torch_state.device.type != 'cpu':
                        torch_state = torch_state.cpu()
                        _log_info("Moved to CPU")
                    
                    # 确保内存连续
                    if not torch_state.is_contiguous():
                        torch_state = torch_state.contiguous()
                        _log_info("Made contiguous")
                    
                    # 检查shape是否匹配
                    current_state = torch.get_rng_state()
                    if torch_state.shape != current_state.shape:
                        _log_warning(f"Shape mismatch: saved {torch_state.shape}, expected {current_state.shape}")
                        _log_info("Attempting to use saved state anyway...")

                except Exception as e:
                    _log_warning(f"Failed to convert torch RNG state: {e}, skipping restore")
                    torch_state = None
            
            if torch_state is not None:
                try:
                    torch.set_rng_state(torch_state)
                    _log_info("Torch RNG state restored")
                    print(f"Torch RNG state restored")
                except Exception as e:
                    _log_warning(f"Failed to restore torch RNG state: {e}")
        
        # 修复 CUDA RNG state
        if 'cuda' in random_states and random_states['cuda'] is not None:
            cuda_states = random_states['cuda']
            if torch.cuda.is_available() and cuda_states:
                try:
                    num_devices = torch.cuda.device_count()
                    current_states = torch.cuda.get_rng_state_all()
                    
                    _log_info(f"CUDA devices available: {num_devices}")
                    _log_info(f"Saved states count: {len(cuda_states)}")
                    _log_info(f"Current states count: {len(current_states)}")
                    
                    # Check if count matches
                    if len(cuda_states) != num_devices:
                        _log_warning(f"Device count mismatch: saved {len(cuda_states)}, current {num_devices}")
                        _log_info("Attempting to restore only available devices...")
                    
                    
                    # 确保每个状态都是正确的格式
                    fixed_states = []
                    for state in cuda_states:
                        if not isinstance(state, torch.ByteTensor):
                            if isinstance(state, torch.Tensor):
                                state = state.byte()
                                print("Successfully converted a CUDA RNG state from Tensor to ByteTensor")
                            elif isinstance(state, (list, tuple)):
                                state = torch.ByteTensor(state)
                                print("Successfully converted a CUDA RNG state from list/tuple to ByteTensor")
                        
                        if state.dtype != torch.uint8:
                            _log_info(f"Converting dtype from {state.dtype} to uint8")
                            state = state.byte()
                        
                        if state.device.type != 'cpu':
                            _log_info(f"Moving from {state.device} to CPU")
                            state = state.cpu()

                        # Check 4: Ensure contiguous memory layout
                        if not state.is_contiguous():
                            _log_info("Making tensor contiguous")
                            state = state.contiguous()
                        
                        # Check 5: Compare shape with reference
                        ref_state = current_states[fixed_states.__len__()] if fixed_states.__len__() < len(current_states) else None
                        if ref_state is not None and state.shape != ref_state.shape:
                            _log_warning(f"Shape mismatch: saved {state.shape}, expected {ref_state.shape}")
                            _log_info("Attempting to use saved state anyway...")
                            
                        fixed_states.append(state)
                    torch.cuda.set_rng_state_all(fixed_states)
                    _log_info("CUDA RNG state restored")
                    



                except Exception as e:
                    _log_warning(f"Failed to restore CUDA RNG state: {e}")
        
        # 恢复 numpy random state
        if 'np' in random_states:
            try:
                np.random.set_state(random_states['np'])
                _log_info("NumPy RNG state restored")
            except Exception as e:
                _log_warning(f"Failed to restore NumPy RNG state: {e}")
        
        # 恢复 python random state
        if 'random' in random_states:
            try:
                random.setstate(random_states['random'])
                _log_info("Python random state restored")
            except Exception as e:
                _log_warning(f"Failed to restore Python random state: {e}")
    
    history = checkpoint.get('history', {})
    config_dict = checkpoint.get('config', {})
    
    # Show saved fields including dataset and model size
    saved_fields = checkpoint.get('saved_fields', [])
    checkpoint_dataset = checkpoint.get('dataset', 'unknown')
    checkpoint_model_size = checkpoint.get('model_size', 'unknown')
    _log_info(f"Checkpoint - Model size: {checkpoint_model_size}, Dataset: {checkpoint_dataset}")
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
                config.checkpoint_path = get_latest_checkpoint(config)
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
                    
                    # Restore model_size and dataset from checkpoint (if not overridden by cmdline)
                    # Only restore if cmdline args didn't override
                    if not hasattr(args, 'model_size') or args.model_size is None:
                        if 'model_size' in saved_config_dict:
                            setattr(self.config, 'model_size', saved_config_dict['model_size'])
                            _log_info(f"Restored model_size from checkpoint: {saved_config_dict['model_size']}")
                    
                    if not hasattr(args, 'dataset') or args.dataset is None:
                        if 'dataset' in saved_config_dict:
                            setattr(self.config, 'dataset', saved_config_dict['dataset'])
                            _log_info(f"Restored dataset from checkpoint: {saved_config_dict['dataset']}")
                else:
                    _log_warning("No configuration found in checkpoint, using current config")
            else:
                raise RuntimeError("Resume mode requires checkpoint, but none found")
        
        # Recalculate dynamic config values (dim, depth, heads, mlp_dim, num_classes)
        # based on model_size and dataset selections from cmdline or checkpoint
        _log_info(f"Recalculating model configuration...")
        
        # Re-extract model configuration based on size
        if config.model_size == 'tiny':
            config.dim = 256
            config.depth = 4
            config.heads = 4
            config.mlp_dim = 256
        elif config.model_size == 'small':
            config.dim = 512
            config.depth = 6
            config.heads = 8
            config.mlp_dim = 512
        elif config.model_size == 'base':
            config.dim = 768
            config.depth = 12
            config.heads = 12
            config.mlp_dim = 768
        
        _log_info(f"  Model size: {config.model_size} -> dim={config.dim}, depth={config.depth}, heads={config.heads}")
        
        # Re-set num_classes based on dataset
        if config.dataset == 'cifar10':
            config.num_classes = 10
        elif config.dataset == 'cifar100':
            config.num_classes = 100
        _log_info(f"  Dataset: {config.dataset} -> num_classes={config.num_classes}")
        
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
        
        if config.model_type == 'vit':
            _log_info("use vit model")
            self.model = VisionTransformer(
                img_size=config.image_size,
                patch_size=config.patch_size,
                num_classes=config.num_classes,
                dim=config.dim,
                depth=config.depth,
                heads=config.heads,
                mlp_dim=config.mlp_dim,
                dropout=config.dropout,
                emb_dropout=config.emb_dropout,
                drop_path_rate=getattr(config, 'drop_path_rate', 0.0)
            ).to(self.device)
        elif config.model_type == 'hybrid':
            _log_info("use hybrid model")
            # 从 config 中读取混合模型的参数
            self.model = HybridVisionTransformer(
                num_classes=config.num_classes,
                cnn_stem_channels=getattr(config, 'hybrid_cnn_stem', 64),
                cnn_stage1_blocks=getattr(config, 'hybrid_cnn_stage1_blocks', 2),
                cnn_stage2_blocks=getattr(config, 'hybrid_cnn_stage2_blocks', 2),
                cnn_stage3_blocks=getattr(config, 'hybrid_cnn_stage3_blocks', 2),
                vit_dim=getattr(config, 'hybrid_vit_dim', 384),
                vit_depth=getattr(config, 'hybrid_vit_depth', 4),
                vit_heads=getattr(config, 'hybrid_vit_heads', 6),
                vit_mlp_ratio=getattr(config, 'hybrid_vit_mlp_ratio', 4),
                dropout=config.dropout,
                emb_dropout=config.emb_dropout,
                drop_path_rate=getattr(config, 'drop_path_rate', 0.0)
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model_type: {config.model_type}")
        
        # Data loaders (using restored config values)
        _log_info(f"Creating data loaders for {config.dataset} with batch_size={self.config.batch_size}")
        
        # Choose between Kornia (GPU) and torchvision (CPU) augmentations
        if getattr(config, 'use_kornia', False):
            _log_info("Using GPU-accelerated Kornia augmentations")
            self.train_loader, self.test_loader = get_dataloaders_with_kornia(
                dataset=config.dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                data_dir=config.data_dir,
                randaug_enabled=config.randaug_enabled,
                randaug_n=config.randaug_n,
                randaug_m=config.randaug_m,
                use_cutout=config.use_cutout,
                device=self.device,
                # color_jitter and geometric augs not supported in simplified Kornia version
            )
        else:
            _log_info("Using standard torchvision (CPU) augmentations")
            self.train_loader, self.test_loader = get_dataloaders(
                dataset=config.dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                data_dir=config.data_dir,
                randaug_enabled=config.randaug_enabled,
                randaug_n=config.randaug_n,
                randaug_m=config.randaug_m,
                use_cutout=config.use_cutout,
                cutout_length=config.cutout_length,
                use_color_jitter=config.use_color_jitter,
                color_jitter_brightness=config.color_jitter_brightness,
                color_jitter_contrast=config.color_jitter_contrast,
                color_jitter_saturation=config.color_jitter_saturation,
                color_jitter_hue=config.color_jitter_hue,
                use_random_rotation=config.use_random_rotation,
                rotation_degrees=config.rotation_degrees,
                use_random_affine=config.use_random_affine,
                affine_translate=config.affine_translate
            )
        
        # Loss and optimizer (using restored config values)
        self.label_smoothing = config.label_smoothing
        
        # Create criterion with optional label smoothing
        kwargs = {}
        if self.label_smoothing > 0:
            # Check if CrossEntropyLoss supports label_smoothing properly
            # The API was added in PyTorch 1.10
            try:
                # First check if the parameter exists in the function signature
                import inspect
                sig = inspect.signature(nn.CrossEntropyLoss.__init__)
                if 'label_smoothing' in sig.parameters:
                    # Try creating a test instance to ensure parameter works
                    test_criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
                    kwargs['label_smoothing'] = self.label_smoothing
                    _log_info(f"Label smoothing enabled: {self.label_smoothing}")
                else:
                    _log_warning(f"Label smoothing not supported by this PyTorch version (requires >= 1.10), disabling")
                    self.label_smoothing = 0.0
            except Exception as e:
                _log_warning(f"Label smoothing initialization failed: {e}")
                _log_warning(f"Label smoothing disabled")
                self.label_smoothing = 0.0
        else:
            _log_info("Label smoothing disabled")
            
        self.criterion = nn.CrossEntropyLoss(**kwargs)
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
        
        # AMP (Automatic Mixed Precision)
        self.use_amp = config.use_amp
        
        # Handle new PyTorch 2.x API for GradScaler
        if hasattr(torch.amp, 'GradScaler'):
            # PyTorch 2.x new API: torch.amp.GradScaler(device_type, args...)
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp if torch.cuda.is_available() and str(self.device) != 'cpu' else False)

        if self.use_amp:
            _log_info(f"AMP enabled: {self.use_amp}")
        else:
            _log_info("AMP disabled")
        
        # MixUp augmentation (batch-level)
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
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
        
        # Resume mode: load model, optimizer, scheduler states AFTER they are created
        if config.mode == 'resume' and config.checkpoint_path:
            _log_info(f"Loading model and training states from: {config.checkpoint_path}")
            
            # Restore random states from checkpoint (already loaded above)
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
                self.model, config.checkpoint_path, config,
                self.optimizer, self.scheduler, self.device,
                scaler=self.scaler
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
        
        # Log data augmentation configuration
        self._log_data_augmentation()
        
        # Log final configuration summary
        _log_info("=" * 60)
        _log_info("Final configuration for training:")
        _log_info(f"  Mode: {config.mode}")
        _log_info(f"  Device: {self.device}")
        _log_info(f"  Model size: {config.model_size}")
        _log_info(f"  Dataset: {config.dataset}")
        _log_info(f"  Epochs: {config.num_epochs}")
        _log_info(f"  Batch size: {config.batch_size}")
        _log_info(f"  Warmup epochs: {config.warmup_epochs}")
        _log_info(f"  Weight decay: {config.weight_decay}")
        _log_info(f"  Label smoothing: {self.label_smoothing}")
        _log_info(f"  AMP: {self.use_amp}")
        _log_info(f"  Gradient clipping: {getattr(config, 'grad_clip', 0.0)}")
        _log_info(f"  Convergence patience: {config.convergence_patience}")
        _log_info("=" * 60)
    
    def _log_data_augmentation(self):
        """Log data augmentation configuration."""
        config = self.config
        _log_info("-" * 60)
        _log_info("Data Augmentation Configuration:")
        
        # Basic augmentation
        _log_info(f"  Basic: RandomCrop(padding=4) + RandomHorizontalFlip")
        
        # RandAugment
        if getattr(config, 'randaug_enabled', False):
            _log_info(f"  RandAugment: enabled (n={getattr(config, 'randaug_n', 2)}, m={getattr(config, 'randaug_m', 9)})")
        else:
            _log_info(f"  RandAugment: disabled")
        
        # Cutout
        if getattr(config, 'use_cutout', False):
            _log_info(f"  Cutout: enabled (length={getattr(config, 'cutout_length', 16)})")
        else:
            _log_info(f"  Cutout: disabled")
        
        # Color Jitter
        if getattr(config, 'use_color_jitter', False):
            _log_info(f"  ColorJitter: enabled")
            _log_info(f"    brightness={getattr(config, 'color_jitter_brightness', 0.2)}")
            _log_info(f"    contrast={getattr(config, 'color_jitter_contrast', 0.2)}")
            _log_info(f"    saturation={getattr(config, 'color_jitter_saturation', 0.2)}")
            _log_info(f"    hue={getattr(config, 'color_jitter_hue', 0.1)}")
        else:
            _log_info(f"  ColorJitter: disabled")
        
        # Rotation
        if getattr(config, 'use_random_rotation', False):
            _log_info(f"  RandomRotation: enabled (degrees=±{getattr(config, 'rotation_degrees', 15)})")
        else:
            _log_info(f"  RandomRotation: disabled")
        
        # Affine
        if getattr(config, 'use_random_affine', False):
            _log_info(f"  RandomAffine: enabled (translate={getattr(config, 'affine_translate', 0.1)})")
        else:
            _log_info(f"  RandomAffine: disabled")
        
        # MixUp (batch-level augmentation)
        if getattr(config, 'use_mixup', False):
            _log_info(f"  MixUp: enabled (alpha={getattr(config, 'mixup_alpha', 0.2)}, prob={getattr(config, 'mixup_prob', 0.5)})")
        else:
            _log_info(f"  MixUp: disabled")
        
        _log_info("-" * 60)
        
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
        
        # Forward pass with AMP
        with torch.amp.autocast("cuda",enabled=self.use_amp if torch.cuda.is_available() and str(self.device) != 'cpu' else False):
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
        print(f"Forward pass successful! Loss: {loss.item():.4f}")
        
        # Backward pass with AMP
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        print("Backward pass successful! Weights updated.")
        
        # Test one forward pass
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images)
            _, predicted = outputs.max(1)
            accuracy = (predicted == labels).sum().item() / labels.size(0)
            print(f"eval pass successful! Accuracy on batch: {100 * accuracy:.2f}%")
    
    def train_epoch(self, epoch):
        """Train for one epoch with optional MixUp."""
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
        
        for images, labels in pbar:
            images, labels = images.to(self.device,non_blocking=True), labels.to(self.device,non_blocking=True)
            mixup_applied = False
            
            # Apply MixUp if enabled and random check passes
            if self.mixup is not None and random.random() < self.mixup_prob:
                images, labels = self.mixup(images, labels)
                mixup_applied = True
                # Note: labels are now one-hot distributions
            
            # Forward pass with AMP autocast
            with torch.amp.autocast("cuda",enabled=self.use_amp if torch.cuda.is_available() and str(self.device) != 'cpu' else False):
                outputs = self.model(images)
                
                # Special handling for MixUp loss
                if mixup_applied:
                    # MixUp requires KL divergence loss with one-hot labels
                    loss = F.kl_div(
                        F.log_softmax(outputs, dim=1),
                        labels,  # Already one-hot from MixUp
                        reduction='batchmean',
                        log_target=False
                    )
                else:
                    # Standard CrossEntropy loss
                    loss = self.criterion(outputs, labels)
            
            # Backward pass with AMP scaler
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping (if enabled)
            if hasattr(self.config, 'grad_clip') and self.config.grad_clip > 0:
                # Unscale gradients before clipping when using AMP
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            # Optimizer step with AMP scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.scheduler.step()
            
            # Statistics - special handling for MixUp
            train_loss += loss.item()
            
            # Calculate accuracy for display
            if mixup_applied:
                # For MixUp, convert one-hot back to class indices for display accuracy
                _, pred_labels = labels.max(1)
            else:
                pred_labels = labels
            
            _, predicted = outputs.max(1)
            total += pred_labels.size(0)
            correct += predicted.eq(pred_labels).sum().item()
            
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
        """Evaluate on test set.
        
        Note: MixUp is NOT used during evaluation (only for training).
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Apply AMP autocast (optional)
                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda",enabled=self.use_amp):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
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
        _log_info(f"Label smoothing: {self.label_smoothing}")
        
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
                    filename_prefix='best',
                    scaler=self.scaler
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
            filename_prefix='final',
            scaler=self.scaler
        )


def test_mode(config):
    """Test mode: evaluate a saved model."""
    _log_info("Testing mode...")
    
    # Auto-select latest checkpoint if not specified
    if config.checkpoint_path is None:
        config.checkpoint_path = get_latest_checkpoint(config)
    
    if config.checkpoint_path is None:
        raise RuntimeError(f"Test mode requires checkpoint, but none found in {get_checkpoint_dir(config)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu')
    
    checkpoint = torch.load(config.checkpoint_path, map_location='cpu',weights_only=False)
    checkpoint_path = config.checkpoint_path
    config = Config.from_dict(checkpoint.get('config', {}))
    print(config)

    #使用checkpoint中的配置创建模型（确保与训练时一致）
    model = VisionTransformer(
        img_size=config.image_size,
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        mlp_dim=config.mlp_dim,
        dropout=config.dropout,
        emb_dropout=config.emb_dropout,
        drop_path_rate=getattr(config, 'drop_path_rate', 0.0)
    ).to(device)
    
    # Load checkpoint
    load_checkpoint(model, checkpoint_path, config, device=device)
    
    # Use training time's test_loader (use loaded dataset config)
    _, test_loader = get_dataloaders(
        dataset=config.dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        data_dir=config.data_dir,
        # Note: Test loader uses default transforms (no augmentation)
    )
    
    # Evaluate (using same criterion as training)
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    # Create criterion with same label smoothing as training
    kwargs = {}
    if config.label_smoothing > 0:
        try:
            import inspect
            sig = inspect.signature(nn.CrossEntropyLoss.__init__)
            if 'label_smoothing' in sig.parameters:
                test_criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
                kwargs['label_smoothing'] = config.label_smoothing
        except Exception:
            pass
    
    criterion = nn.CrossEntropyLoss(**kwargs)
    
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
    parser = argparse.ArgumentParser(description='Train or test ViT on CIFAR-10/CIFAR-100')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['dry_run', 'train', 'test', 'resume'],
                       help='Training mode: dry_run, train, test, resume')
    parser.add_argument('--model-type', type=str, default='vit', choices=['vit', 'hybrid'],
                    help='Model architecture: vit or hybrid')
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
    parser.add_argument('--model-size', type=str, default=None, choices=['tiny', 'small', 'base'],
                       help='Model size: tiny, small, or base')
    parser.add_argument('--dataset', type=str, default=None, choices=['cifar10', 'cifar100'],
                       help='Dataset: cifar10 or cifar100')
    parser.add_argument('--label-smoothing', type=float, default=None,
                       help='Label smoothing epsilon (0.0 to 0.5, typical 0.1)')
    
    # AMP
    parser.add_argument('--amp', action='store_true', default=False,
                       help='Enable Automatic Mixed Precision (AMP) training')

    # EMA
    parser.add_argument('--ema-decay', type=float, default=None,
                       help='EMA decay rate (0.0=disabled, recommend 0.9999)')

    # Gradient Clipping
    parser.add_argument('--grad-clip', type=float, default=None,
                       help='Max gradient norm (0.0=disabled, recommend 1.0)')

    # RandAugment
    parser.add_argument('--randaug', action='store_true', default=False,
                       help='Enable RandAugment data augmentation')
    parser.add_argument('--randaug-n', type=int, default=None,
                       help='RandAugment: number of transformations')
    parser.add_argument('--randaug-m', type=int, default=None,
                       help='RandAugment: magnitude (1-10)')

    # Cutout
    parser.add_argument('--cutout', action='store_true', default=False,
                       help='Enable Cutout/RandomErasing augmentation')
    parser.add_argument('--cutout-length', type=int, default=None,
                       help='Cutout patch length in pixels')
    
    # Color Jitter
    parser.add_argument('--color-jitter', action='store_true', default=False,
                       help='Enable ColorJitter augmentation (brightness/contrast/saturation/hue)')
    parser.add_argument('--color-jitter-brightness', type=float, default=None,
                       help='ColorJitter brightness jitter range (default 0.2)')
    parser.add_argument('--color-jitter-contrast', type=float, default=None,
                       help='ColorJitter contrast jitter range (default 0.2)')
    parser.add_argument('--color-jitter-saturation', type=float, default=None,
                       help='ColorJitter saturation jitter range (default 0.2)')
    parser.add_argument('--color-jitter-hue', type=float, default=None,
                       help='ColorJitter hue jitter range (default 0.1)')
    
    # Geometric augmentation
    parser.add_argument('--rotation', action='store_true', default=False,
                       help='Enable random rotation augmentation')
    parser.add_argument('--rotation-degrees', type=float, default=None,
                       help='Max rotation degrees (default 15)')
    parser.add_argument('--affine', action='store_true', default=False,
                       help='Enable random affine transformation')
    parser.add_argument('--affine-translate', type=float, default=None,
                       help='Max translation as fraction of image (default 0.1)')
    
    # MixUp augmentation
    parser.add_argument('--mixup', action='store_true', default=False,
                       help='Enable MixUp augmentation')
    parser.add_argument('--mixup-alpha', type=float, default=None,
                       help='MixUp beta distribution alpha (default 0.2)')
    parser.add_argument('--mixup-prob', type=float, default=None,
                       help='Probability of applying MixUp to a batch (default 0.5)')
    
    # Stochastic Depth (DropPath)
    parser.add_argument('--drop-path', type=float, default=None,
                        help='Stochastic Depth drop rate (0.0=disabled, recommend 0.1-0.2)')
     
     # Kornia GPU Augmentations
    parser.add_argument('--kornia', action='store_true', default=False,
                        help='Enable GPU-accelerated augmentations with Kornia (requires: pip install kornia)')
    
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Enable verbose logging')
    parser.add_argument('--debug', action='store_true', default=False,
                       help='Enable debug logging')
    return parser.parse_args()


if __name__ == "__main__":
    
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    sys.stdout.flush()
        
    args = parse_args()
    config = Config.from_args(args)
    
    # Initialize logging BEFORE creating trainer
    force_new_log = (config.mode != 'resume')
    
    # For dry_run, show INFO to console for visibility
    console_level = logging.INFO if config.mode == 'dry_run' else logging.WARNING
    
    try:
        init_logging(
            verbose=config.verbose, 
            debug=config.debug, 
            log_to_file=config.mode != 'dry_run',
            console_output=True,
            console_level=console_level,
            force_new_log=force_new_log
        )
        print(f"init_logging OK")
    except Exception as e:
        print(f"init_logging failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"Creating Trainer...")
    sys.stdout.flush()
    
    trainer = Trainer(config)
    
    try:
        if config.mode == 'dry_run':
            print(f"Starting dry_run...")
            trainer.dry_run()
        elif config.mode == 'test':
            print(f"Starting test...")
            if config.checkpoint_path is None:
                config.checkpoint_path = get_latest_checkpoint(config)
            
            if config.checkpoint_path is None:
                raise RuntimeError(f"Test mode requires checkpoint, but none found in {get_checkpoint_dir(config)}")
            
            test_mode(config)
        else:  # train or resume
            print(f"Starting train...")
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
