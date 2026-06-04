"""Utility functions for ResNet training on CIFAR-10/CIFAR-100"""

import matplotlib.pyplot as plt
import numpy as np
import torch

import logging
import sys
from datetime import datetime
import os


class ImmediateFileHandler(logging.FileHandler):
    """File handler that flushes immediately after each log message."""
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logger(name='ViT-Trainer', level=logging.INFO, log_to_file=True, log_dir='logs', 
                 console_output=True, console_min_level=logging.WARNING, force_new_log=True):
    """
    Setup logger with console and file handlers.
    
    Args:
        name: Logger name
        level: Logging level (logging.INFO, logging.DEBUG, etc.)
        log_to_file: Whether to save logs to file
        log_dir: Directory to save log files
        console_output: Whether to output to console
        console_min_level: Minimum level for console output (default: WARNING)
        force_new_log: Create new log file (don't append to existing)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
    
    logger.setLevel(logging.DEBUG)
    
    # Use a simpler formatter without timestamps for better readability
    # (timestamps will be added by the handlers)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler - only show WARNING and above by default
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_min_level)  # Only show logs >= console_min_level
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler - show all logs (with immediate flush)
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        if force_new_log:
            # Create new log file with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'training_{timestamp}.log'
        else:
            # Use fixed log name (for resume sessions)
            log_filename = 'training_current.log'
        log_path = os.path.join(log_dir, log_filename)
        # Use custom handler that flushes immediately
        file_handler = ImmediateFileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG if level <= logging.DEBUG else logging.INFO)  # File gets ALL logs
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Also create a symlink to latest log
        latest_link = os.path.join(log_dir, 'latest.log')
        if os.path.exists(latest_link) or os.path.islink(latest_link):
            os.unlink(latest_link)
        try:
            os.symlink(log_filename, latest_link)
        except:
            # Windows might not support symlinks, create a text file with path
            with open(os.path.join(log_dir, 'latest.txt'), 'w') as f:
                f.write(log_path)
    
    return logger

# Global logger instance
_logger = None


def get_logger():
    """Get or create the global logger instance."""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


def _log_info(msg):
    """Log info-level message."""
    logger = get_logger()
    logger.info(msg)
    for handler in logger.handlers:
        if isinstance(handler, ImmediateFileHandler):
            handler.flush()


def _log_debug(msg):
    """Log debug-level message."""
    logger = get_logger()
    logger.debug(msg)
    for handler in logger.handlers:
        if isinstance(handler, ImmediateFileHandler):
            handler.flush()


def _log_warning(msg):
    """Log warning-level message."""
    logger = get_logger()
    logger.warning(msg)
    for handler in logger.handlers:
        handler.flush()


def _log_error(msg):
    """Log error-level message."""
    logger = get_logger()
    logger.error(msg)
    for handler in logger.handlers:
        handler.flush()


def init_logging(verbose=False, debug=False, log_to_file=True, log_dir='logs',
                 console_output=True, console_level=logging.INFO, force_new_log=True):
    """Initialize logging with configurable verbosity."""
    global _logger
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    _logger = setup_logger(
        level=level, log_to_file=log_to_file, log_dir=log_dir,
        console_output=console_output, console_min_level=console_level,
        force_new_log=force_new_log
    )

    if debug:
        _log_debug("Debug logging enabled")
    elif verbose:
        _log_info("Verbose logging enabled")


def plot_results(history):
    """Plot training and testing curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['test_loss'], 'r-', label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Testing Loss')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[1].plot(epochs, history['test_acc'], 'r-', label='Test Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Testing Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300)
    plt.show()


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def visualize_patches(image, patch_size=4):
    """Visualize how an image is split into patches."""
    import torchvision.transforms as transforms
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    image_np = np.array(image)
    H, W = image_np.shape[:2]
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_np)
    for i in range(num_patches_h + 1):
        ax.axhline(y=i * patch_size, color='red', linewidth=1.5)
    for j in range(num_patches_w + 1):
        ax.axvline(x=j * patch_size, color='red', linewidth=1.5)
    ax.set_title(f'Image split into {num_patches_h}x{num_patches_w}={num_patches_h*num_patches_w} patches')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('patches_visualization.png', dpi=300)
    plt.show()
    return num_patches_h * num_patches_w