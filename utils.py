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


def setup_logger(name='ResNet-Trainer', level=logging.INFO, log_to_file=True,
                 log_dir='logs', console_output=True, console_min_level=logging.INFO,
                 force_new_log=True):
    """Set up logger with console and file handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_min_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        if force_new_log:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'training_{timestamp}.log'
        else:
            log_filename = 'training_current.log'
        log_path = os.path.join(log_dir, log_filename)
        file_handler = ImmediateFileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

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