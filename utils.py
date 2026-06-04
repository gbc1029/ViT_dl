"""Utility functions for ViT training and evaluation"""

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
        self.flush()  # Force flush to disk immediately

class LevelFilter(logging.Filter):
    """Filter logs by level for console output."""
    
    def __init__(self, min_level, max_level=None):
        self.min_level = min_level
        self.max_level = max_level
    
    def filter(self, record):
        if self.max_level is None:
            return record.levelno >= self.min_level
        return self.min_level <= record.levelno <= self.max_level


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
_log_file_path = None


def get_logger():
    """Get or create the global logger instance."""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


def get_log_file_path():
    """Get the current log file path."""
    global _log_file_path
    return _log_file_path


def _log_info(msg, *args, **kwargs):
    """Log info level message to file only (not console by default)."""
    logger = get_logger()
    if args or kwargs:
        logger.info(msg, *args, **kwargs)
    else:
        logger.info(msg)
    
    # Flush file handler
    for handler in logger.handlers:
        if isinstance(handler, ImmediateFileHandler):
            handler.flush()


def _log_debug(msg, *args, **kwargs):
    """Log debug level message to file only (not console by default)."""
    logger = get_logger()
    if args or kwargs:
        logger.debug(msg, *args, **kwargs)
    else:
        logger.debug(msg)
    
    # Flush file handler
    for handler in logger.handlers:
        if isinstance(handler, ImmediateFileHandler):
            handler.flush()


def _log_warning(msg, *args, **kwargs):
    """Log warning level message to both file and console."""
    logger = get_logger()
    if args or kwargs:
        logger.warning(msg, *args, **kwargs)
    else:
        logger.warning(msg)
    
    # Flush all handlers
    for handler in logger.handlers:
        handler.flush()


def _log_error(msg, *args, **kwargs):
    """Log error level message to both file and console."""
    logger = get_logger()
    if args or kwargs:
        logger.error(msg, *args, **kwargs)
    else:
        logger.error(msg)
    
    # Flush all handlers
    for handler in logger.handlers:
        handler.flush()


def set_log_level(level):
    """
    Set log level for the global logger.
    
    Args:
        level: logging.INFO, logging.DEBUG, logging.WARNING, etc.
    """
    logger = get_logger()
    logger.setLevel(level)


def set_console_level(level):
    """
    Set console output minimum level.
    
    Args:
        level: logging.INFO, logging.WARNING, logging.ERROR, etc.
    """
    logger = get_logger()
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(level)
            break


def init_logging(verbose=False, debug=False, log_to_file=True, log_dir='logs', 
                 console_output=True, console_level=logging.WARNING, force_new_log=True):
    """
    Initialize logging with configurable verbosity.
    
    Args:
        verbose: Enable verbose logging (INFO level to file)
        debug: Enable debug logging (DEBUG level to file)
        log_to_file: Save logs to file
        log_dir: Directory for log files
        console_output: Output to console
        console_level: Minimum level for console output (default: WARNING)
        force_new_log: Create new log file (don't append)
    """
    global _logger, _log_file_path
    
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    _logger = setup_logger(
        level=level, 
        log_to_file=log_to_file, 
        log_dir=log_dir,
        console_output=console_output,
        console_min_level=console_level,
        force_new_log=force_new_log
    )
    
    # Get log file path from handlers
    for handler in _logger.handlers:
        if isinstance(handler, ImmediateFileHandler):
            _log_file_path = handler.baseFilename
            break
    
    # Log startup info (these go to file only since console level is WARNING)
    if debug:
        _log_debug("Debug logging enabled")
        _log_debug(f"Log file: {_log_file_path}")
        _log_debug(f"Console output level: {logging.getLevelName(console_level)}")
    elif verbose:
        _log_info("Verbose logging enabled")
        _log_info(f"Log file: {_log_file_path}")
        _log_info(f"Console output level: {logging.getLevelName(console_level)}")


class TrainingLogger:
    """Context manager for training session logging with real-time flushing."""
    
    def __init__(self, experiment_name=None, log_dir='logs', verbose=True):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.verbose = verbose
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        _log_info("=" * 80)
        _log_info(f"Training session started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.experiment_name:
            _log_info(f"Experiment: {self.experiment_name}")
        _log_info("=" * 80)
        
        # Force flush
        self._flush_file_handlers()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = end_time - self.start_time
        _log_info("=" * 80)
        _log_info(f"Training session ended at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        _log_info(f"Total duration: {duration}")
        if exc_type is not None:
            _log_error(f"Session ended with error: {exc_type.__name__}: {exc_val}")
        _log_info("=" * 80)
        
        # Force flush before exit
        self._flush_file_handlers()
    
    def _flush_file_handlers(self):
        """Flush only file handlers."""
        logger = get_logger()
        for handler in logger.handlers:
            if isinstance(handler, ImmediateFileHandler):
                handler.flush()
    
    def log_epoch(self, epoch, train_loss, train_acc, test_loss, test_acc, lr):
        """Log epoch metrics (to file only)."""
        _log_info(f"Epoch {epoch:3d} | "
                 f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                 f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
                 f"LR: {lr:.2e}")
        self._flush_file_handlers()


# Console-only print for important messages that should always be visible
def console_print(msg, level='info'):
    """
    Print to console only (not logged to file).
    Use this for progress bars or temporary messages.
    """
    print(msg)


def log_print(msg, level='info'):
    """
    Print to both console and file.
    Use this for important messages that need attention.
    """
    print(msg)
    if level == 'info':
        _log_info(msg)
    elif level == 'debug':
        _log_debug(msg)
    elif level == 'warning':
        _log_warning(msg)
    elif level == 'error':
        _log_error(msg)


def plot_results(history):
    """
    Plot training and testing curves.
    
    Args:
        history: Dictionary containing training history
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['test_loss'], 'r-', label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Testing Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
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
    """
    Visualize how an image is split into patches.
    
    Args:
        image: Tensor of shape (C, H, W)
        patch_size: Size of each patch
    """
    import torchvision.transforms as transforms
    
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    
    image_np = np.array(image)
    H, W = image_np.shape[:2]
    
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_np)
    
    # Draw patch boundaries
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
