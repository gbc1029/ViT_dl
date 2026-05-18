"""Utility functions for ViT training and evaluation"""

import matplotlib.pyplot as plt
import numpy as np


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
