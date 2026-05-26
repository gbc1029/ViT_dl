"""Training utilities for ViT and ResNet models"""

import os
import torch
from datetime import datetime


def get_timestamp_filename(base_name, ext='pth'):
    """
    Generate filename with timestamp.
    
    Args:
        base_name: Base name for the file
        ext: File extension
    
    Returns:
        Filename with timestamp
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base_name}_{timestamp}.{ext}"


def save_checkpoint(model, optimizer, scheduler, history, config, 
                    filename='checkpoint.pth'):
    """
    Save model checkpoint with timestamp.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        history: Training history
        config: Configuration
        filename: Base filename
    
    Returns:
        Path to saved checkpoint
    """
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    # Add timestamp to filename
    save_path = os.path.join(save_dir, get_timestamp_filename(
        filename.replace('.pth', '')
    ))
    
    # Prepare checkpoint dict
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': config.__dict__ if hasattr(config, '__dict__') else config
    }
    
    # Add scheduler if exists
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to: {save_path}")
    return save_path


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None, 
                   device='cuda'):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        optimizer: Optimizer to load state (optional)
        scheduler: Scheduler to load state (optional)
        device: Device to load to
    
    Returns:
        Dictionary with loaded checkpoint
    """
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
    return {
        'history': history,
        'config': config
    }


def evaluate_model(model, test_loader, criterion=None, device='cuda'):
    """
    Evaluate model on test set.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        criterion: Loss criterion (optional)
        device: Device to use
    
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            # Calculate loss if criterion provided
            if criterion is not None:
                loss = criterion(outputs, labels)
                test_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    results = {
        'accuracy': 100. * correct / total,
        'correct': correct,
        'total': total
    }
    
    if criterion is not None:
        results['loss'] = test_loss / len(test_loader)
    
    return results


def test_all_classes(model, test_loader, class_names=None, device='cuda'):
    """
    Evaluate model and show per-class accuracy.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        class_names: List of class names
        device: Device to use
    """
    model.eval()
    
    num_classes = len(class_names) if class_names else 10
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    confusion_matrix = torch.zeros(num_classes, num_classes)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1
                confusion_matrix[label][predicted[i].item()] += 1
    
    print("\nPer-class accuracy:")
    print("-" * 50)
    for i in range(num_classes):
        acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        class_name = class_names[i] if class_names else f"Class {i}"
        print(f"{class_name:15s}: {acc:.2f}% ({class_correct[i]:4d}/{class_total[i]:4d})")
    print("-" * 50)
    
    # Overall accuracy
    total_correct = sum(class_correct)
    total_samples = sum(class_total)
    print(f"\nOverall accuracy: {100. * total_correct / total_samples:.2f}% ({total_correct}/{total_samples})")
    
    return confusion_matrix
