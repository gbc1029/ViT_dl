"""Configuration for ResNet on CIFAR-10"""

class Config:
    # Model parameters
    num_classes = 10
    model_type = 'resnet18'  # 'resnet18' or 'resnet34'
    
    # Training parameters
    batch_size = 128
    num_epochs = 200  # ResNet needs more epochs
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    
    # Learning rate milestones for MultiStepLR
    lr_milestones = [100, 150]
    lr_gamma = 0.1
    
    # Data parameters
    num_workers = 4
    
    # Device
    device = 'cuda'
    
    # Training mode
    mode = 'train'  # Options: 'dry_run', 'train', 'test', 'resume'
    checkpoint_path = None  # Path to checkpoint for test or resume
