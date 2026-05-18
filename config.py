"""Configuration for ViT on CIFAR-10"""

class Config:
    # Model parameters
    image_size = 32
    patch_size = 4
    num_classes = 10
    dim = 512
    depth = 6
    heads = 8
    mlp_dim = 512
    dropout = 0.1
    emb_dropout = 0.1
    
    # Training parameters
    batch_size = 64
    num_epochs = 30
    learning_rate = 3e-4
    weight_decay = 0.03
    warmup_epochs = 5
    
    # Data parameters
    num_workers = 4
    
    # Device
    device = 'cuda'
