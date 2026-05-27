"""Configuration for ViT on CIFAR-10/CIFAR-100"""

class Config:
    # Dataset selection: 'cifar10' or 'cifar100'
    dataset = 'cifar10'
    
    # Model size: 'tiny' (t), 'small' (s), 'base' (b)
    model_size = 'small'
    
    # Extract model configuration based on size
    if model_size == 'tiny':
        # Tiny ViT: ~5M parameters
        dim = 256
        depth = 4
        heads = 4
        mlp_dim = 256
    elif model_size == 'small':
        # Small ViT: ~10M parameters (current default)
        dim = 512
        depth = 6
        heads = 8
        mlp_dim = 512
    elif model_size == 'base':
        # Base ViT: ~22M parameters
        dim = 768
        depth = 12
        heads = 12
        mlp_dim = 768
    else:
        raise ValueError(f"Invalid model_size: {model_size}. Must be 'tiny', 'small', or 'base'")
    
    # Model parameters (auto-set based on model_size)
    image_size = 32
    patch_size = 4
    
    # Number of classes (auto-set based on dataset)
    num_classes = 10 if dataset == 'cifar10' else 100
    
    dropout = 0.1
    emb_dropout = 0.1
    drop_path_rate = 0.0  # Stochastic Depth drop rate (0.0 = disabled). Recommended: 0.1-0.2
    
    # Training parameters
    batch_size = 64
    num_epochs = 100
    learning_rate = 3e-4
    weight_decay = 0.03
    warmup_epochs = 5
    
    # Data parameters
    num_workers = 4
    data_dir = './data'
    
    # Device
    device = 'cuda'
    
    # Training mode
    mode = 'train'  # Options: 'dry_run', 'train', 'test', 'resume'
    checkpoint_path = None  # Path to checkpoint for test or resume
    
    # Convergence check (double check)
    convergence_threshold = 0.001  # Accuracy improvement threshold
    convergence_patience = 5  # Consecutive epochs without improvement
    param_change_threshold = 1e-6  # Parameter change threshold
    
    # Label smoothing (requires PyTorch >= 1.10, default: disabled)
    label_smoothing = 0.0  # Recommended: 0.1 when enabled
    
    # Enhanced methods: AMP (Auto Mixed Precision)
    use_amp = False  # Enable Automatic Mixed Precision for faster training and less memory usage

    # Enhanced methods: EMA (Exponential Moving Average)  
    ema_decay = 0.0  # EMA decay rate (0.0 = disabled). Recommended: 0.999 or 0.9999 when enabled

    # Enhanced methods: Gradient Clipping
    grad_clip = 0.0  # Max gradient norm (0.0 = disabled). Recommended: 1.0 for ViT when enabled

    # Enhanced methods: Data Augmentation
    # RandAugment parameters
    randaug_enabled = False  # Enable RandAugment
    randaug_n = 2           # Number of augmentation transformations
    randaug_m = 9           # Magnitude of augmentation (1-10)

     # Advanced data augmentation options
    use_cutout = False         # Enable Cutout/RandomErasing
    cutout_length = 16         # Cutout patch size (pixels)
     
    # GPU-accelerated augmentations with Kornia (experimental)
    use_kornia = True         # Enable GPU-based Kornia augmentations (requires: pip install kornia)
    # Note: When use_kornia=True, augmentations are applied on GPU for better performance
    
    # Basic augmentation enhancements
    use_color_jitter = False   # Enable ColorJitter (brightness/contrast/saturation/hue)
    color_jitter_brightness = 0.2   # Brightness jitter range [max(0, 1 - brightness), 1 + brightness]
    color_jitter_contrast = 0.2     # Contrast jitter range [max(0, 1 - contrast), 1 + contrast]
    color_jitter_saturation = 0.2   # Saturation jitter range [max(0, 1 - saturation), 1 + saturation]
    color_jitter_hue = 0.1          # Hue jitter range (-hue, +hue), should be in [-0.5, 0.5]
    
    # Geometric augmentation
    use_random_rotation = False   # Enable random rotation
    rotation_degrees = 15         # Max rotation degrees
    
    use_random_affine = False     # Enable random affine transformation
    affine_translate = 0.1        # Max translation as fraction of image size
    
    # MixUp augmentation (batch-level mixing, applied in training loop)
    use_mixup = False        # Enable MixUp augmentation
    mixup_alpha = 0.2        # Beta distribution alpha parameter (typical 0.2)
    mixup_prob = 0.5         # Probability of applying MixUp to a batch (0.5 = 50% chance)
    
    # Checkpoint selection (default: auto-select latest if not specified)
    checkpoint_selection = 'latest'  # 'exact' or 'latest' (for resume/test mode)
    
    # Logging control
    verbose = True  # Enable verbose logging
    debug = False  # Debug logging (default: off)