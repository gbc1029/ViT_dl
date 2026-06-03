"""Configuration for ResNet on CIFAR-10/CIFAR-100 (matching ViT training pipeline)"""

class Config:
    def __init__(self, dataset='cifar10', model_type='resnet_tiny', **kwargs):
        # Dataset selection: 'cifar10' or 'cifar100'
        self.dataset = dataset or 'cifar10'

        # Model selection: 'resnet_tiny' or 'resnet_small'
        self.model_type = model_type or 'resnet_tiny'

        # Number of classes is derived from dataset
        self.num_classes = 10 if self.dataset == 'cifar10' else 100

        self.batch_size = 64
        self.num_epochs = 100
        self.learning_rate = 3e-4   # AdamW LR (same as ViT)
        self.weight_decay = 0.03    # AdamW weight decay (same as ViT)
        self.warmup_epochs = 5      # Warmup epochs (same as ViT)

        # Learning rate milestones for MultiStepLR
        self.lr_milestones = [60, 90, 120]
        self.lr_gamma = 0.2

        # Data parameters
        self.num_workers = 4
        self.data_dir = './data'

        # Device
        self.device = 'cuda'

        # Training mode
        self.mode = 'train'  # Options: 'dry_run', 'train', 'test', 'resume'
        self.checkpoint_path = None  # Path to checkpoint for test or resume

        # Convergence check (double check)
        self.convergence_threshold = 0.001
        self.convergence_patience = 5
        self.param_change_threshold = 1e-6

        # Label smoothing
        self.label_smoothing = 0.0  # Recommended: 0.1 when enabled

        # AMP (Auto Mixed Precision)
        self.use_amp = False

        # EMA (Exponential Moving Average)
        self.ema_decay = 0.0  # 0.0 = disabled. Recommended: 0.9999

        # Gradient clipping
        self.grad_clip = 0.0  # 0.0 = disabled. Recommended: 1.0

        # Data augmentation
        self.randaug_enabled = False
        self.randaug_n = 2
        self.randaug_m = 9
        self.use_cutout = False
        self.cutout_length = 16
        self.use_color_jitter = False
        self.color_jitter_brightness = 0.2
        self.color_jitter_contrast = 0.2
        self.color_jitter_saturation = 0.2
        self.color_jitter_hue = 0.1
        self.use_random_rotation = False
        self.rotation_degrees = 15
        self.use_random_affine = False
        self.affine_translate = 0.1
        self.use_mixup = False
        self.mixup_alpha = 0.2
        self.mixup_prob = 0.5
        self.use_kornia = False

        # Checkpoint selection
        self.checkpoint_selection = 'latest'

        # Logging control
        self.verbose = True
        self.debug = False

        #self.update(vars(kwargs))
        self._normalize()

    #def update(self, options):
    #    """Update config values from a dictionary of overrides."""
    #    for key, value in options.items():
    #        if value is not None:
    #            setattr(self, key, value)
    #    self.num_classes = 10 if self.dataset == 'cifar10' else 100

    def _normalize(self):
        self.dataset = str(self.dataset).lower()
        if self.dataset not in ('cifar10', 'cifar100'):
            raise ValueError("dataset must be 'cifar10' or 'cifar100'")

        self.model_type = str(self.model_type).lower()
        if self.model_type not in ('resnet_tiny', 'resnet_small'):
            raise ValueError("model_type must be 'resnet_tiny' or 'resnet_small'")

        self.num_classes = 10 if self.dataset == 'cifar10' else 100

