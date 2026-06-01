"""Configuration for ViT on CIFAR-10/CIFAR-100"""

class Config:
    def __init__(self):
        # Dataset selection
        self.dataset = 'cifar10'
        self.model_size = 'small'
        
        # 动态设置模型参数
        self._set_model_params()
        
        # 动态设置类别数
        self._set_num_classes()
        
        # Model parameters
        self.image_size = 32
        self.patch_size = 4
        
        self.dropout = 0.1
        self.emb_dropout = 0.1
        self.drop_path_rate = 0.0
        
        # Training parameters
        self.batch_size = 64
        self.num_epochs = 100
        self.learning_rate = 3e-4
        self.weight_decay = 0.03
        self.warmup_epochs = 5
        
        # Data parameters
        self.num_workers = 4
        self.data_dir = './data'
        
        # Device
        self.device = 'cuda'
        
        # Training mode
        self.mode = 'train'
        self.checkpoint_path = None
        
        # Convergence check
        self.convergence_threshold = 0.001
        self.convergence_patience = 5
        self.param_change_threshold = 1e-6
        
        # Label smoothing
        self.label_smoothing = 0.0
        
        # Enhanced methods
        self.use_amp = False
        self.ema_decay = 0.0
        self.grad_clip = 0.0
        
        # Data augmentation
        self.randaug_enabled = False
        self.randaug_n = 2
        self.randaug_m = 9
        
        self.use_cutout = False
        self.cutout_length = 16
        
        self.use_kornia = True
        
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
        
        self.checkpoint_selection = 'latest'
        
        # Logging
        self.verbose = True
        self.debug = False
    def _set_model_params(self):
        """Set model architecture parameters based on model_size."""
        if self.model_size == 'tiny':
            self.dim = 256
            self.depth = 4
            self.heads = 4
            self.mlp_dim = 256
        elif self.model_size == 'small':
            self.dim = 512
            self.depth = 6
            self.heads = 8
            self.mlp_dim = 512
        elif self.model_size == 'base':
            self.dim = 768
            self.depth = 12
            self.heads = 12
            self.mlp_dim = 768
        else:
            raise ValueError(f"Invalid model_size: {self.model_size}")
    
    def _set_num_classes(self):
        """Set number of classes based on dataset."""
        self.num_classes = 10 if self.dataset == 'cifar10' else 100
    
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Recalculate dependent fields
        self._set_model_params()
        self._set_num_classes()
    
    def to_dict(self):
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create from dictionary."""
        instance = cls()
        instance.update(**config_dict)
        return instance