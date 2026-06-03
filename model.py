"""ResNet CNN implementation for CIFAR-10/CIFAR-100 comparison with ViT"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block for ResNet.
    
    Supports stride-based downsampling and 1x1 convolution shortcut.
    """
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    """Generic ResNet for CIFAR with flexible channel configuration.
    
    Args:
        block: Block class (BasicBlock)
        channels: List of channel counts per stage [c1, c2, c3, c4]
        block_counts: List of block counts per stage [n1, n2, n3, n4]
        num_classes: Number of output classes
        zero_init_residual: If True, zero-initialize the last BN in each block
    """
    
    def __init__(self, block, channels, block_counts, num_classes=10,
                 zero_init_residual=True):
        super().__init__()
        self.in_planes = channels[0]
        
        # Stem: Conv3x3 + BN + ReLU (no pooling, preserves 32x32 for CIFAR)
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        
        # Residual stages
        self.layer1 = self._make_layer(block, channels[0], block_counts[0], stride=1)
        self.layer2 = self._make_layer(block, channels[1], block_counts[1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], block_counts[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], block_counts[3], stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(channels[3] * block.expansion, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        
        return out


# ========== ResNet-Tiny (1.64M, for CIFAR-10) ==========
#
#  | Stage      | Operation                         | Output     | Params  |
#  |------------|-----------------------------------|------------|---------|
#  | Stem       | Conv3×3(3→32) + BN                | 32×32×32   | 928     |
#  | Stage1     | 3 × BasicBlock(32→32)             | 32×32×32   | 55,680  |
#  | Stage2     | Downsample(32→64) + 1×BasicBlock  | 16×16×64   | 131,712 |
#  | Stage3     | Downsample(64→128) + 1×BasicBlock | 8×8×128    | 525,568 |
#  | Stage4     | Downsample(128→256)               | 4×4×256    | 919,040 |
#  | Classifier | GAP + Linear(256, num_classes)    | num_classes| ~2,570  |
#  | **Total**  |                                   |            | ~1.64M  |

def resnet_tiny(num_classes=10):
    """ResNet-Tiny: ~1.64M parameters, designed for CIFAR-10."""
    return ResNetCIFAR(
        block=BasicBlock,
        channels=[32, 64, 128, 256],
        block_counts=[3, 2, 2, 1],  # 3 + (1+1) + (1+1) + (1) = 3+2+2+1=8 blocks
        num_classes=num_classes
    )


# ========== ResNet-Small (for CIFAR-100) ==========
#
#  | Stage  | Operation                           | Output   | Channels | Blocks |
#  |--------|-------------------------------------|----------|----------|--------|
#  | Stem   | Conv3×3(3→46) + BN + ReLU           | 32×32    | 46       | 1      |
#  | Stage1 | 3 × BasicBlock(46→46)               | 32×32    | 46       | 3      |
#  | Stage2 | Downsample(46→92,s2)+2×BasicBlock   | 16×16    | 92       | 1+2    |
#  | Stage3 | Downsample(92→184,s2)+3×BasicBlock  | 8×8      | 184      | 1+3    |
#  | Stage4 | Downsample(184→368,s2)+2×BasicBlock | 4×4      | 368      | 1+2    |
#  | Head   | GAP + Linear(368, num_classes)       | -        | -        | 1      |

def resnet_small(num_classes=100):
    """ResNet-Small for CIFAR-100 comparison."""
    return ResNetCIFAR(
        block=BasicBlock,
        channels=[46, 92, 184, 368],
        block_counts=[3, 3, 4, 3],  # 3 + (1+2) + (1+3) + (1+2) = 3+3+4+3=13 blocks
        num_classes=num_classes
    )


# ========== Legacy: Standard ResNet18/34 (keep for backward compat) ==========

class ResNet(nn.Module):
    """Standard ResNet for CIFAR-10 (64→128→256→512 channels)."""
    
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


if __name__ == "__main__":
    # Test models
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    x = torch.randn(2, 3, 32, 32)
    
    for name, model_fn, cls in [
        ("ResNet-Tiny (CIFAR-10)", resnet_tiny, 10),
        ("ResNet-Tiny (CIFAR-100)", resnet_tiny, 100),
        ("ResNet-Small (CIFAR-10)", resnet_small, 10),
        ("ResNet-Small (CIFAR-100)", resnet_small, 100),
        ("ResNet18", ResNet18, 10),
        ("ResNet34", ResNet34, 10),
    ]:
        model = model_fn(num_classes=cls)
        out = model(x)
        params = count_params(model)
        print(f"{name:30s} | Params: {params/1e6:.2f}M ({params:,}) | Output: {out.shape}")
