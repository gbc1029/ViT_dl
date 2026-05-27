#!/usr/bin/env python
"""Test all new advanced data augmentation methods"""

from data_loader import get_dataloaders

def test_augmentation(test_name, **kwargs):
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"{'='*60}")
    
    try:
        train_loader, test_loader = get_dataloaders(
            dataset='cifar10',
            batch_size=16,
            num_workers=0,
            data_dir='./data',
            **kwargs
        )
        
        # Get one batch
        images, labels = next(iter(train_loader))
        print(f"✅ PASSED")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")

if __name__ == "__main__":
    print("Testing All Data Augmentation Methods")
    
    # Test 1: Baseline
    test_augmentation(
        "Baseline (RandomCrop + RandomFlip)",
        randaug_enabled=False,
        use_cutout=False,
        use_color_jitter=False
    )
    
    # Test 2: RandAugment
    test_augmentation(
        "RandAugment (n=2, m=9)",
        randaug_enabled=True,
        randaug_n=2,
        randaug_m=9
    )
    
    # Test 3: ColorJitter
    test_augmentation(
        "ColorJitter",
        use_color_jitter=True,
        color_jitter_brightness=0.2,
        color_jitter_contrast=0.2,
        color_jitter_saturation=0.2,
        color_jitter_hue=0.1
    )
    
    # Test 4: RandomRotation
    test_augmentation(
        "RandomRotation (15 degrees)",
        use_random_rotation=True,
        rotation_degrees=15
    )
    
    # Test 5: RandomAffine
    test_augmentation(
        "RandomAffine (translate=0.1)",
        use_random_affine=True,
        affine_translate=0.1
    )
    
    # Test 6: Cutout
    test_augmentation(
        "Cutout (length=16)",
        use_cutout=True,
        cutout_length=16
    )
    
    # Test 7: Combined (RandAugment + Cutout + ColorJitter)
    test_augmentation(
        "Combined (RandAugment + Cutout)",
        randaug_enabled=True,
        randaug_n=2,
        randaug_m=9,
        use_cutout=True,
        cutout_length=16
    )
    
    # Test 8: Maximum Augmentation
    test_augmentation(
        "Maximum Augmentation (All methods)",
        randaug_enabled=True,
        randaug_n=3,
        randaug_m=12,
        use_cutout=True,
        cutout_length=16,
        use_color_jitter=True,
        color_jitter_brightness=0.3,
        color_jitter_contrast=0.3,
        use_random_rotation=True,
        rotation_degrees=20,
        use_random_affine=True,
        affine_translate=0.15
    )
    
    print(f"\n{'='*60}")
    print("All augmentation tests completed!")
    print("Now you can use these augmentations in training:")
    print("  python train.py --randaug --cutout --color-jitter --rotation")
    print(f"{'='*60}")