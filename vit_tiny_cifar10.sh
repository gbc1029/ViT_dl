#!/bin/bash
# ============================================================
# ViT Training Script: Tiny model on CIFAR-10
# ============================================================

echo "========================================"
echo "ViT Training: Tiny model + CIFAR-10"
echo "========================================"

# Configuration
MODEL_SIZE="tiny"
DATASET="cifar10"
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
PER_GPU_BATCH=128
TOTAL_BATCH=$((PER_GPU_BATCH * NUM_GPUS))

echo "Configuration:"
echo "  Model: ViT-$MODEL_SIZE"
echo "  Dataset: $DATASET"
echo "  GPUs: $NUM_GPUS"
echo "  Batch size: $TOTAL_BATCH (per GPU: $PER_GPU_BATCH)"
echo "========================================"

# Train command with anti-overfitting settings
python train.py \
  --mode train \
  --model-size $MODEL_SIZE \
  --dataset $DATASET \
  --batch-size $TOTAL_BATCH \
  --epochs 300 \
  --lr 5e-4 \
  --weight-decay 0.05 \
  --warmup-epochs 10 \
  --drop-path 0.1 \
  --grad-clip 1.0 \
  --amp \
  --randaug --randaug-n 2 --randaug-m 9 \
  --cutout --cutout-length 8 \
  --mixup --mixup-alpha 0.2 --mixup-prob 0.5 \
  --label-smoothing 0.1 \
  --multi-gpu \
  2>&1 | tee -a logs/train_${MODEL_SIZE}_${DATASET}.log