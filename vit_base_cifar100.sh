#!/bin/bash
# ============================================================
# ViT Training Script: Base model on CIFAR-100
# ============================================================

echo "========================================"
echo "ViT Training: Base model + CIFAR-100"
echo "========================================"

# Configuration
MODEL_SIZE="base"
DATASET="cifar100"
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
PER_GPU_BATCH=64  # Base model is larger, reduce per-GPU batch
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
  --epochs 500 \
  --lr 3e-4 \
  --weight-decay 0.08 \
  --warmup-epochs 20 \
  --drop-path 0.3 \
  --grad-clip 1.0 \
  --amp \
  --randaug --randaug-n 3 --randaug-m 15 \
  --cutout --cutout-length 16 \
  --mixup --mixup-alpha 0.4 --mixup-prob 0.6 \
  --label-smoothing 0.2 \
  --multi-gpu \
  2>&1 | tee -a logs/train_${MODEL_SIZE}_${DATASET}.log