#!/bin/bash
# ============================================================
# ViT Training Script: Small model on CIFAR-100
# ============================================================

echo "========================================"
echo "ViT Training: Small model + CIFAR-100"
echo "========================================"

# Configuration
MODEL_SIZE="small"
DATASET="cifar100"
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
  --epochs 400 \
  --lr 5e-4 \
  --weight-decay 0.06 \
  --warmup-epochs 15 \
  --drop-path 0.2 \
  --grad-clip 1.0 \
  --amp \
  --randaug --randaug-n 2 --randaug-m 12 \
  --cutout --cutout-length 16 \
  --mixup --mixup-alpha 0.3 --mixup-prob 0.5 \
  --label-smoothing 0.15 \
  --multi-gpu \
  2>&1 | tee -a logs/train_${MODEL_SIZE}_${DATASET}.log