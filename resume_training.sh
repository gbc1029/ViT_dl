#!/bin/bash
# Resume training from the latest checkpoint
# Usage: bash resume_training.sh [cifar10|cifar100] [tiny|small|base]

DATASET=${1:-cifar10}
MODEL_SIZE=${2:-small}

echo "========================================"
echo "Resuming ViT Training"
echo "Dataset: $DATASET"
echo "Model: ViT-$MODEL_SIZE"
echo "========================================"

# Detect number of available GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l 2>/dev/null || echo 1)

# Model-specific batch size
if [ "$MODEL_SIZE" = "tiny" ]; then
    BATCH_SIZE=$((128 * NUM_GPUS))
    EPOCHS=200
    WARMUP=10
elif [ "$MODEL_SIZE" = "small" ]; then
    BATCH_SIZE=$((128 * NUM_GPUS))
    EPOCHS=300
    WARMUP=20
else
    BATCH_SIZE=$((64 * NUM_GPUS))
    EPOCHS=400
    WARMUP=30
fi

echo "Configuration:"
echo "  Batch size: $BATCH_SIZE"
echo "  Target epochs: $EPOCHS"
echo "  GPUs: $NUM_GPUS"
echo "========================================"

# Resume training - will automatically load the latest checkpoint
python train.py \
  --mode resume \
  --dataset $DATASET \
  --model-size $MODEL_SIZE \
  --batch-size $BATCH_SIZE \
  --epochs $EPOCHS \
  --amp \
  --grad-clip 1.0 \
  --label-smoothing 0.1 \
  --drop-path 0.1 \
  --randaug --randaug-n 2 --randaug-m 10 \
  --cutout --cutout-length 16 \
  --mixup --mixup-alpha 0.2 --mixup-prob 0.5 \
  --multi-gpu \
  --verbose
