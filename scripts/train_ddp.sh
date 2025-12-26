#!/bin/bash
# TRM Distributed Training Script (Multi-GPU)
# Usage: ./scripts/train_ddp.sh [options]
#
# Examples:
#   ./scripts/train_ddp.sh                               # Use all available GPUs
#   CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_ddp.sh     # Use specific GPUs
#   NUM_GPUS=2 ./scripts/train_ddp.sh                   # Specify number of GPUs
#   NUM_GPUS=2 ./scripts/train_ddp.sh --epochs 50       # With extra args
#
# Environment variables:
#   NUM_GPUS      - Number of GPUs (default: auto-detect)
#   TRAIN_DATA    - Path to training data
#   VAL_DATA      - Path to validation data (optional)
#   VAL_SPLIT     - Validation split ratio (default: 0.1)
#   CONFIG        - Config file path (default: configs/default.yaml)
#   TOKENIZER     - Tokenizer path (optional, auto-trains if not set)
#   VOCAB_SIZE    - Vocabulary size for tokenizer training
#   OUTPUT_DIR    - Output directory (default: outputs)
#   MASTER_PORT   - Port for distributed training (default: 29500)

set -e

# Number of GPUs (auto-detect if not set)
if [ -z "$NUM_GPUS" ]; then
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi -L | wc -l)
    else
        NUM_GPUS=1
    fi
fi

# Master port for distributed training
MASTER_PORT="${MASTER_PORT:-29500}"

# Default paths
TRAIN_DATA="${TRAIN_DATA:-data/train.jsonl}"
VAL_DATA="${VAL_DATA:-}"
VAL_SPLIT="${VAL_SPLIT:-0.1}"
CONFIG="${CONFIG:-configs/default.yaml}"
TOKENIZER="${TOKENIZER:-}"
VOCAB_SIZE="${VOCAB_SIZE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"

# Show configuration
echo "============================================"
echo "TRM Distributed Training Configuration"
echo "============================================"
echo "Number of GPUs: $NUM_GPUS"
echo "Master port:    $MASTER_PORT"
echo "Train data:     $TRAIN_DATA"
echo "Val data:       ${VAL_DATA:-'(using val-split)'}"
echo "Val split:      $VAL_SPLIT"
echo "Config:         $CONFIG"
echo "Tokenizer:      ${TOKENIZER:-'(auto-train)'}"
echo "Vocab size:     ${VOCAB_SIZE:-'(from config)'}"
echo "Output dir:     $OUTPUT_DIR"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "============================================"
echo ""

# Build training arguments
TRAIN_ARGS="--train-data $TRAIN_DATA"
TRAIN_ARGS="$TRAIN_ARGS --config $CONFIG"
TRAIN_ARGS="$TRAIN_ARGS --output-dir $OUTPUT_DIR"
TRAIN_ARGS="$TRAIN_ARGS --checkpoint-dir $CHECKPOINT_DIR"

if [ -n "$VAL_DATA" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --val-data $VAL_DATA"
elif [ -n "$VAL_SPLIT" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --val-split $VAL_SPLIT"
fi

if [ -n "$TOKENIZER" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --tokenizer-path $TOKENIZER"
fi

if [ -n "$VOCAB_SIZE" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --vocab-size $VOCAB_SIZE"
fi

# Pass through any additional arguments
TRAIN_ARGS="$TRAIN_ARGS $@"

# Run with torchrun
CMD="uv run torchrun --standalone --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT tools/train.py $TRAIN_ARGS"

echo "Running: $CMD"
echo ""

exec $CMD
