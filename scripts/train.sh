#!/bin/bash
# TRM Training Script
# Usage: ./scripts/train.sh [options]
#
# Examples:
#   ./scripts/train.sh                                    # Use defaults (auto-train tokenizer)
#   ./scripts/train.sh --epochs 50 --batch-size 16       # Override settings
#   CUDA_VISIBLE_DEVICES=0 ./scripts/train.sh            # Use specific GPU
#   TOKENIZER=tokenizer/my.model ./scripts/train.sh      # Use existing tokenizer
#   VOCAB_SIZE=16000 ./scripts/train.sh                  # Custom vocab size
#
# Tokenizer behavior:
#   1. If TOKENIZER is set, uses that tokenizer
#   2. If outputs/tokenizer/tokenizer.model exists, reuses it
#   3. Otherwise, auto-trains a new tokenizer from train data

set -e

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
echo "TRM Training Configuration"
echo "============================================"
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

# Build command
CMD="uv run python tools/train.py"
CMD="$CMD --train-data $TRAIN_DATA"
CMD="$CMD --config $CONFIG"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --checkpoint-dir $CHECKPOINT_DIR"

# Add optional arguments
if [ -n "$VAL_DATA" ]; then
    CMD="$CMD --val-data $VAL_DATA"
elif [ -n "$VAL_SPLIT" ]; then
    CMD="$CMD --val-split $VAL_SPLIT"
fi

if [ -n "$TOKENIZER" ]; then
    CMD="$CMD --tokenizer-path $TOKENIZER"
fi

if [ -n "$VOCAB_SIZE" ]; then
    CMD="$CMD --vocab-size $VOCAB_SIZE"
fi

# Pass through any additional arguments
CMD="$CMD $@"

echo "Running: $CMD"
echo ""

# Run training
exec $CMD
