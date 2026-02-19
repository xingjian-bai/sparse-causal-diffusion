#!/usr/bin/env bash
# Inference script for the decoupled encoder-decoder model
#
# Usage:
#   ./scripts/inference.sh <checkpoint_path>
#   ./scripts/inference.sh path/to/ema.pth --context-lengths 36 72 144
#
# Prerequisites:
#   1. Download Minecraft validation data to datasets/minecraft/
#   2. Download pretrained DCAE to pretrained/dcae/DCAE_Minecraft_Res128.pth

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate scd

if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint_path> [additional_args...]"
    echo "Example: $0 experiments/scd_minecraft/models/checkpoint-100000/ema.pth"
    exit 1
fi

CHECKPOINT="$1"
shift  # Remove first argument, pass rest to script

CONFIG="${CONFIG:-options/scd_minecraft.yml}"
NUM_GPUS="${NUM_GPUS:-8}"
PORT="${PORT:-29500}"
DATA_ROOT="${DATA_ROOT:-.}"

echo "Running inference with checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"
echo "Using $NUM_GPUS GPUs"

accelerate launch \
    --num_processes "$NUM_GPUS" \
    --num_machines 1 \
    --main_process_port "$PORT" \
    inference/run_decoupled_inference.py \
    --opt "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    "$@"
