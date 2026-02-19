#!/usr/bin/env bash
# Training script for the decoupled encoder-decoder model
#
# Usage:
#   ./scripts/train.sh                                # Train with default config
#   ./scripts/train.sh options/scd_minecraft.yml     # Train with specific config
#
# Prerequisites:
#   1. Download Minecraft dataset to datasets/minecraft/ and datasets/minecraft_latent/
#   2. Download pretrained DCAE to pretrained/dcae/DCAE_Minecraft_Res128.pth

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate scd

CONFIG="${1:-options/scd_minecraft.yml}"
NUM_GPUS="${NUM_GPUS:-8}"
PORT="${PORT:-29500}"

echo "Training with config: $CONFIG"
echo "Using $NUM_GPUS GPUs"

accelerate launch \
    --num_processes "$NUM_GPUS" \
    --num_machines 1 \
    --main_process_port "$PORT" \
    train.py \
    -opt "$CONFIG"
