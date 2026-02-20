#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate scd

if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint_path> [extra args...]"
    exit 1
fi

CHECKPOINT="$1"
shift

CONFIG="${CONFIG:-options/scd_minecraft.yml}"
NUM_GPUS="${NUM_GPUS:-8}"
PORT="${PORT:-29500}"

accelerate launch \
    --num_processes "$NUM_GPUS" \
    --num_machines 1 \
    --main_process_port "$PORT" \
    inference/run_decoupled_inference.py \
    --opt "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    "$@"
