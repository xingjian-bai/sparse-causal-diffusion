#!/usr/bin/env bash
# Download datasets and pretrained models from HuggingFace.
#
# Prerequisites:
#   - Run setup_env.sh first
#   - export HF_TOKEN=hf_... (see https://huggingface.co/settings/tokens)
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash scripts/setup_data.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

eval "$(conda shell.bash hook)"
conda activate scd

log()  { echo ""; echo "=> $*"; }
skip() { echo "   [skip] $*"; }

download_hf_dataset() {
    local repo_id=$1 local_dir=$2
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='${repo_id}', repo_type='dataset',
                  local_dir='${local_dir}', token='${HF_TOKEN:-}' or None)
"
}

download_hf_file() {
    local repo_id=$1 filename=$2 local_dir=$3
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='${repo_id}', filename='${filename}',
                local_dir='${local_dir}', token='${HF_TOKEN:-}' or None)
"
}

extract_and_cleanup() {
    local dataset_dir=$1
    local num_tars
    num_tars=$(find "$dataset_dir" -maxdepth 1 -name "shard-*.tar" | wc -l)
    if [ "$num_tars" -eq 0 ]; then
        return
    fi
    echo "   Extracting $num_tars tar shards in $dataset_dir ..."
    for tar_file in "$dataset_dir"/shard-*.tar; do
        tar xf "$tar_file" -C "$dataset_dir"
        rm "$tar_file"
    done
    # Clean up HuggingFace download cache
    rm -rf "$dataset_dir/.cache"
    echo "   Extraction complete, tars and cache removed."
}

# ---------------------------------------------------------------------------
# 1. Datasets
# ---------------------------------------------------------------------------
log "Downloading Minecraft datasets"

mkdir -p datasets

if [ -d "datasets/minecraft/training" ]; then
    skip "datasets/minecraft/training/ already exists"
else
    if [ ! -d "datasets/minecraft" ] || [ -z "$(find datasets/minecraft -maxdepth 1 -name 'shard-*.tar' 2>/dev/null)" ]; then
        echo "   Downloading raw videos (for validation/inference)..."
        download_hf_dataset "guyuchao/Minecraft" "datasets/minecraft"
    fi
    extract_and_cleanup "datasets/minecraft"
fi

if [ -d "datasets/minecraft_latent/training" ]; then
    skip "datasets/minecraft_latent/training/ already exists"
else
    if [ ! -d "datasets/minecraft_latent" ] || [ -z "$(find datasets/minecraft_latent -maxdepth 1 -name 'shard-*.tar' 2>/dev/null)" ]; then
        echo "   Downloading pre-extracted latents (for training)..."
        download_hf_dataset "guyuchao/Minecraft_Latent" "datasets/minecraft_latent"
    fi
    extract_and_cleanup "datasets/minecraft_latent"
fi

# ---------------------------------------------------------------------------
# 2. Pretrained DCAE
# ---------------------------------------------------------------------------
log "Downloading pretrained DCAE checkpoint"

DCAE_FILE="pretrained/dcae/DCAE_Minecraft_Res128-a5677f66.pth"
DCAE_LINK="pretrained/dcae/DCAE_Minecraft_Res128.pth"

if [ -f "$DCAE_FILE" ]; then
    skip "$DCAE_FILE already exists"
else
    mkdir -p pretrained
    download_hf_file "guyuchao/FAR_Models" "dcae/DCAE_Minecraft_Res128-a5677f66.pth" "pretrained"
fi

# Symlink to the name expected by the config
if [ -f "$DCAE_FILE" ] && [ ! -e "$DCAE_LINK" ]; then
    ln -s "DCAE_Minecraft_Res128-a5677f66.pth" "$DCAE_LINK"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
log "Data setup complete!"
echo ""
echo "   NUM_GPUS=8 ./scripts/train.sh          # training"
echo "   ./scripts/inference.sh path/to/ema.pth  # inference"
