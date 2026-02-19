#!/usr/bin/env bash
# Create conda environment and install all dependencies.
#
# Usage:
#   bash scripts/setup_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="scd"
PYTHON_VERSION="3.10"

log()  { echo ""; echo "=> $*"; }
skip() { echo "   [skip] $*"; }

# Initialize conda for this shell session
eval "$(conda shell.bash hook)"

# ---------------------------------------------------------------------------
# 1. Conda environment
# ---------------------------------------------------------------------------
log "Creating conda environment: ${ENV_NAME}"

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    skip "Environment '${ENV_NAME}' already exists"
else
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

# ---------------------------------------------------------------------------
# 2. Dependencies
# ---------------------------------------------------------------------------
log "Installing PyTorch + pip dependencies"

if python -c "import torch; assert torch.__version__.startswith('2.4')" 2>/dev/null; then
    skip "PyTorch 2.4 already installed"
else
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu124
fi

pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
log "Environment ready!"
echo ""
echo "   conda activate ${ENV_NAME}"
echo "   Next: bash scripts/setup_data.sh"
