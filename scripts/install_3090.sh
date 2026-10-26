#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo 'Activate Conda environment first: conda activate phasemamba-infer' >&2
    exit 2
fi
if [[ "$(uname -s)" != Linux ]]; then
    echo 'Expected native Linux/WSL2 with NVIDIA CUDA access' >&2
    exit 2
fi
python -m pip install --upgrade pip
python -m pip install 'torch==2.5.1' 'torchvision==0.20.1' --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-infer.txt
python scripts/install_mamba_wheels.py
python -m pip check
python -m pm_infer.run --check
