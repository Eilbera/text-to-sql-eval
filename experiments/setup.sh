#!/bin/bash
# One-time environment setup in the sandbox. Touches nothing outside
# /home/baby/claude_experiments.
set -e
cd /home/baby/claude_experiments

mkdir -p data results models code

# Copy (read-only from originals) the BIRD databases into the sandbox
if [ ! -d data/dev_databases ]; then
    cp -r /home/baby/sqltune/dev_databases data/dev_databases
fi

python3 -m venv venv
source venv/bin/activate
pip install -q -U pip

# CUDA torch for aarch64; fall back to default index if unavailable
pip install -q torch --index-url https://download.pytorch.org/whl/cu130 \
    || pip install -q torch

pip install -q transformers peft datasets accelerate requests

python - <<'EOF'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available - aborting setup"
EOF

echo "SETUP_OK"
