#!/bin/bash
# Master runner: setup -> smoke test -> full pipeline.
cd /home/baby/claude_experiments
mkdir -p results

bash code/setup.sh > results/setup.log 2>&1 \
    || { echo SETUP > results/FAILED; touch results/DONE; exit 1; }

source venv/bin/activate

python code/smoke.py > results/smoke.log 2>&1 \
    || { echo SMOKE > results/FAILED; touch results/DONE; exit 1; }

bash code/pipeline.sh
