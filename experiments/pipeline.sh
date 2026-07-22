#!/bin/bash
# Overnight experiment pipeline. Order puts the core before/after experiment
# first so a partial night still yields the comparison.
cd /home/baby/claude_experiments
source venv/bin/activate
mkdir -p results models

log() { echo "[$(date '+%H:%M:%S')] $1" >> results/pipeline.log; }

log "pipeline start"

log "STEP 1: eval untrained 1.5B"
python code/eval_hf.py --tag 15b_base > results/eval_15b_base.log 2>&1 \
    && log "STEP 1 ok" || log "STEP 1 FAILED"

log "STEP 2: LoRA SFT"
if python code/train_sft.py > results/train.log 2>&1; then
    log "STEP 2 ok"
    log "STEP 3: eval fine-tuned 1.5B"
    python code/eval_hf.py --tag 15b_tuned \
        --adapter /home/baby/claude_experiments/models/qwen15b-sql-lora \
        > results/eval_15b_tuned.log 2>&1 \
        && log "STEP 3 ok" || log "STEP 3 FAILED"
else
    log "STEP 2 FAILED - skipping tuned eval"
fi

log "STEP 4: reproduce 27B ollama baseline"
python code/eval_ollama.py > results/eval_27b.log 2>&1 \
    && log "STEP 4 ok" || log "STEP 4 FAILED"

log "pipeline end"
touch results/DONE
