#!/bin/bash
# Round-6: finish pulling the q8 quant, then full 500-q run with the
# winning config (evidence + desc + retry + guidelines) on it.
cd /home/baby/claude_experiments
source venv/bin/activate
log() { echo "[$(date '+%H:%M:%S')] $1" >> results/probes.log; }

log "pulling qwen3.5:27b-q8_0"
if ! ollama pull qwen3.5:27b-q8_0 >> results/pull_q8.log 2>&1; then
    log "q8 pull FAILED"
    exit 1
fi
log "q8 pull ok, v5_full_q8 start"

python code/eval_ollama_v2.py --tag v5_full_q8 \
    --model qwen3.5:27b-q8_0 \
    --evidence --desc --retry 2 --guidelines \
    > results/eval_v5_full_q8.log 2>&1 \
    && log "v5_full_q8 ok" || log "v5_full_q8 FAILED"
