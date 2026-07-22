#!/bin/bash
# v3 full run (guidelines config), then a self-consistency probe on top.
cd /home/baby/claude_experiments
source venv/bin/activate
log() { echo "[$(date '+%H:%M:%S')] $1" >> results/probes.log; }

log "v3_full start"
python code/eval_ollama_v2.py --tag v3_full \
    --evidence --desc --retry 2 --guidelines \
    > results/eval_v3_full.log 2>&1 \
    && log "v3_full ok" || log "v3_full FAILED"

log "probe p8_sc3 start"
python code/eval_ollama_v2.py --tag p8_sc3 --stride 10 \
    --evidence --desc --retry 1 --guidelines --sc 3 \
    > results/probe_p8_sc3.log 2>&1 \
    && log "probe p8_sc3 ok" || log "probe p8_sc3 FAILED"
log "round 4 done"
