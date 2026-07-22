#!/bin/bash
# Round-3 probe: column-selection guidelines. Waits for the v2_full run.
cd /home/baby/claude_experiments
source venv/bin/activate
log() { echo "[$(date '+%H:%M:%S')] $1" >> results/probes.log; }

log "probes3 queued, waiting for v2_full"
while pgrep -f "tag v2_full" > /dev/null; do sleep 60; done
log "GPU free, round-3 probe start"

log "probe p7_guidelines start"
python code/eval_ollama_v2.py --tag p7_guidelines --stride 10 \
    --evidence --desc --retry 2 --guidelines \
    > results/probe_p7_guidelines.log 2>&1 \
    && log "probe p7_guidelines ok" || log "probe p7_guidelines FAILED"
