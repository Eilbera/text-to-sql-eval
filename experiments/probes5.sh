#!/bin/bash
# Round-5: execution-informed self-refine probe; auto-launch full run if it
# clearly beats the 62% guidelines probe on the same subset.
cd /home/baby/claude_experiments
source venv/bin/activate
log() { echo "[$(date '+%H:%M:%S')] $1" >> results/probes.log; }

log "probe p9_refine start"
python code/eval_ollama_v2.py --tag p9_refine --stride 10 \
    --evidence --desc --retry 2 --guidelines --refine \
    > results/probe_p9_refine.log 2>&1 \
    && log "probe p9_refine ok" || { log "probe p9_refine FAILED"; exit 1; }

acc=$(venv/bin/python -c "import json; print(json.load(open('results/score_p9_refine.json'))['summary']['accuracy_set'])")
log "p9_refine accuracy_set=$acc"

if venv/bin/python -c "import sys; sys.exit(0 if float('$acc') >= 0.64 else 1)"; then
    log "refine wins, v4_full start"
    python code/eval_ollama_v2.py --tag v4_full \
        --evidence --desc --retry 2 --guidelines --refine \
        > results/eval_v4_full.log 2>&1 \
        && log "v4_full ok" || log "v4_full FAILED"
else
    log "refine does not clear 0.64 bar; v3_full stands as final"
fi
