#!/bin/bash
# Round-2 probes: desc lever, kitchen sink (no think), capped-think sample.
# Replaces the unbounded-think probes p4/p5, which were ~10 min/question.
cd /home/baby/claude_experiments
source venv/bin/activate
log() { echo "[$(date '+%H:%M:%S')] $1" >> results/probes.log; }

run() {
    tag=$1; shift
    log "probe $tag start"
    python code/eval_ollama_v2.py --tag "$tag" "$@" \
        > "results/probe_$tag.log" 2>&1 \
        && log "probe $tag ok" || log "probe $tag FAILED"
}

run p4_ev_desc_retry --stride 10 --evidence --desc --retry 2
run p5_ev_rows_desc_retry --stride 10 --evidence --rows --desc --retry 2 --ctx 24576
run p6_think_capped --stride 25 --evidence --retry 1 --think --num-predict 3000

log "probes done"
touch results/PROBES_DONE
