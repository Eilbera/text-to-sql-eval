#!/bin/bash
# Ablation probes for the 27B accuracy push. Waits for the 1.5B eval to
# free the GPU, then runs each config on the same 50-question subset.
cd /home/baby/claude_experiments
source venv/bin/activate
log() { echo "[$(date '+%H:%M:%S')] $1" >> results/probes.log; }

log "waiting for 15b_base eval to finish"
while pgrep -f "eval_hf.py --tag 15b_base" > /dev/null; do sleep 30; done
log "GPU free, probes start"

run() {
    tag=$1; shift
    log "probe $tag start"
    python code/eval_ollama_v2.py --tag "$tag" --stride 10 "$@" \
        > "results/probe_$tag.log" 2>&1 \
        && log "probe $tag ok" || log "probe $tag FAILED"
}

run p0_base
run p1_ev --evidence
run p2_ev_retry --evidence --retry 2
run p3_ev_rows_retry --evidence --rows --retry 2
run p4_ev_think --evidence --think
run p5_ev_think_retry --evidence --think --retry 2

log "probes done"
touch results/PROBES_DONE
