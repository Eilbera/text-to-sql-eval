# Text-to-SQL on BIRD mini-dev: baseline, prompting ablations, and LoRA SFT

An eval harness measuring how well a local LLM translates English questions into SQL, using the BIRD mini-dev benchmark (~500 questions over 11 real SQLite databases) — followed by a systematic accuracy push that took qwen3.5:27b from **25.4% to ~62%** with no training, plus a LoRA fine-tuning experiment on a small model.

**Metric: execution accuracy.** The model's query and the gold query are both executed against the real database; the model scores only if the results match. Comparing results instead of SQL text matters because many different queries are correct.

## Results

| Config | Execution accuracy |
|---|---|
| qwen3.5:27b (Q4_K_M via Ollama), zero-shot, temp 0 — baseline | **25.4%** (full 500 questions) |
| + evidence hints + column descriptions + retry×2 + guidelines | **~62%** (50-question probe subset) |

The push was run as six rounds of ablation probes, each testing lever combinations on a cheap 50-question subset (every 10th question) before committing to full runs. Levers in `experiments/eval_ollama_v2.py`, all independently toggleable:

- `--evidence` — include BIRD's official per-question hint (the baseline ignored these)
- `--desc` — append BIRD's column-description docs to the schema
- `--rows` — append 3 sample rows per table
- `--retry N` — execution-guided repair: feed SQLite errors back to the model, up to N times
- `--guidelines` — short rules targeting the most common failure (select exactly the requested columns, no formatting/CAST, follow hint formulas)
- `--think` — model reasoning mode
- `--sc K` — self-consistency: K samples at temp 0.7, majority vote on execution results
- `--refine` — show the model its query's first rows and ask it to verify or fix

What was learned:

- **Missing information beat clever inference.** The big wins were the evidence hints and column documentation — context the model didn't have — plus letting it retry on SQL errors.
- **Unbounded reasoning mode was dropped for cost** (~10 min/question); a token-capped variant was probed instead.
- **Self-consistency and self-refine didn't clear the bar.** Each was tested against the reigning config on the same subset and had to beat it to be kept (see `probes5.sh` for the auto-promotion logic).

## Repo layout

- Root: the original baseline harness (`eval_baseline.py` and friends). Runs anywhere with Ollama.
- `experiments/`: the accuracy-push and fine-tuning code, archived as-run. Paths are hardcoded to the remote GPU box it ran on (`/home/baby/claude_experiments`) — adjust `common.py` to rerun.
  - `eval_ollama_v2.py` — the lever-based eval; grades with both ordered-rows equality (comparable to the 25.4% baseline) and set-of-rows equality (official BIRD style)
  - `probes.sh` … `probes6.sh` — the six probe rounds, run as queued overnight jobs
  - `train_sft.py`, `eval_hf.py`, `pipeline.sh` — the fine-tuning experiment
  - `analyze.py`, `report.py` — ablation tables, failure buckets, run diffs

## Fine-tuning experiment

LoRA SFT of Qwen2.5-1.5B-Instruct on 2k examples of `b-mc2/sql-create-context` (r=16, alpha=16, adapters on all attention + MLP projections, 2 epochs, bf16), with the training prompt kept character-identical to the eval prompt. `pipeline.sh` runs the before/after comparison: eval untrained 1.5B → train → eval tuned 1.5B.

The planned follow-up — GRPO with execution-match as the reward — was not run.

## Run the baseline

1. Download BIRD mini-dev from [https://bird-bench.github.io](https://bird-bench.github.io) and update `DB_DIR`/`QUESTIONS` in `eval_baseline.py`
2. Have Ollama running with `qwen3.5:27b` pulled
3. `pip install -r requirements.txt && python3 eval_baseline.py`

## Known limitations

- The ~62% figure is from the 50-question probe subset; the full-500 confirmation runs (and all `score_*.json` result files) lived on the remote GPU box and were not preserved.
- Baseline result comparison is order-sensitive (v2 also reports the official set-based metric).
- The q8_0 quant rerun of the winning config (`probes6.sh`) was launched but its result was likewise not preserved.
