# grpo-qwen

GRPO (Group Relative Policy Optimization) fine-tuning of **Qwen/Qwen3.6-27B**
with an **accuracy reward** on a ~2k-example slice of **GSM8K**.

## What's here

| File | Purpose |
|------|---------|
| `train_grpo.py` | Training entrypoint using TRL's `GRPOTrainer` (QLoRA by default). |
| `rewards.py` | `accuracy_reward` — 1.0 if the final answer matches the gold answer, else 0.0. |
| `data.py` | Loads GSM8K, slices to ~2k, formats prompts + `answer` column. |
| `requirements.txt` | Pinned dependencies. |

## How GRPO works (quick version)

For each prompt, the trainer samples a **group** of `G` completions
(`--num-generations`), scores each with the reward function, and updates the
policy toward the completions that beat the group average. There is no separate
value/critic model — the group itself provides the baseline.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional (fast generation on a big model):
pip install vllm
```

> Qwen3.6-27B needs a recent `transformers` (released Apr 2026). If you hit an
> "unknown model type" error, upgrade: `pip install -U transformers`.

## Train

```bash
# Default: QLoRA (4-bit), 2k GSM8K examples, group size 8
python train_grpo.py

# Faster generation with vLLM
python train_grpo.py --use-vllm

# Full bf16 (no 4-bit) — needs ~55GB+ VRAM
python train_grpo.py --no-4bit
```

## Hardware notes

- **QLoRA (default):** base model loads in 4-bit (~18GB) + LoRA adapters +
  activations. Plan for a single 40–80GB GPU (A100/H100) for comfortable
  group sizes; reduce `--num-generations`, `--per-device-batch-size`, and
  `--max-completion-length` to fit smaller cards.
- `--per-device-batch-size` **must be divisible by** `--num-generations`.
- `bitsandbytes` 4-bit requires Linux + NVIDIA. On Mac/CPU, smoke-test the
  pipeline with a tiny model and `--no-4bit`, e.g.
  `python train_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --no-4bit --num-samples 64`.

## Customize the reward

Edit `rewards.py`. You can also pass multiple reward functions to
`GRPOTrainer(reward_funcs=[...])` (e.g. add a format reward that checks for
`\boxed{}`) and the trainer will sum them.
