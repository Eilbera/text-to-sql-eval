"""Dataset loading for GRPO.

Loads GSM8K, slices it to ~2k examples, and formats each row as a
conversational prompt with a ground-truth `answer` column that the
reward function consumes.
"""

from __future__ import annotations

import re

from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step, then give "
    "the final answer on its own line in the form \\boxed{ANSWER}."
)


def _gold_answer(raw: str) -> str:
    """GSM8K answers look like '...reasoning...\n#### 42'. Keep the final value."""
    if "####" in raw:
        return raw.split("####")[-1].strip().replace(",", "")
    return raw.strip()


def build_dataset(num_samples: int = 2000, split: str = "train", seed: int = 42):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))

    def _format(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "answer": _gold_answer(example["answer"]),
        }

    return ds.map(_format, remove_columns=ds.column_names)


if __name__ == "__main__":
    d = build_dataset(8)
    print(d)
    print(d[0])
