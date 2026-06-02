"""Reward functions for GRPO.

`accuracy_reward` checks whether the model's final answer matches the
ground-truth answer. It uses `math_verify` for robust mathematical
equivalence (so "1/2", "0.5", and "\\frac{1}{2}" all count as equal),
and falls back to normalized string comparison when parsing fails.
"""

from __future__ import annotations

import re

try:
    from math_verify import parse, verify

    _HAS_MATH_VERIFY = True
except Exception:  # pragma: no cover - fallback if dependency missing
    _HAS_MATH_VERIFY = False


def _extract_answer(text: str) -> str:
    """Pull the final answer out of a completion.

    Priority:
      1. Content inside \\boxed{...}
      2. Text after a '#### ' marker (GSM8K style)
      3. Text after 'answer is'/'answer:'
      4. The last number in the string
      5. The stripped text itself
    """
    if text is None:
        return ""

    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].strip()

    if "####" in text:
        return text.split("####")[-1].strip()

    m = re.search(r"answer\s*(?:is|:)\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")

    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return text.strip()


def _normalize(s: str) -> str:
    return re.sub(r"[\s,$%]", "", s).strip().lower()


def accuracy_reward(completions, answer, **kwargs):
    """Return 1.0 for a correct final answer, else 0.0.

    Signature matches what `GRPOTrainer` passes:
      - `completions`: list of model outputs. With a conversational dataset
        each item is a list of {"role", "content"} messages.
      - `answer`: list of ground-truth answers (one per prompt), forwarded
        from the dataset's "answer" column via **kwargs.
    """
    rewards = []
    for completion, gold in zip(completions, answer):
        # Support both conversational and plain-text completion formats.
        if isinstance(completion, list):
            text = completion[-1]["content"]
        else:
            text = completion

        pred = _extract_answer(text)
        gold_ans = _extract_answer(str(gold))

        correct = False
        if _HAS_MATH_VERIFY:
            try:
                correct = verify(parse(gold_ans), parse(pred))
            except Exception:
                correct = False
        if not correct:
            correct = _normalize(pred) == _normalize(gold_ans) and pred != ""

        rewards.append(1.0 if correct else 0.0)
    return rewards
