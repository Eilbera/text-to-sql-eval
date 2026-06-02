"""GRPO fine-tuning of Qwen3.6-27B with an accuracy reward.

Group Relative Policy Optimization (GRPO) samples a *group* of completions
per prompt, scores each with the reward function(s), and pushes the policy
toward the above-average completions in each group (no value model needed).

Defaults target a single large GPU via QLoRA (4-bit). A full 27B finetune is
impractical on most setups, so we load the base model in 4-bit and train LoRA
adapters on top.

Run:
    python train_grpo.py
    python train_grpo.py --num-samples 2000 --model Qwen/Qwen3.6-27B
"""

from __future__ import annotations

import argparse

import torch
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from data import build_dataset
from rewards import accuracy_reward


def parse_args():
    p = argparse.ArgumentParser(description="GRPO + accuracy_reward on Qwen3.6-27B")
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--output-dir", default="outputs/grpo-qwen3.6-27b")
    p.add_argument("--num-generations", type=int, default=8,
                   help="Group size G: completions sampled per prompt.")
    p.add_argument("--per-device-batch-size", type=int, default=8,
                   help="Must be divisible by --num-generations.")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-prompt-length", type=int, default=512)
    p.add_argument("--max-completion-length", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--no-4bit", action="store_true",
                   help="Disable 4-bit loading (needs ~55GB+ VRAM in bf16).")
    p.add_argument("--use-vllm", action="store_true",
                   help="Use vLLM for generation (much faster; needs `pip install vllm`).")
    return p.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(num_samples=args.num_samples)

    quant_config = None
    model_kwargs = {"torch_dtype": torch.bfloat16, "attn_implementation": "eager"}
    if not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # TRL builds the model from the id + these kwargs (so it can also create
    # the frozen reference model correctly).
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_steps=100,
        report_to="none",
        use_vllm=args.use_vllm,
        model_init_kwargs=model_kwargs,
    )

    trainer = GRPOTrainer(
        model=args.model,
        processing_class=tokenizer,
        reward_funcs=[accuracy_reward],
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Done. Adapters saved to {args.output_dir}")


if __name__ == "__main__":
    main()
