"""Eval a HuggingFace model (optionally with a LoRA adapter) on BIRD mini-dev."""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import BASE, PROMPT, grade, load_questions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default=None)
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def ask(schema, question):
        prompt = PROMPT.format(schema=schema, question=question)
        msgs = [{"role": "user", "content": prompt}]
        enc = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_dict=True,
            return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        return tokenizer.decode(out[0][prompt_len:],
                                skip_special_tokens=True)

    ds = load_questions()
    grade(ds, ask, f"{BASE}/results/score_{args.tag}.json", args.tag)


if __name__ == "__main__":
    main()
