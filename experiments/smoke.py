"""Fast end-to-end sanity check before the real pipeline: one generation,
one training step. Catches API/version bugs in minutes instead of hours."""
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

from common import BASE, PROMPT, db_path_for, get_schema, load_questions

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda")

# 1. one real generation on the first BIRD question
ex = load_questions()[0]
schema = get_schema(db_path_for(ex))
msgs = [{"role": "user",
         "content": PROMPT.format(schema=schema, question=ex["question"])}]
enc = tokenizer.apply_chat_template(
    msgs, add_generation_prompt=True, return_dict=True,
    return_tensors="pt").to("cuda")
out = model.generate(**enc, max_new_tokens=64, do_sample=False,
                     pad_token_id=tokenizer.eos_token_id)
prompt_len = enc["input_ids"].shape[1]
print("GEN SAMPLE:", tokenizer.decode(out[0][prompt_len:],
                                      skip_special_tokens=True)[:200])

# 2. one LoRA training step on 8 examples
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=16, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]))

ds = load_dataset("b-mc2/sql-create-context", split="train").select(range(8))


def fmt(e):
    m = [{"role": "user", "content": PROMPT.format(schema=e["context"],
                                                   question=e["question"])},
         {"role": "assistant", "content": e["answer"]}]
    return tokenizer(tokenizer.apply_chat_template(m, tokenize=False),
                     truncation=True, max_length=1024)


ds = ds.map(fmt, remove_columns=ds.column_names)
Trainer(
    model=model,
    train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    args=TrainingArguments(
        output_dir=BASE + "/models/smoke", per_device_train_batch_size=4,
        max_steps=1, bf16=True, report_to="none", save_strategy="no"),
).train()

print("SMOKE_OK")
