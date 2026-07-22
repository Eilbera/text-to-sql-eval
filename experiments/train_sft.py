"""LoRA SFT of Qwen2.5-1.5B-Instruct on b-mc2/sql-create-context.

Real run (2 epochs over 2k examples), unlike the 20-step smoke test in the
original train_sft.py. Prompt format is identical to the eval prompt so the
model is trained on exactly the distribution it is tested on.
"""
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

from common import BASE, PROMPT

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUT = BASE + "/models/qwen15b-sql-lora"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda")

lora = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

ds = load_dataset("b-mc2/sql-create-context", split="train").select(
    range(2000))


def fmt(ex):
    msgs = [
        {"role": "user",
         "content": PROMPT.format(schema=ex["context"],
                                  question=ex["question"])},
        {"role": "assistant", "content": ex["answer"]},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False)
    return tokenizer(text, truncation=True, max_length=1024)


ds = ds.map(fmt, remove_columns=ds.column_names)

trainer = Trainer(
    model=model,
    train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    args=TrainingArguments(
        output_dir=BASE + "/models/checkpoints",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        warmup_steps=10,
        learning_rate=2e-4,
        logging_steps=10,
        bf16=True,
        save_strategy="no",
        report_to="none",
    ),
)

trainer.train()
model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)
print("Saved adapter to", OUT, flush=True)
