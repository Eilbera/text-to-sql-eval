from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length = 1024,
    load_in_4bit= False,
    dtype = None,)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing = "unsloth",
)

ds = load_dataset("b-mc2/sql-create-context", split="train")
ds = ds.select(range(2000))
print(ds[0])

PROMPT = """given this SQLite schema:
{schema}
Question: {question}

Write ONE SQLite query that answer it. Output ONLY the SQL, nothing else."""

def fmt(ex):
    msg = [{"role": "user", "content": PROMPT.format(schema=ex["context"], question=ex["question"])},
    {"role": "assistant", "content": ex["answer"]},
    ]
    return {"text": tokenizer.apply_chat_template(msg, tokenize=False)}
ds = ds.map(fmt, remove_columns = ds.column_names)
print(ds[0]["text"])

trainer = SFTTrainer(
    model = model,
    train_dataset = ds,
    args = SFTConfig(
        output_dir="qwen-sql-lora",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        max_steps=20,  
        learning_rate=2e-4,
        logging_steps=5,
        bf16=True,
        dataset_text_field="text",
        max_length=1024,
        report_to="none",
    ),
)

trainer.train()
model.save_pretrained("qwen-sql-lora")
tokenizer.save_pretrained("qwen-sql-lora")
print("Done - saved model to qwen-sql-lora")