import sqlite3
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=1024,
    load_in_4bit=False,
)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16)

DB_DIR = "/home/baby/sqltune/dev_databases/"

def run_query(db_path, sql):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()

def get_schema(db_path):
    rows = run_query(db_path, "SELECT sql FROM sqlite_master WHERE type='table' AND sql NOT NULL")
    return "\n".join(r[0] for r in rows)

# Build training pairs in the SAME prompt format the eval uses.
# NOTE: training on mini_dev itself is data leakage — fine for this
# pipeline test only, never for real experiments.
ds = load_dataset("birdsql/bird_mini_dev")["mini_dev_sqlite"]
texts = []
for ex in list(ds)[:100]:
    schema = get_schema(DB_DIR + ex["db_id"] + "/" + ex["db_id"] + ".sqlite")
    prompt = f"""given this SQLite schema:
    {schema}
    Question: {ex["question"]}

    Write ONE SQLite query that answer it. Output ONLY the SQL, nothing else."""
    msgs = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ex["SQL"]},
    ]
    texts.append({"text": tokenizer.apply_chat_template(msgs, tokenize=False)})

print(texts[0]["text"])
input("press enter to continue...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=Dataset.from_list(texts),
    args=SFTConfig(
        max_steps=30,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        output_dir="test_run",
        dataset_text_field="text",
    ),
)
trainer.train()

model.save_pretrained_gguf("qwen-sql-test", tokenizer, quantization_method="q8_0")
print("done - GGUF saved in qwen-sql-test/")
