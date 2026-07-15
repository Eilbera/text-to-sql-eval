import sqlite3
import time
from datasets import load_dataset
from unsloth import FastLanguageModel

MODEL = "unsloth/Qwen2.5-1.5B-Instruct"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length = 1024,
    load_in_4bit= False,
)
FastLanguageModel.for_inference(model)

DB_DIR = "/home/baby/sqltune/dev_databases/"
ds = load_dataset("birdsql/bird_mini_dev")

def run_query(db_path, sql, timeout=10):
    connection = sqlite3.connect(db_path)
    deadline = time.time() + timeout
    connection.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 1000)
    try:
        rows = connection.execute(sql).fetchall()
    finally:
        connection.close()
    return rows

def row_match(model_answer, gold_answer):
    if model_answer is None:
        return False
    return set(model_answer) == set(gold_answer)

def get_schema(db_path):
    rows = run_query(db_path, "SELECT sql FROM sqlite_master WHERE type='table' AND sql NOT NULL")
    return "\n".join([row[0] for row in rows])

def ask_qwen(schema, question):
    prompt = f"""given this SQLite schema:
    {schema}
    Question: {question}

    Write ONE SQLite query that answer it. Output ONLY the SQL, nothing else."""
    msgs = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = model.generate(input_ids=ids, max_new_tokens=256, do_sample=False)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

def clean_sql(text):
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.lower().startswith("sql"):
            text = text[3:]
    return text.strip()

correct = 0
total = 0
for i, ex in enumerate(ds["mini_dev_sqlite"]):
  
    db_path = DB_DIR + ex["db_id"] +  "/" + ex["db_id"] + ".sqlite"
    schema = get_schema(db_path)
    model_sql = clean_sql(ask_qwen(schema, ex["question"]))

    try:
        model_answer = run_query(db_path, model_sql)
    except Exception:
        model_answer = None

    gold_answer = run_query(db_path, ex["SQL"])
    correct += row_match(model_answer, gold_answer)
    total += 1
    print(total, "correct so far:", correct)

print(f"Accuracy: {correct / total}")
with open("baseline_score.txt", "w") as f:
    f.write(f"{correct / total}\n")