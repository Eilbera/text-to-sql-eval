import sqlite3
import json
import requests

DB_DIR = "/Users/eilberamansour/Desktop/qwen-sql-eval/minidev/MINIDEV/dev_databases/"
QUESTIONS = "/Users/eilberamansour/Desktop/qwen-sql-eval/minidev/MINIDEV/mini_dev_sqlite.json"

with open(QUESTIONS, "r") as f:
    ds = json.load(f)

def run_query(db_path, sql):
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(sql).fetchall()
    finally:
        connection.close()
    return rows

def get_schema(db_path):
    rows = run_query(db_path, "SELECT sql FROM sqlite_master WHERE type='table' AND sql NOT NULL")
    return "\n".join([row[0] for row in rows])

def ask_qwen(schema, question):
    prompt = f"""given this SQLite schema:
    {schema}
    Question: {question}

    Write ONE SQLite query that answer it. Output ONLY the SQL, nothing else."""
    resp = requests.post("http://localhost:11434/api/chat", json={
        "model": "qwen3.5:27b",
        "messages": [{"role": "user", "content": prompt}],
        "think": False,
        "stream": False,
        "options": {"num_ctx": 8192, "temperature": 0},
    })
    return resp.json()["message"]["content"]

def clean_sql(text):
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.lower().startswith("sql"):
            text = text[3:]
    return text.strip()

correct = 0
total = 0

for i, ex in enumerate(ds):
    db_path = DB_DIR + ex["db_id"] +  "/" + ex["db_id"] + ".sqlite"
    schema = get_schema(db_path)
    model_sql = clean_sql(ask_qwen(schema, ex["question"]))

    try:
        model_answer = run_query(db_path, model_sql)
    except Exception:
        model_answer = None

    gold_answer = run_query(db_path, ex["SQL"])
    correct += model_answer == gold_answer
    total += 1
    print(total, "correct so far:", correct)

print(f"Accuracy: {correct / total}")
with open("baseline_score.txt", "w") as f:
    f.write(f"{correct / total}\n")