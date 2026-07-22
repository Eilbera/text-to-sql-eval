"""Shared helpers for the sandbox experiments.

Prompt and grading are kept character-identical to eval_baseline.py so
scores stay comparable across models.
"""
import json
import sqlite3
import time

BASE = "/home/baby/claude_experiments"
DB_DIR = BASE + "/data/dev_databases/"
QUESTIONS = BASE + "/data/mini_dev_sqlite.json"

# Exact prompt from eval_baseline.py (indentation included)
PROMPT = """given this SQLite schema:
    {schema}
    Question: {question}

    Write ONE SQLite query that answer it. Output ONLY the SQL, nothing else."""


def load_questions():
    with open(QUESTIONS) as f:
        return json.load(f)


def db_path_for(ex):
    return DB_DIR + ex["db_id"] + "/" + ex["db_id"] + ".sqlite"


def run_query(db_path, sql, timeout=30):
    connection = sqlite3.connect(db_path)
    deadline = time.time() + timeout
    connection.set_progress_handler(
        lambda: 1 if time.time() > deadline else 0, 100000)
    try:
        rows = connection.execute(sql).fetchall()
    finally:
        connection.close()
    return rows


def get_schema(db_path):
    rows = run_query(
        db_path,
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql NOT NULL")
    return "\n".join(row[0] for row in rows)


def clean_sql(text):
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.lower().startswith("sql"):
            text = text[3:]
    return text.strip()


def grade(ds, ask_fn, out_path, tag):
    """Run the eval loop; ask_fn(schema, question) -> raw model text."""
    per_question = []
    correct = 0
    for i, ex in enumerate(ds):
        db_path = db_path_for(ex)
        schema = get_schema(db_path)
        try:
            model_sql = clean_sql(ask_fn(schema, ex["question"]))
        except Exception as e:
            model_sql = f"<generation error: {e}>"

        try:
            model_answer = run_query(db_path, model_sql)
        except Exception:
            model_answer = None

        try:
            gold_answer = run_query(db_path, ex["SQL"])
        except Exception:
            gold_answer = "<gold failed>"

        ok = model_answer == gold_answer
        correct += ok
        per_question.append({
            "question_id": ex["question_id"],
            "db_id": ex["db_id"],
            "question": ex["question"],
            "model_sql": model_sql,
            "sql_ran": model_answer is not None,
            "correct": ok,
        })
        print(f"[{tag}] {i + 1}/{len(ds)} correct so far: {correct}",
              flush=True)

    summary = {
        "tag": tag,
        "total": len(ds),
        "correct": correct,
        "accuracy": correct / len(ds),
        "sql_ran_rate": sum(q["sql_ran"] for q in per_question) / len(ds),
    }
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_question": per_question}, f,
                  indent=1)
    print(f"[{tag}] FINAL accuracy: {summary['accuracy']:.4f}", flush=True)
    return summary
