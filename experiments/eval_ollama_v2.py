"""Improved qwen3.5:27b eval with independently toggleable accuracy levers.

Levers (all off = original 25.4% baseline config):
  --evidence   include the BIRD evidence hint in the prompt
  --think      enable qwen3.5 reasoning mode
  --retry N    execution-guided retry: feed SQL errors back, up to N times
  --rows       append 3 sample rows per table to the schema
  --desc       append BIRD column-description docs to the schema
  --sc K       self-consistency: K samples at temp 0.7, majority by result

Grades with BOTH metrics: ordered-rows equality (comparable to the 25.4%
baseline) and set-of-rows equality (official BIRD style).
"""
import argparse
import csv
import glob
import json
import os
from collections import Counter

import requests

from common import (BASE, DB_DIR, clean_sql, db_path_for, get_schema,
                    load_questions, run_query)

OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:27b"

PROMPT = """given this SQLite schema:
    {schema}
    Question: {question}

    Write ONE SQLite query that answer it. Output ONLY the SQL, nothing else."""

PROMPT_EVIDENCE = """given this SQLite schema:
    {schema}
    Question: {question}
    Hint: {evidence}

    Write ONE SQLite query that answer it. Output ONLY the SQL, nothing else."""

RETRY_MSG = ("That SQL failed with this SQLite error:\n{error}\n"
             "Fix the query. Output ONLY the SQL, nothing else.")

GUIDELINES = """
    Rules:
    - SELECT exactly the columns the question asks for, in that order. No extra columns, even if they seem helpful.
    - Do not round, format, CAST, or concatenate values unless the question explicitly asks for it.
    - When the hint gives a formula or definition, follow it exactly.
    - Only join tables that are strictly needed to answer the question."""

REFINE_MSG = ("Your query returned these first rows:\n{result}\n"
              "Check your SQL against the question: exactly the requested "
              "columns (no extras), right filters, right aggregation. "
              "If it is correct, output the SAME SQL unchanged. "
              "If not, output the corrected SQL. Output ONLY the SQL.")


def chat(messages, think, temperature, num_ctx, num_predict):
    resp = requests.post(OLLAMA, json={
        "model": MODEL,
        "messages": messages,
        "think": think,
        "stream": False,
        "options": {"num_ctx": num_ctx, "temperature": temperature,
                    "num_predict": num_predict},
    }, timeout=900)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def sample_rows(db_path, per_table=3):
    tables = [r[0] for r in run_query(
        db_path, "SELECT name FROM sqlite_master WHERE type='table'")]
    parts = []
    for t in tables:
        try:
            rows = run_query(db_path, f'SELECT * FROM "{t}" LIMIT {per_table}')
        except Exception:
            continue
        parts.append(f"-- 3 sample rows from {t}:\n-- " +
                     "\n-- ".join(str(r)[:300] for r in rows))
    return "\n".join(parts)


def result_key(rows):
    return tuple(sorted(map(str, rows)))


_desc_cache = {}


def column_descriptions(db_id):
    if db_id in _desc_cache:
        return _desc_cache[db_id]
    parts = []
    pattern = f"{DB_DIR}{db_id}/database_description/*.csv"
    for path in sorted(glob.glob(pattern)):
        table = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, encoding="utf-8-sig", errors="replace") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue
        lines = []
        for r in rows:
            col = (r.get("original_column_name") or "").strip()
            desc = (r.get("column_description") or "").strip()
            val = (r.get("value_description") or "").strip()
            txt = "; ".join(x for x in (desc, val) if x)
            txt = " ".join(txt.split())[:200]
            if col and txt and txt.lower() != col.lower():
                lines.append(f"--   {col}: {txt}")
        if lines:
            parts.append(f"-- column meanings for {table}:\n"
                         + "\n".join(lines))
    _desc_cache[db_id] = "\n".join(parts)
    return _desc_cache[db_id]


def solve(ex, args):
    """Return (sql, attempts) for one question."""
    db_path = db_path_for(ex)
    schema = get_schema(db_path)
    if args.rows:
        schema += "\n\n" + sample_rows(db_path)
    if args.desc:
        schema += "\n\n" + column_descriptions(ex["db_id"])
    if args.evidence and ex.get("evidence"):
        user = PROMPT_EVIDENCE.format(schema=schema, question=ex["question"],
                                      evidence=ex["evidence"])
    else:
        user = PROMPT.format(schema=schema, question=ex["question"])
    if args.guidelines:
        user += GUIDELINES

    def one_chain(temperature):
        msgs = [{"role": "user", "content": user}]
        raw = chat(msgs, args.think, temperature, args.ctx,
                   args.num_predict)
        sql = clean_sql(raw)
        tries = 1
        for _ in range(args.retry):
            try:
                run_query(db_path, sql)
                break
            except Exception as e:
                msgs += [{"role": "assistant", "content": raw},
                         {"role": "user",
                          "content": RETRY_MSG.format(error=e)}]
                raw = chat(msgs, args.think, temperature, args.ctx,
                   args.num_predict)
                sql = clean_sql(raw)
                tries += 1
        return sql, tries

    def refine_pass(sql):
        try:
            rows = run_query(db_path, sql)[:5]
            result = "\n".join(str(r)[:200] for r in rows) or "(no rows)"
        except Exception as e:
            result = f"error: {e}"
        msgs = [{"role": "user", "content": user},
                {"role": "assistant", "content": sql},
                {"role": "user",
                 "content": REFINE_MSG.format(result=result)}]
        raw = chat(msgs, args.think, 0, args.ctx, args.num_predict)
        refined = clean_sql(raw)
        try:
            run_query(db_path, refined)
        except Exception:
            return sql  # keep the original if the refinement broke it
        return refined

    if args.sc <= 1:
        sql, tries = one_chain(0)
        if args.refine:
            sql = refine_pass(sql)
            tries += 1
        return sql, tries

    # self-consistency: majority vote on execution result
    candidates = []
    for _ in range(args.sc):
        sql, tries = one_chain(0.7)
        try:
            key = result_key(run_query(db_path, sql))
        except Exception:
            key = "<error>"
        candidates.append((sql, key, tries))
    votes = Counter(k for _, k, _ in candidates if k != "<error>")
    if not votes:
        return candidates[0][0], sum(c[2] for c in candidates)
    best_key = votes.most_common(1)[0][0]
    sql = next(s for s, k, _ in candidates if k == best_key)
    return sql, sum(c[2] for c in candidates)


def main():
    global MODEL
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--evidence", action="store_true")
    p.add_argument("--think", action="store_true")
    p.add_argument("--retry", type=int, default=0)
    p.add_argument("--rows", action="store_true")
    p.add_argument("--desc", action="store_true")
    p.add_argument("--guidelines", action="store_true")
    p.add_argument("--refine", action="store_true")
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--num-predict", type=int, default=-1,
                   help="cap total output tokens (bounds thinking cost)")
    p.add_argument("--sc", type=int, default=1)
    p.add_argument("--stride", type=int, default=1,
                   help="take every Nth question (spread probe subset)")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    MODEL = args.model

    ds = load_questions()[::args.stride]
    if args.limit:
        ds = ds[:args.limit]

    per_question = []
    n_ordered = n_set = 0
    for i, ex in enumerate(ds):
        db_path = db_path_for(ex)
        try:
            sql, tries = solve(ex, args)
        except Exception as e:
            sql, tries = f"<generation error: {e}>", 0

        try:
            model_answer = run_query(db_path, sql)
        except Exception:
            model_answer = None
        try:
            gold_answer = run_query(db_path, ex["SQL"])
        except Exception:
            gold_answer = "<gold failed>"

        ok_ordered = model_answer == gold_answer
        ok_set = (model_answer is not None
                  and not isinstance(gold_answer, str)
                  and result_key(model_answer) == result_key(gold_answer))
        n_ordered += ok_ordered
        n_set += ok_set
        per_question.append({
            "question_id": ex["question_id"],
            "db_id": ex["db_id"],
            "difficulty": ex.get("difficulty"),
            "question": ex["question"],
            "model_sql": sql,
            "tries": tries,
            "sql_ran": model_answer is not None,
            "correct": ok_ordered,
            "correct_set": ok_set,
        })
        print(f"[{args.tag}] {i + 1}/{len(ds)} "
              f"ordered: {n_ordered}  set: {n_set}", flush=True)

    summary = {
        "tag": args.tag,
        "model": MODEL,
        "config": {k: getattr(args, k) for k in
                   ("evidence", "think", "retry", "rows", "desc",
                    "guidelines", "refine", "sc", "stride", "limit")},
        "total": len(ds),
        "correct": n_ordered,
        "accuracy": n_ordered / len(ds),
        "correct_set": n_set,
        "accuracy_set": n_set / len(ds),
        "sql_ran_rate": sum(q["sql_ran"] for q in per_question) / len(ds),
    }
    out_path = f"{BASE}/results/score_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_question": per_question}, f,
                  indent=1)
    print(f"[{args.tag}] FINAL accuracy: {summary['accuracy']:.4f} "
          f"(set: {summary['accuracy_set']:.4f})", flush=True)


if __name__ == "__main__":
    main()
