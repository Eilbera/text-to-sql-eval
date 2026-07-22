"""Analysis tools for the accuracy push.

  python analyze.py table            -> ablation table of all score_*.json
  python analyze.py errors TAG       -> failure buckets for one run
  python analyze.py diff TAG1 TAG2   -> questions where runs disagree
"""
import glob
import json
import os
import sys
from collections import Counter

from common import BASE, db_path_for, run_query

RESULTS = BASE + "/results"


def load_all():
    runs = {}
    for path in sorted(glob.glob(f"{RESULTS}/score_*.json")):
        tag = os.path.basename(path)[len("score_"):-len(".json")]
        with open(path) as f:
            runs[tag] = json.load(f)
    return runs


def table():
    print(f"{'tag':24s} {'n':>4s} {'acc':>7s} {'set':>7s} {'ran':>6s}  config")
    for tag, r in load_all().items():
        s = r["summary"]
        acc_set = s.get("accuracy_set")
        cfg = s.get("config", {})
        cfg_str = " ".join(k if v is True else f"{k}={v}"
                           for k, v in cfg.items()
                           if v not in (False, None, 0, 1)) or "-"
        print(f"{tag:24s} {s['total']:4d} {s['accuracy']:7.1%} "
              f"{acc_set:7.1%}" if acc_set is not None else
              f"{tag:24s} {s['total']:4d} {s['accuracy']:7.1%} {'':7s}",
              end="")
        print(f" {s['sql_ran_rate']:6.1%}  {cfg_str}")


def errors(tag):
    with open(f"{RESULTS}/score_{tag}.json") as f:
        r = json.load(f)
    fails = [q for q in r["per_question"] if not q.get("correct_set",
                                                       q["correct"])]
    print(f"{tag}: {len(fails)} failures of {len(r['per_question'])}")

    buckets = Counter()
    for q in fails:
        if not q["sql_ran"]:
            buckets["A_sql_error"] += 1
            continue
        db_path = db_path_for(q)
        try:
            rows = run_query(db_path, q["model_sql"])
        except Exception:
            buckets["A_sql_error"] += 1
            continue
        if not rows:
            buckets["B_ran_but_empty"] += 1
        else:
            buckets["C_ran_wrong_result"] += 1
    print("failure buckets:", dict(buckets))

    by_diff = Counter(q.get("difficulty") for q in fails)
    total_diff = Counter(q.get("difficulty") for q in r["per_question"])
    print("failures by difficulty:",
          {d: f"{by_diff[d]}/{total_diff[d]}" for d in total_diff})

    print("\n--- sample failures (ran, wrong result) ---")
    shown = 0
    for q in fails:
        if not q["sql_ran"] or shown >= 8:
            continue
        print(f"\n#{q['question_id']} [{q['db_id']}] ({q.get('difficulty')})")
        print(f"  Q: {q['question'][:160]}")
        print(f"  model: {q['model_sql'][:200]}")
        shown += 1


def diff(tag1, tag2):
    runs = load_all()
    q1 = {q["question_id"]: q for q in runs[tag1]["per_question"]}
    q2 = {q["question_id"]: q for q in runs[tag2]["per_question"]}
    common_ids = set(q1) & set(q2)
    key = lambda q: q.get("correct_set", q["correct"])
    fixed = [i for i in common_ids if not key(q1[i]) and key(q2[i])]
    broke = [i for i in common_ids if key(q1[i]) and not key(q2[i])]
    print(f"{tag1} -> {tag2} on {len(common_ids)} shared questions")
    print(f"fixed {len(fixed)}: {sorted(fixed)}")
    print(f"broke {len(broke)}: {sorted(broke)}")
    for i in sorted(broke)[:5]:
        print(f"\nBROKE #{i} [{q2[i]['db_id']}] {q2[i]['question'][:140]}")
        print(f"  {tag1}: {q1[i]['model_sql'][:180]}")
        print(f"  {tag2}: {q2[i]['model_sql'][:180]}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "table"
    if cmd == "table":
        table()
    elif cmd == "errors":
        errors(sys.argv[2])
    elif cmd == "diff":
        diff(sys.argv[2], sys.argv[3])
