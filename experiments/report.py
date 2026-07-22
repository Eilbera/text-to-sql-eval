"""Compare score_*.json files into a final report: accuracies, deltas,
which questions the fine-tune fixed vs broke, per-db breakdown."""
import json
import os
from collections import defaultdict

from common import BASE

RESULTS = BASE + "/results"


def load(tag):
    path = f"{RESULTS}/score_{tag}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    tags = ["15b_base", "15b_tuned", "27b_ollama"]
    runs = {t: load(t) for t in tags}

    print("=== Summary ===")
    for t in tags:
        r = runs[t]
        if r is None:
            print(f"{t:12s}  (missing)")
            continue
        s = r["summary"]
        print(f"{t:12s}  acc {s['accuracy']:.1%}  "
              f"({s['correct']}/{s['total']})  "
              f"sql ran {s['sql_ran_rate']:.1%}")

    base, tuned = runs["15b_base"], runs["15b_tuned"]
    if base and tuned:
        bq = {q["question_id"]: q for q in base["per_question"]}
        tq = {q["question_id"]: q for q in tuned["per_question"]}
        fixed = [i for i in bq if not bq[i]["correct"] and tq[i]["correct"]]
        broke = [i for i in bq if bq[i]["correct"] and not tq[i]["correct"]]
        print(f"\n=== Fine-tune effect (1.5B) ===")
        print(f"fixed  {len(fixed)} questions: {sorted(fixed)[:20]}")
        print(f"broke  {len(broke)} questions: {sorted(broke)[:20]}")

        print("\n=== Per-db accuracy (base -> tuned) ===")
        per_db = defaultdict(lambda: [0, 0, 0])  # total, base_ok, tuned_ok
        for i, q in bq.items():
            d = per_db[q["db_id"]]
            d[0] += 1
            d[1] += q["correct"]
            d[2] += tq[i]["correct"]
        for db, (n, b, t) in sorted(per_db.items()):
            print(f"{db:28s} {b:3d} -> {t:3d}  of {n}")

        print("\n=== Sample fixed questions ===")
        for i in sorted(fixed)[:5]:
            print(f"\n#{i} [{tq[i]['db_id']}] {tq[i]['question']}")
            print(f"  base : {bq[i]['model_sql'][:150]}")
            print(f"  tuned: {tq[i]['model_sql'][:150]}")


if __name__ == "__main__":
    main()
