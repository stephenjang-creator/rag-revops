"""Print how many eval-seed questions fall into each clause category.

A quick corpus-shape sanity check: reads eval/eval_seed.jsonl and tallies the
category suffix of each question id. Run from the repo root:

    python scripts/category_counts.py
"""

import json
from collections import Counter

cats = Counter()

with open("eval/eval_seed.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        cats[r["id"].rsplit("__", 1)[-1]] += 1

for cat, n in sorted(cats.items()):
    print(f"{n:4d}  {cat}")
print(f"\n{len(cats)} distinct categories")