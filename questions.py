import json
from collections import Counter

cats = Counter()
for line in open("eval/eval_seed.jsonl", encoding="utf-8"):
    r = json.loads(line)
    cats[r["id"].rsplit("__", 1)[-1]] += 1

for cat, n in sorted(cats.items()):
    print(f"{n:4d}  {cat}")
print(f"\n{len(cats)} distinct categories")