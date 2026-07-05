import json
from pathlib import Path

# Pull the actual gold ANSWER SPANS for termination-for-convenience.
# These are the exact clause texts CUAD annotated — what "the right answer" looks like.
spans = []
for line in open("eval/eval_seed.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"].rsplit("__", 1)[-1] == "Termination For Convenience":
        spans.append((Path(r["contract_file"]).stem, r["gold_answer"]))

print(f"total gold spans: {len(spans)}\n")
for name, span in spans[:12]:
    print(f"[{name[:30]}]")
    print(f"  {span[:200]}")
    print()