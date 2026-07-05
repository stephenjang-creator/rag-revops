import json
from pathlib import Path
from collections import defaultdict
from rag_revops.config import load_settings
from rag_revops.vectorstore import ChromaStore

s = load_settings()
store = ChromaStore(s.vectorstore)

# What doc_ids does Chroma actually store?
pool = store.get_all()
store_doc_ids = set(ch.doc_id for ch in pool)
print("Sample stored doc_ids:")
for d in list(store_doc_ids)[:3]:
    print(f"  {d!r}")

# What do the gold contract_file stems look like?
gold_files = set()
for line in open("eval/eval_seed.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"].rsplit("__", 1)[-1] == "Termination For Convenience":
        gold_files.add(r["contract_file"])
print("\nSample gold contract_file values:")
for g in list(gold_files)[:3]:
    print(f"  {g!r}  -> stem: {Path(g).stem!r}")

# The critical test: do gold stems actually intersect stored doc_ids?
gold_stems = {Path(g).stem for g in gold_files}
overlap = gold_stems & store_doc_ids
print(f"\ngold stems: {len(gold_stems)}")
print(f"overlap with stored doc_ids: {len(overlap)}")
print(f"gold stems NOT in store: {len(gold_stems - store_doc_ids)}")