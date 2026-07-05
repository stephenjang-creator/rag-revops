import json
from pathlib import Path
from rag_revops.config import load_settings
from rag_revops.vectorstore import ChromaStore
from rag_revops.embeddings import CohereEmbedder

s = load_settings()
store = ChromaStore(s.vectorstore)
emb = CohereEmbedder(s.embeddings)

# Take 5 gold contracts and their actual annotated clause spans
gold_spans = {}
for line in open("eval/eval_seed.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"].rsplit("__", 1)[-1] == "Termination For Convenience":
        stem = Path(r["contract_file"]).stem
        gold_spans.setdefault(stem, []).append(r["gold_answer"])

# Retrieve the pool the way the analytical retriever does
vec = emb.embed_query("Which contracts allow termination for convenience?")
pool = store.query(vec, top_k=min(s.analytical.pool_size, store.count()))
by_contract = {}
for ch in pool:
    by_contract.setdefault(ch.doc_id, [])
    if len(by_contract[ch.doc_id]) < s.analytical.chunks_per_contract:
        by_contract[ch.doc_id].append(ch)

# For 5 gold contracts: is the clause text actually IN the retrieved chunks?
checked = 0
for stem, spans in gold_spans.items():
    if stem not in by_contract:
        continue
    chunks_text = " ".join(c.text for c in by_contract[stem]).lower()
    # check if a distinctive fragment of the gold span appears in the chunks
    span_frag = spans[0][:40].lower()
    found = span_frag in chunks_text
    print(f"{'FOUND' if found else 'MISSING'}: {stem[:35]}")
    print(f"    gold clause: {spans[0][:70]}")
    print(f"    in retrieved chunks: {found}")
    checked += 1
    if checked >= 5:
        break