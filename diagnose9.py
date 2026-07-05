import json
from pathlib import Path
from rag_revops.config import load_settings
from rag_revops.vectorstore import ChromaStore
from rag_revops.embeddings import CohereEmbedder
from rag_revops.rerank import CohereReranker

s = load_settings()
store = ChromaStore(s.vectorstore)
emb = CohereEmbedder(s.embeddings)
reranker = CohereReranker(s.rerank)

gold = set()
for line in open("eval/eval_seed.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"].rsplit("__", 1)[-1] == "Termination For Convenience":
        gold.add(Path(r["contract_file"]).stem)

# one representative per contract, reranked (same as the real pipeline)
vec = emb.embed_query("Which contracts allow termination for convenience?")
pool = store.query(vec, top_k=store.count())
best = {}
for ch in pool:
    if ch.doc_id not in best:
        best[ch.doc_id] = ch
reranked = reranker.rerank("Which contracts allow termination for convenience?", list(best.values()))

# Where do the 40 gold contracts land in the reranked ordering?
ranks = [(i, c.doc_id in gold) for i, c in enumerate(reranked, 1)]
gold_ranks = [i for i, is_gold in ranks if is_gold]
print(f"gold contracts and their rerank POSITIONS (out of {len(reranked)}):")
print(f"  {sorted(gold_ranks)}")
print(f"\ngold in top 30: {sum(1 for r in gold_ranks if r <= 30)} / {len(gold)}")
print(f"gold in top 50: {sum(1 for r in gold_ranks if r <= 50)} / {len(gold)}")