import json
from pathlib import Path
from collections import defaultdict
from rag_revops.config import load_settings
from rag_revops.vectorstore import ChromaStore
from rag_revops.embeddings import CohereEmbedder
from rag_revops.hybrid import HybridRetriever
from rag_revops.rerank import CohereReranker

s = load_settings()
store = ChromaStore(s.vectorstore)
print(f"total chunks in store: {store.count()}")
print(f"analytical.fetch_k:    {s.analytical.fetch_k}")
print(f"coverage: fetching {s.analytical.fetch_k}/{store.count()} = {s.analytical.fetch_k/store.count()*100:.0f}% of chunks")

# 1. Ground truth from CUAD: which contracts have "Termination For Convenience"
category = "Termination For Convenience"
gold = set()
for line in open("eval/eval_seed.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"].rsplit("__", 1)[-1] == category:
        gold.add(Path(r["contract_file"]).stem)
print(f"GOLD: {len(gold)} contracts have {category!r}")

# 2. What does retrieval surface, and at what rerank scores per contract?
s = load_settings()
store, emb = ChromaStore(s.vectorstore), CohereEmbedder(s.embeddings)
hybrid = HybridRetriever(s, store, emb)
reranker = CohereReranker(s.rerank)

q = "Which contracts allow termination for convenience?"
fused = hybrid.retrieve(q, top_k=s.analytical.fetch_k, fetch_k=s.analytical.fetch_k)
reranked = reranker.rerank(q, fused)

# best rerank score per contract
best = defaultdict(float)
for c in reranked:
    best[c.doc_id] = max(best[c.doc_id], c.score)

# 3. For each GOLD contract, did we retrieve it, and what was its best score?
print(f"\nGold contracts and their best rerank score (threshold={s.analytical.min_relevance}):")
for doc_id in sorted(best, key=best.get, reverse=True):
    if doc_id in gold:
        mark = "PASS" if best[doc_id] >= s.analytical.min_relevance else "DROPPED"
        print(f"  {best[doc_id]:.3f}  [{mark}]  {doc_id[:45]}")