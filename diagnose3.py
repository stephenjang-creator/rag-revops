from collections import defaultdict
from pathlib import Path
import json
from rag_revops.config import load_settings
from rag_revops.vectorstore import ChromaStore
from rag_revops.embeddings import CohereEmbedder
from rag_revops.rerank import CohereReranker

s = load_settings()
store = ChromaStore(s.vectorstore)
emb = CohereEmbedder(s.embeddings)
reranker = CohereReranker(s.rerank)

q = "Which contracts allow termination for convenience?"

# ground truth
gold = set()
for line in open("eval/eval_seed.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"].rsplit("__", 1)[-1] == "Termination For Convenience":
        gold.add(Path(r["contract_file"]).stem)

# one representative per contract
vec = emb.embed_query(q)
pool = store.query(vec, top_k=min(s.analytical.pool_size, store.count()))
best = {}
for ch in pool:
    if ch.doc_id not in best:
        best[ch.doc_id] = ch
reps = list(best.values())

reranked = reranker.rerank(q, reps)

# how do GOLD contracts score vs non-gold?
gold_scores = [c.score for c in reranked if c.doc_id in gold]
nongold_scores = [c.score for c in reranked if c.doc_id not in gold]
print(f"gold contracts: {len(gold_scores)}  |  rerank score range: {min(gold_scores):.3f} - {max(gold_scores):.3f}")
print(f"  gold above 0.10: {sum(1 for x in gold_scores if x >= 0.10)}")
print(f"  gold above 0.05: {sum(1 for x in gold_scores if x >= 0.05)}")
print(f"  gold above 0.02: {sum(1 for x in gold_scores if x >= 0.02)}")
print(f"\ntop 15 gold contract scores: {sorted(gold_scores, reverse=True)[:15]}")
print(f"\nnon-gold above 0.10: {sum(1 for x in nongold_scores if x >= 0.10)} (these would be false positives)")