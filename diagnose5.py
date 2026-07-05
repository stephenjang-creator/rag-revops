import json
from pathlib import Path
from collections import defaultdict
from rag_revops.config import load_settings
from rag_revops.vectorstore import ChromaStore
from rag_revops.embeddings import CohereEmbedder
from rag_revops.rerank import CohereReranker

s = load_settings()
store = ChromaStore(s.vectorstore)
emb = CohereEmbedder(s.embeddings)
reranker = CohereReranker(s.rerank)

q = "Which contracts allow termination for convenience?"

gold = set()
for line in open("eval/eval_seed.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"].rsplit("__", 1)[-1] == "Termination For Convenience":
        gold.add(Path(r["contract_file"]).stem)

# Pull the full pool, group ALL chunks by contract (not just the best one)
vec = emb.embed_query(q)
pool = store.query(vec, top_k=store.count())
by_contract = defaultdict(list)
for ch in pool:
    by_contract[ch.doc_id].append(ch)

# Take up to 5 chunks per contract, rerank the whole batch, aggregate by max
candidates = []
for doc_id, chunks in by_contract.items():
    candidates.extend(chunks[:5])
print(f"reranking {len(candidates)} chunks (up to 5 per contract)...")
reranked = reranker.rerank(q, candidates)

best_by_contract = defaultdict(float)
for c in reranked:
    best_by_contract[c.doc_id] = max(best_by_contract[c.doc_id], c.score)

gold_scores = sorted((best_by_contract[d] for d in gold), reverse=True)
print(f"gold contracts above 0.10: {sum(1 for x in gold_scores if x >= 0.10)} / {len(gold)}")
print(f"gold contracts above 0.02: {sum(1 for x in gold_scores if x >= 0.02)} / {len(gold)}")
print(f"top 15 gold scores: {[round(x,3) for x in gold_scores[:15]]}")