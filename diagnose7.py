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

# ground truth
gold = set()
for line in open("eval/eval_seed.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"].rsplit("__", 1)[-1] == "Termination For Convenience":
        gold.add(Path(r["contract_file"]).stem)

# The concept expressed the way contracts ACTUALLY phrase it
expansions = [
    "terminate this agreement for convenience",
    "terminate this agreement without cause",
    "terminate at any time in its sole discretion",
    "terminate without penalty upon written notice",
    "either party may terminate upon prior written notice",
    "terminate for any reason upon notice",
]

# Retrieve each expansion, collect best chunk per contract across ALL expansions
best_chunk = {}
for exp in expansions:
    vec = emb.embed_query(exp)
    pool = store.query(vec, top_k=store.count())
    for ch in pool:
        # keep the highest dense-similarity chunk per contract, across expansions
        if ch.doc_id not in best_chunk or ch.score > best_chunk[ch.doc_id].score:
            best_chunk[ch.doc_id] = ch

reps = list(best_chunk.values())
print(f"reranking {len(reps)} contract representatives (one per contract, best across expansions)...")

# Rerank against the ORIGINAL business query
reranked = reranker.rerank("Which contracts allow termination for convenience?", reps)

best_by_contract = {c.doc_id: c.score for c in reranked}
gold_scores = sorted((best_by_contract.get(d, 0.0) for d in gold), reverse=True)

for thresh in [0.10, 0.05, 0.02, 0.01]:
    n = sum(1 for x in gold_scores if x >= thresh)
    print(f"gold above {thresh}: {n} / {len(gold)}")
print(f"\ntop 20 gold scores: {[round(x,3) for x in gold_scores[:20]]}")