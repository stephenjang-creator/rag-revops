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

vec = emb.embed_query(q)
pool = store.query(vec, top_k=min(s.analytical.pool_size, store.count()))
best = {}
for ch in pool:
    if ch.doc_id not in best:
        best[ch.doc_id] = ch
reps = list(best.values())
print(f"representatives (one per contract): {len(reps)}")

# THE KEY LINE - pass top_n=len(reps) explicitly
reranked = reranker.rerank(q, reps, top_n=len(reps))
print(f"reranked results returned: {len(reranked)}")

above = [c for c in reranked if c.score >= s.analytical.min_relevance]
print(f"above min_relevance ({s.analytical.min_relevance}): {len(above)}")
print(f"\nscore distribution of top 20:")
for c in reranked[:20]:
    print(f"  {c.score:.4f}  {c.doc_id[:40]}")