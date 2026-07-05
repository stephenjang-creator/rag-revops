from collections import defaultdict
from rag_revops.config import load_settings
from rag_revops.vectorstore import ChromaStore
from rag_revops.embeddings import CohereEmbedder

s = load_settings()
store = ChromaStore(s.vectorstore)
emb = CohereEmbedder(s.embeddings)

q = "Which contracts allow termination for convenience?"

# Step 1: how big is the raw dense pool, and how many distinct contracts does it touch?
pool_size = min(s.analytical.pool_size, store.count())
print(f"pool_size requested: {s.analytical.pool_size}, store count: {store.count()}, using: {pool_size}")

vec = emb.embed_query(q)
pool = store.query(vec, top_k=pool_size)
print(f"pool actually returned: {len(pool)} chunks")

contracts = defaultdict(int)
for ch in pool:
    contracts[ch.doc_id] += 1
print(f"distinct contracts represented in pool: {len(contracts)}")