from rag_revops.config import load_settings
from rag_revops.rerank import CohereReranker
from rag_revops.vectorstore import ChromaStore, RetrievedChunk

s = load_settings()
reranker = CohereReranker(s.rerank)

# Hand the reranker chunks with OBVIOUS termination-for-convenience language,
# taken verbatim from the gold spans we saw, and see what it scores them.
test_chunks = [
    RetrievedChunk("t1", "DocA", "Either party may terminate this Agreement without cause at any time effective upon thirty (30) days' written notice.", 0.0, {}),
    RetrievedChunk("t2", "DocB", "Either Party shall have the right to terminate this Agreement before the end of the Term for its convenience upon written notice to the other Party.", 0.0, {}),
    RetrievedChunk("t3", "DocC", "This Agreement may be terminated at any time, without the payment of any penalty, by the Board of Trustees.", 0.0, {}),
    RetrievedChunk("t4", "DocD", "The quick brown fox jumps over the lazy dog. This sentence is about nothing relevant.", 0.0, {}),
    RetrievedChunk("t5", "DocE", "Payment terms are net thirty days from invoice date.", 0.0, {}),
]

print("Reranking 5 hand-picked chunks against the query...")
print("(t1-t3 = real termination-for-convenience clauses; t4-t5 = irrelevant)\n")

reranked = reranker.rerank("Which contracts allow termination for convenience?", test_chunks)

for c in reranked:
    print(f"  {c.score:.4f}  {c.chunk_id}  {c.text[:60]}")

print(f"\nrerank model in config: {s.rerank.model}")