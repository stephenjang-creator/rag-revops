import re
from pathlib import Path
from rag_revops.config import load_settings
from rag_revops.vectorstore import ChromaStore
from rag_revops.embeddings import CohereEmbedder

s = load_settings()
store = ChromaStore(s.vectorstore)
emb = CohereEmbedder(s.embeddings)

# Ground truth by grepping: which contracts actually contain a 30-day payment phrasing?
patterns = [r"net 30", r"net thirty", r"thirty \(30\) days", r"30 days", r"within thirty"]
rx = re.compile("|".join(patterns), re.IGNORECASE)

contracts_dir = Path("data/raw/contracts")
true_contracts = set()
for f in contracts_dir.glob("*.txt"):
    if rx.search(f.read_text(encoding="utf-8", errors="ignore")):
        true_contracts.add(f.stem)
print(f"contracts containing a 30-day phrasing (grep ground truth): {len(true_contracts)}")

# Now: of those, how many have that phrasing in their TOP chunks the judge sees?
vec = emb.embed_query("Which contracts have Net 30 payment terms?")
pool = store.query(vec, top_k=min(s.analytical.pool_size, store.count()))
by_contract = {}
for ch in pool:
    by_contract.setdefault(ch.doc_id, [])
    if len(by_contract[ch.doc_id]) < s.analytical.chunks_per_contract:
        by_contract[ch.doc_id].append(ch)

in_chunks = 0
missing = 0
for stem in true_contracts:
    if stem not in by_contract:
        continue
    chunk_text = " ".join(c.text for c in by_contract[stem])
    if rx.search(chunk_text):
        in_chunks += 1
    else:
        missing += 1
print(f"  of those, phrasing IS in the judge's chunks: {in_chunks}")
print(f"  phrasing MISSING from judge's chunks (retrieval miss): {missing}")
print(f"\nThis tells us: retriever surfaces the right chunk for {in_chunks}/{len(true_contracts)} true contracts")