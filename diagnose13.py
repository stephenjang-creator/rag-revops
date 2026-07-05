import re
from pathlib import Path
from rag_revops.config import load_settings
from rag_revops.vectorstore import ChromaStore
from rag_revops.embeddings import CohereEmbedder

s = load_settings()
store = ChromaStore(s.vectorstore)
emb = CohereEmbedder(s.embeddings)

# Look at what the judge actually SEES for a few contracts that have payment language
rx = re.compile(r"(net 30|thirty \(30\) days|within thirty|30 days)", re.IGNORECASE)
pay_rx = re.compile(r"(pay|payment|invoice|remit|due|amount)", re.IGNORECASE)

vec = emb.embed_query("Which contracts have Net 30 payment terms?")
pool = store.query(vec, top_k=min(s.analytical.pool_size, store.count()))
by_contract = {}
for ch in pool:
    by_contract.setdefault(ch.doc_id, [])
    if len(by_contract[ch.doc_id]) < s.analytical.chunks_per_contract:
        by_contract[ch.doc_id].append(ch)

shown = 0
for doc_id, chunks in by_contract.items():
    text = " ".join(c.text for c in chunks)
    m = rx.search(text)
    if not m:
        continue
    # show the 30-day phrase in context — is payment language nearby?
    idx = m.start()
    window = text[max(0, idx-120):idx+120]
    has_payment_nearby = bool(pay_rx.search(window))
    print(f"[{doc_id[:35]}] payment-word near '30 days'? {has_payment_nearby}")
    print(f"    ...{window.strip()}...")
    print()
    shown += 1
    if shown >= 8:
        break