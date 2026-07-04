# PolicyLens — Grounded Q&A over Revenue Operations Documents

**A retrieval-augmented question-answering system that lets a rep, deal-desk analyst, or CS manager ask plain-English questions about contract terms, SLAs, and policy language — and get an answer with an exact citation, or an honest "I can't find that."**

Built to reduce the escalation tax: the constant back-and-forth where a rep pings Deal Desk to ask "can this customer terminate for convenience?" or "what's our data-return obligation on churn?" Most of those answers already live in the MSA, the SLA, or a service description. This system finds the passage, answers from it, and cites where it came from — so the human stays in the loop with a source to verify, never a hallucinated clause.

> **AI-first, human-in-the-loop.** The system is designed to *decline* rather than guess. If retrieved passages don't support an answer, it says so. Citations are enforced, not decorative.

---

## 🔗 Live demo

**[Try it here → `https://<your-app>.streamlit.app`](https://share.streamlit.io)** *(update this URL after deploying)*

The hosted demo is **bring-your-own-key**: it bakes in no API keys. Paste your own
Anthropic + Cohere keys into the sidebar — they stay in your browser session only,
are never logged, and are never committed. Reviewers without keys still see the full
UI, the corpus, the retrieved passages, and the citation / decline behavior.

*(A short walkthrough GIF goes here for reviewers who don't have keys.)*

---

## Why this exists (the ops problem)

| Without | With PolicyLens |
|---|---|
| Rep escalates to Deal Desk → wait → context-switch | Rep self-serves the answer in seconds |
| Answer buried in a 40-page PDF | Top passage surfaced with section reference |
| "I think our cap is 12 months of fees?" | Grounded answer + exact clause, or an explicit "not found" |

## Data provenance (zero proprietary data)

Every document in this project is **public**. Nothing from any employer is used.

- **CUAD** (Contract Understanding Atticus Dataset) — 510 real commercial contracts with expert clause annotations, released under **CC BY 4.0**. Used as both corpus and the backbone of the evaluation set.
- **Published SaaS agreements / SLAs** from vendors who post them publicly (e.g., model agreements and legal pages). These are copyrighted; this repo ships a **downloader script** and links to sources rather than redistributing raw PDFs.

See [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) for the licensing note and attribution.

---

## Architecture (three phases)

**Phase 1 — Fundamentals (this scaffold)**
Ingest PDF / Markdown / HTML → token-aware chunking (500–800 tokens, 100 overlap) → ChromaDB vector store → retrieval pipeline that pulls top-k chunks and generates a **cited** answer.

**Phase 2 — Production quality** *(planned)*
Hybrid retrieval (BM25 + vector) → Cohere cross-encoder re-ranker → citation enforcement (decline when unsupported) → all prompts in versioned config.

**Phase 3 — Faithfulness measurement** *(planned)*
50–200 human-verified Q/A pairs → offline faithfulness eval (Ragas) → CI gate that fails the build when quality drops below threshold.

## Tech stack

- **Orchestration:** LangGraph (explicit state graph — makes the "decline to answer" branch a first-class conditional edge)
- **Vector DB:** ChromaDB
- **Embeddings + Reranker:** Cohere (`embed` + `rerank`)
- **Generation:** Anthropic (Claude)
- **Evaluation:** Ragas
- **Language:** Python 3.11+

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Set keys (see .env.example)
cp .env.example .env   # add ANTHROPIC_API_KEY and COHERE_API_KEY

# 3. Fetch public corpus
python scripts/download_corpus.py

# 4. Ingest + chunk + embed into Chroma
python -m rag_revops.ingest --source data/raw --persist data/processed/chroma

# 5. Ask a question
python -m rag_revops.query "Can a customer terminate for convenience, and with how much notice?"
```

## Repo layout

```
rag-revops/
├── config/
│   └── settings.yaml         # all tunables: chunk size, top_k, model names, prompts
├── src/rag_revops/
│   ├── config.py             # typed settings loader (Pydantic)
│   ├── loaders.py            # PDF / MD / HTML → normalized text
│   ├── chunking.py           # token-aware splitter (500–800 tok, 100 overlap)
│   ├── embeddings.py         # Cohere embedding wrapper
│   ├── vectorstore.py        # Chroma persistence + query
│   ├── ingest.py             # CLI: load → chunk → embed → store
│   ├── retrieval.py          # top-k retrieval
│   ├── generation.py         # cited-answer generation (Anthropic)
│   ├── graph.py              # LangGraph pipeline wiring
│   └── query.py              # CLI: ask a question
├── scripts/
│   └── download_corpus.py    # fetch CUAD + public agreements
├── tests/
│   ├── test_chunking.py
│   └── test_loaders.py
├── docs/
│   └── DATA_PROVENANCE.md
├── .github/workflows/ci.yml  # lint + tests (eval gate added in Phase 3)
├── pyproject.toml
└── .env.example
```

## Deploying the live demo (Streamlit Community Cloud)

The demo runs from this repo with a **committed, prebuilt vector index** (Streamlit
Cloud's filesystem is ephemeral, so the index ships in the repo). Build it once
locally, then deploy.

```bash
# 1. Fetch corpus and build a small, public demo subset
python scripts/download_corpus.py
#    unzip CUAD: data/raw/cuad/data.zip -> data/raw/cuad/full_contract_txt/
python scripts/build_demo_subset.py --max-contracts 18

# 2. Build the index locally (needs COHERE_API_KEY for embeddings)
python -m rag_revops.ingest --source data/raw/demo_subset

# 3. Commit the prebuilt index (deliberate .gitignore exception)
git add -f data/processed/chroma
git commit -m "Add prebuilt demo vector index (public data only)"
git push

# 4. Deploy: share.streamlit.io -> New app -> point at this repo, main file app.py
```

Streamlit Cloud installs from `requirements.txt` and reruns on every push. No keys
are stored on the platform — the app reads keys the visitor enters in the sidebar.

**Cost/exposure:** none to you. The demo spends the *visitor's* API credits, and
your keys never touch the hosted runtime.

## License

Code: MIT. Data: see `docs/DATA_PROVENANCE.md` (CUAD is CC BY 4.0; vendor agreements are linked, not redistributed).
