# Deal Desk Helper

**A citation-grounded contract-analysis tool that answers two questions a Deal Desk actually asks: _"what does this contract say about X?"_ and _"which contracts across our whole book have clause X?"_ — and bridges the gap between how a business user phrases a question and how contracts are actually written.**

Built to cut the escalation tax — the constant back-and-forth where a rep pings Deal Desk to ask "can this customer terminate for convenience?" or "which of our agreements have uncapped liability?" Those answers live in the contracts; this tool finds them, cites the exact passage, and stays honest — it declines rather than guessing when the documents don't support an answer.

> **AI-first, human-in-the-loop.** Answers are grounded in retrieved passages and cite their source. When support is weak, the system says so instead of hallucinating a clause. Built in Python.

---

## 🔗 Live demo

**[Try it → `https://rag-revops.streamlit.app`](https://rag-revops.streamlit.app)

Bring-your-own-key: the hosted demo bakes in no API keys. Paste your own Anthropic + Cohere keys into the sidebar — they live only in your browser session, never logged, never committed. Reviewers without keys still see the full UI, the corpus, and the retrieval/citation behavior.

---

## What it does

**Two modes over the same corpus:**

**1. Ask about a contract** — pick one contract (searchable list) and ask in plain English. The answer draws only from that contract, cites the passages it used with inline `[n]` markers, and declines when the contract doesn't cover the question. A "search all contracts" toggle widens the scope when you want it.

**2. Find contracts across the corpus** — "which contracts allow termination for convenience?" scans every contract and returns the matching set, each with a one-line reason and citation. This is the harder capability, and the interesting one: it recognizes clauses that mean the same thing but are worded completely differently.

**Query rewriting** sits in front of both. A Deal Desk user types "can either party do a TFC" and the system reformulates it to "Can either party terminate this agreement for convenience, and what notice is required?" before retrieving — expanding abbreviations and jargon into contract language, and showing the interpretation so the user can see and trust what was searched.

---

## The problem that makes this non-trivial

The hard part of "which contracts have X" is that **contracts rarely use the query's words.** A search for _termination for convenience_ has to match:

- "Either party may terminate this Agreement **without cause** upon thirty days' notice"
- "...terminate **in its sole discretion**..."
- "...terminate **for any reason**..."
- "...terminate **without penalty**..."

A keyword search misses all of these. A vector/embedding search scores them low — they don't resemble the phrase _termination for convenience_. Even a cross-encoder reranker (the standard precision tool) scores these paraphrased clauses near **zero**, because it judges surface similarity, not legal meaning.

The fix is to use an **LLM as the membership judge** for cross-corpus queries: Claude reads each contract's excerpts and decides — with explicit legal-equivalence knowledge — whether it genuinely satisfies the criterion, recognizing that "terminate without cause" _is_ termination for convenience. The reranker still orders which contracts the judge sees; it no longer gates membership. This recovered matches that pure retrieval scored at zero (16 vs. 4 on the termination query in one corpus).

---

## Architecture

```
Query
  |
  v
Query rewriting (Claude)  -- "TFC" -> "termination for convenience..."
  |
  +---------------- single-contract mode ----------------+
  |                                                       |
  |   Hybrid retrieval (BM25 + vector, RRF fusion)        |
  |   -> Cohere cross-encoder rerank                      |
  |   -> citation-enforced generation (Claude)            |
  |   -> answer with [n] citations, or decline            |
  |                                                       |
  +---------------- cross-corpus mode --------------------+
                                                          |
      Wide dense fetch (every contract represented)       |
      -> one representative chunk per contract            |
      -> rerank for ORDERING (not membership)             |
      -> LLM judge decides membership by legal meaning    |
      -> matching contracts, each with reason + citation  |
```

**Tech stack** (Python throughout):

| Layer | Choice | Why |
|---|---|---|
| Orchestration | **LangGraph** | Explicit state graph — the decline branch is a first-class conditional edge, not buried logic |
| Retrieval | **Hybrid: BM25 + vector, RRF fusion** | BM25 catches exact terms (section refs, figures); vectors catch paraphrase; RRF fuses on rank, not incompatible scores |
| Vector store | **ChromaDB** | Local, persistent, metadata filtering (enables single-contract scoping) |
| Embeddings + rerank | **Cohere** (`embed-v3`, `rerank-v3`) | Strong reranker; one vendor for both |
| Generation + judge | **Anthropic (Claude)** | Strong instruction-following for grounding; legal-equivalence reasoning the reranker lacks |
| Evaluation | **Custom LLM-as-judge + set-based P/R/F1** | Transparent, no heavy framework dependency |
| UI | **Streamlit** | Python-native, bring-your-own-key hosted demo |

Prompts live in a **versioned config file** (`config/settings.yaml`), not scattered string literals — changing a prompt is a reviewable diff.

---

## The engineering story (how the hard parts got solved)

The cross-corpus capability didn't work on the first build. Getting it right meant diagnosing three distinct failures **with data**, not guesswork — each ruled out with a targeted experiment before any fix:

**1. A silent result cap.** The reranker returned only its top-5 results regardless of how many contracts were passed in — so 95 of 100 contracts were dropped before scoring, and the eval's default-fill made them look like zero-relevance. Diagnosed by instrumenting the reranked-result count; fixed by threading a `top_n` override so analytical mode scores every contract while single-doc mode keeps its cap.

**2. A semantic gap.** With the cap fixed, the reranker still scored paraphrased clauses ("without cause," "sole discretion") at ~0.00 — indistinguishable from irrelevant text. A controlled test (feeding the reranker known clause text) proved the cross-encoder was working correctly; it simply has no legal knowledge that "without cause" means "for convenience." Fixed by moving membership from the reranker's score to an **LLM judge** that reasons about equivalence.

**3. Polysemy, correctly handled.** Testing "Net 30 payment terms" surfaced a subtle win: a grep for "30 days" returns 70 contracts, but most are cure periods, notice windows, or reporting deadlines — not payment terms. The LLM judge correctly _disambiguates_, confirming only the contracts where "30 days" governs payment. A keyword search can't do this; the judge can.

The takeaway that generalized: **the query-rewriting front-end and the LLM-judge back-end give jargon-tolerance at both ends** — the question going in and the clause matching coming out.

---

## Evaluation

Quality is measured, not asserted.

- **Golden set from CUAD annotations.** The [Contract Understanding Atticus Dataset](https://www.atticusprojectai.org/cuad) (510 real contracts, expert clause annotations, CC BY 4.0) provides ground truth: which contracts have which clause. The golden set is built directly from those annotations.
- **Single-doc faithfulness** — a custom LLM-as-judge evaluator decomposes each answer into claims and checks each against the retrieved context (declines excluded, since faithfulness on a non-answer is meaningless).
- **Cross-corpus precision/recall/F1** — set comparison against the CUAD ground-truth sets. Deterministic (no judge calls), so it's cheap enough to gate CI.
- **CI faithfulness gate** — GitHub Actions runs the eval on every pull request and fails the build if faithfulness drops below threshold, so a well-meaning change to chunking or a prompt can't silently degrade quality.

---

## Data provenance (zero proprietary data)

Every document is **public**. Nothing from any employer is used. The corpus and evaluation set are built from **CUAD** (CC BY 4.0). Handling licensing and provenance as a first-class concern — rather than quietly ingesting whatever's available — is itself part of the deliverable: a contract-analysis tool that can't say where its knowledge came from is a governance liability. See [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).

---

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cp .env.example .env          # add ANTHROPIC_API_KEY + COHERE_API_KEY

# Build the corpus + index from the CUAD file
python scripts/extract_cuad.py --input path/to/CUADv1.json --contracts-out data/raw/contracts --eval-out eval/eval_seed.jsonl --max-contracts 100
python -m rag_revops.ingest --source data/raw/contracts

# CLI
python -m rag_revops.query "How is liability capped?"
python -m rag_revops.query_analytical "Which contracts allow termination for convenience?"

# Or the full UI (both modes)
streamlit run app.py
```

## Evaluation & tests

```bash
pytest                                          # unit tests (retrieval, fusion, judge logic)
python -m eval.build_golden --seed eval/eval_seed.jsonl --out eval/golden.jsonl
python -m eval.run_eval --golden eval/golden.jsonl --limit 15          # faithfulness
python -m eval.build_analytical_golden --seed eval/eval_seed.jsonl --out eval/analytical_golden.jsonl
python -m eval.run_analytical_eval --golden eval/analytical_golden.jsonl --limit 3   # set-based P/R/F1
```

## Repo layout

```
src/rag_revops/
  loaders.py chunking.py embeddings.py vectorstore.py   # ingestion + storage
  bm25.py hybrid.py rerank.py retrieval.py              # retrieval
  query_rewrite.py                                       # jargon -> contract language
  graph.py generation.py query.py                        # single-doc pipeline (LangGraph)
  analytical.py analytical_generation.py query_analytical.py  # cross-corpus + LLM judge
eval/
  build_golden.py run_eval.py judge.py                   # faithfulness eval
  build_analytical_golden.py run_analytical_eval.py      # set-based eval
config/settings.yaml                                     # versioned prompts + tunables
app.py                                                   # Streamlit UI (both modes)
.github/workflows/ci.yml                                 # tests + faithfulness gate
```

## License

Code: MIT. Data: CUAD is CC BY 4.0 — see `docs/DATA_PROVENANCE.md`.
