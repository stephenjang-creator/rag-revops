# Deal Desk Helper

**A citation-grounded contract-analysis tool that answers two questions a Deal Desk actually asks: *"what does this contract say about X?"* and <i>"which contracts across our whole book have clause X?"</i> — and bridges the gap between how a business user phrases a question and how contracts are actually written.**

Built to cut the escalation tax — the constant back-and-forth where a rep pings Deal Desk to ask "can this customer terminate for convenience?" or "which of our agreements have uncapped liability?" Those answers live in the contracts; this tool finds them, cites the exact passage, and stays honest — it declines rather than guessing when the documents don't support an answer.

> \*\*AI-first, human-in-the-loop.\*\* Answers are grounded in retrieved passages and cite their source. When support is weak, the system says so instead of hallucinating a clause. Built in Python.

\---

## 🔗 Live demo

[**Try it → `https://rag-revops.streamlit.app`**](https://share.streamlit.io)

Bring-your-own-key: the hosted demo bakes in no API keys. Paste your own Anthropic + Cohere keys into the sidebar — they live only in your browser session, never logged, never committed. Reviewers without keys still see the full UI, the corpus, and the retrieval/citation behavior.

*(Walkthrough GIF here.)*

\---

## What it does

**Three modes over the same corpus:**

**1. Ask about a contract** — pick one contract (searchable list) and ask in plain English. The answer draws only from that contract, cites the passages it used with inline `\[n]` markers, and declines when the contract doesn't cover the question. To ask across the whole corpus, use mode 2.

**2. Find contracts across the corpus** — "which contracts allow termination for convenience?" scans every contract and returns the matching set, each with a one-line reason and citation. This is the harder capability, and the interesting one: it recognizes clauses that mean the same thing but are worded completely differently.

**3. Find clause language** — "give me language for a termination-for-convenience clause" (or an NDA, a limitation of liability, an indemnity). The **cross-encoder reranker** pulls the most on-point real passages from the corpus, then Claude **drafts suggested language for a new contract**, grounded in those passages and citing the source contracts with inline `\[n]` markers. So the headline output is reusable drafting language — traceable to real agreements, not invented — and it declines rather than fabricating a clause the corpus can't support. This is the exact opposite of mode 2: where mode 2 must *suppress* the reranker (it scores paraphrased clauses near zero), mode 3 is precision retrieval — the reranker's calibrated score is the signal that gates which passages are strong enough to draft from.

**Query rewriting** sits in front of all three. A Deal Desk user types "can either party do a TFC" and the system reformulates it to "Can either party terminate this agreement for convenience, and what notice is required?" before retrieving — expanding abbreviations and jargon into contract language, and showing the interpretation so the user can see and trust what was searched.

\---

## The problem that makes this non-trivial

The hard part of "which contracts have X" is that **contracts rarely use the query's words.** A search for *termination for convenience* has to match:

* "Either party may terminate this Agreement **without cause** upon thirty days' notice"
* "...terminate **in its sole discretion**..."
* "...terminate **for any reason**..."
* "...terminate **without penalty**..."

A keyword search misses all of these. A vector/embedding search scores them low — they don't resemble the phrase *termination for convenience*. Even a cross-encoder reranker (the standard precision tool) scores these paraphrased clauses near **zero**, because it judges surface similarity, not legal meaning.

The fix is to use an **LLM as the membership judge** for cross-corpus queries: Claude reads each contract's excerpts and decides — with explicit legal-equivalence knowledge — whether it genuinely satisfies the criterion, recognizing that "terminate without cause" *is* termination for convenience. The reranker still orders which contracts the judge sees; it no longer gates membership. This recovered matches that pure retrieval scored at zero (16 vs. 4 on the termination query in one corpus).

\---

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
  |   -> answer with \[n] citations, or decline            |
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

|Layer|Choice|Why|
|-|-|-|
|Orchestration|**LangGraph**|Explicit state graph — the decline branch is a first-class conditional edge, not buried logic|
|Retrieval|**Hybrid: BM25 + vector, RRF fusion**|BM25 catches exact terms (section refs, figures); vectors catch paraphrase; RRF fuses on rank, not incompatible scores|
|Vector store|**ChromaDB**|Local, persistent, metadata filtering (enables single-contract scoping)|
|Embeddings + rerank|**Cohere** (`embed-v3`, `rerank-v3`)|Strong reranker; one vendor for both|
|Generation + judge|**Anthropic (Claude)**|Strong instruction-following for grounding; legal-equivalence reasoning the reranker lacks|
|Evaluation|**Custom LLM-as-judge + set-based P/R/F1**|Transparent, no heavy framework dependency|
|UI|**Streamlit**|Python-native, bring-your-own-key hosted demo|

Prompts live in a **versioned config file** (`config/settings.yaml`), not scattered string literals — changing a prompt is a reviewable diff.

\---

## The engineering story (how the hard parts got solved)

The cross-corpus capability didn't work on the first build. Getting it right meant diagnosing three distinct failures **with data**, not guesswork — each ruled out with a targeted experiment before any fix:

**1. A silent result cap.** The reranker returned only its top-5 results regardless of how many contracts were passed in — so 95 of 100 contracts were dropped before scoring, and the eval's default-fill made them look like zero-relevance. Diagnosed by instrumenting the reranked-result count; fixed by threading a `top\_n` override so analytical mode scores every contract while single-doc mode keeps its cap.

**2. A semantic gap.** With the cap fixed, the reranker still scored paraphrased clauses ("without cause," "sole discretion") at \~0.00 — indistinguishable from irrelevant text. A controlled test (feeding the reranker known clause text) proved the cross-encoder was working correctly; it simply has no legal knowledge that "without cause" means "for convenience." Fixed by moving membership from the reranker's score to an **LLM judge** that reasons about equivalence.

**3. Polysemy, correctly handled.** Testing "Net 30 payment terms" surfaced a subtle win: a grep for "30 days" returns 70 contracts, but most are cure periods, notice windows, or reporting deadlines — not payment terms. The LLM judge correctly *disambiguates*, confirming only the contracts where "30 days" governs payment. A keyword search can't do this; the judge can.

The takeaway that generalized: **the query-rewriting front-end and the LLM-judge back-end give jargon-tolerance at both ends** — the question going in and the clause matching coming out.

\---

## Evaluation

Quality is measured, not asserted.

* **Golden set from CUAD annotations.** The [Contract Understanding Atticus Dataset](https://www.atticusprojectai.org/cuad) (510 real contracts, expert clause annotations, CC BY 4.0) provides ground truth: which contracts have which clause. The golden set is built directly from those annotations.
* **Single-doc faithfulness** — a custom LLM-as-judge evaluator decomposes each answer into claims and checks each against the retrieved context (declines excluded, since faithfulness on a non-answer is meaningless).
* **Cross-corpus precision/recall/F1** — set comparison against the CUAD ground-truth sets. Deterministic (no judge calls), so it's cheap enough to gate CI.
* **CI faithfulness gate** — GitHub Actions runs the eval on every pull request and fails the build if faithfulness drops below threshold, so a well-meaning change to chunking or a prompt can't silently degrade quality.

\---

## Observability \& monitoring

Instrumented end-to-end, so a request isn't a black box — you can see what was retrieved, how it was reordered, the exact prompt and response, and what it cost. Tracing is **env-gated and off by default** (the deployed BYO-key demo runs untraced); the operational metrics that gate CI need no hosted service.

**Per-request tracing (Langfuse).** Every stage is a timed span: query rewrite → retrieval (which chunks, with scores) → rerank (the reordering, and the `top\_n` result-cap as a standing metric) → citation-enforced generation, plus the LLM-judge membership decisions in analytical mode. The four model calls carry token counts, so cost rolls up automatically.

!\[Langfuse trace of an analytical query](docs/langfuse-trace.png)

<!-- Screenshot: the nested `analytical\_query` trace tree — rewrite, retrieve,
     and the judge batches — with per-span latency and token/cost. Replace the

**Operational metrics (SRE view).** Every request writes one line to a local sink; a reporter computes the signals averages hide. On a representative batch:

|Metric|Value|Note|
|-|-|-|
|Latency **p50 / p95**|**21s / 60s**|the gap is the two modes — single-doc vs. analytical judge fan-out; the 27s *mean* hides it|
|Citation coverage|**100%**|of answered requests (declines/errors excluded from the denominator)|
|Failure rate|**0%**|errors are recorded and re-raised, never swallowed|
|Decline rate|**17%**|the system refuses to answer when the corpus can't ground it — correct behavior, not a miss|
|LLM calls / request|**1 → 11**|single-doc ≈ 1; analytical fans out to query-rewrite + guidance + judge batches|

Latency here is dominated by trial-tier API rate-limit pacing, not compute — a production key removes most of the tail.

```bash
python -m rag\_revops.metrics\_report              # p50/p95, coverage, failure/decline rate
python -m rag\_revops.metrics\_report --since-min 60 --json
```

**Regression gating.** CI runs two gates on every pull request: the faithfulness gate above, and an operational gate that reads the metrics the eval run itself produced (no extra API spend). The build fails if **citation coverage drops below 80%**, **p95 latency exceeds 90s**, or the **failure rate exceeds 5%** — so a change that quietly degrades grounding or blows up latency can't merge. Prompts live in versioned config (`config/settings.yaml`) and the config version is stamped on every trace and metric, so a prompt edit is attributable to any metric shift it causes. Full detail in [`OBSERVABILITY.md`](OBSERVABILITY.md).

\---

## Data provenance (zero proprietary data)

Every document is **public**. Nothing from any employer is used. The corpus and evaluation set are built from **CUAD** (CC BY 4.0). Handling licensing and provenance as a first-class concern — rather than quietly ingesting whatever's available — is itself part of the deliverable: a contract-analysis tool that can't say where its knowledge came from is a governance liability. See [`docs/DATA\_PROVENANCE.md`](docs/DATA_PROVENANCE.md).

\---

## Run it locally

```bash
python -m venv .venv \&\& source .venv/bin/activate    # Windows: .\\.venv\\Scripts\\Activate.ps1
pip install -e ".\[dev]"
cp .env.example .env          # add ANTHROPIC\_API\_KEY + COHERE\_API\_KEY

# Build the corpus + index from the CUAD file
python scripts/extract\_cuad.py --input path/to/CUADv1.json --contracts-out data/raw/contracts --eval-out eval/eval\_seed.jsonl --max-contracts 100
python -m rag\_revops.ingest --source data/raw/contracts

# CLI
python -m rag\_revops.query "How is liability capped?"
python -m rag\_revops.query\_analytical "Which contracts allow termination for convenience?"

# Or the full UI (both modes)
streamlit run app.py
```

## Evaluation \& tests

```bash
pytest                                          # unit tests (retrieval, fusion, judge logic)
python -m eval.build\_golden --seed eval/eval\_seed.jsonl --out eval/golden.jsonl
python -m eval.run\_eval --golden eval/golden.jsonl --limit 15          # faithfulness
python -m eval.build\_analytical\_golden --seed eval/eval\_seed.jsonl --out eval/analytical\_golden.jsonl
python -m eval.run\_analytical\_eval --golden eval/analytical\_golden.jsonl --limit 3   # set-based P/R/F1
```

## Repo layout

```
src/rag\_revops/
  loaders.py chunking.py embeddings.py vectorstore.py   # ingestion + storage
  bm25.py hybrid.py rerank.py retrieval.py              # retrieval
  query\_rewrite.py                                       # jargon -> contract language
  graph.py generation.py query.py                        # single-doc pipeline (LangGraph)
  analytical.py analytical\_generation.py query\_analytical.py  # cross-corpus + LLM judge
eval/
  build\_golden.py run\_eval.py judge.py                   # faithfulness eval
  build\_analytical\_golden.py run\_analytical\_eval.py      # set-based eval
config/settings.yaml                                     # versioned prompts + tunables
app.py                                                   # Streamlit UI (both modes)
.github/workflows/ci.yml                                 # tests + faithfulness gate
```

## License

Code: MIT. Data: CUAD is CC BY 4.0 — see `docs/DATA\_PROVENANCE.md`.

