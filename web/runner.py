"""Live pipeline → event stream adapter.

Runs the existing `rag_revops` pipeline synchronously and emits the same events
the recorded demo uses, via an ``emit(event)`` callback. Mirrors the orchestration
in ``app.py`` (route → rewrite → one of the three skills). The server owns keys,
concurrency, and the sync→async bridge; this module is pure pipeline + emit.

Heavy imports are lazy so importing the web app (for the demo/health endpoints,
or tests) doesn't require a built Chroma index.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from . import events as E

Emit = Callable[[dict[str, Any]], None]

_SKILL_LABELS = {
    "ask_one_contract": "Ask about a contract",
    "find_contracts": "Find contracts with clauses",
    "draft_clause": "Draft clause language",
}
_MODE_TO_SKILL = {
    "draft": "draft_clause",
    "find": "find_contracts",
    "single": "ask_one_contract",
}


def run_live(question: str, mode: str, doc_id: str | None, emit: Emit) -> None:
    """Drive the pipeline for one question, emitting events as it goes. Runs in a
    worker thread; `emit` is a thread-safe hand-off to the SSE generator."""
    from rag_revops.config import load_settings
    from rag_revops.query_rewrite import QueryRewriter
    from rag_revops.router import QueryRouter

    settings = load_settings()
    t0 = time.perf_counter()

    def ms() -> str:
        return f"{time.perf_counter() - t0:.1f}s"

    # 1. Translate (query rewrite) --------------------------------------------
    emit(E.stage("Translate", "Expanding shorthand into contract language", ms(), "running"))
    rq = QueryRewriter(settings).rewrite(question)
    emit(E.rewrite(question, rq.rewritten, rq.changed))
    emit(E.stage("Translate", "Shorthand expanded into contract language", ms(), "done"))
    run_query = rq.rewritten

    # 2. Route ----------------------------------------------------------------
    emit(E.stage("Route", "Choosing the right skill", ms(), "running"))
    contract_hint: str | None = None
    if mode == "auto":
        decision = QueryRouter(settings).route(question)
        skill = decision.skill
        contract_hint = decision.contract_hint
        label = _SKILL_LABELS.get(skill, skill)
        emit(E.route(skill, label, decision.reason, decision.needs_clarification))
        if decision.needs_clarification:
            emit(E.stage("Route", "Ambiguous — needs a more specific question", ms(), "done"))
            emit(E.decline("I'm not sure which skill you want here.", decision.reason))
            emit(E.done(total_ms=(time.perf_counter() - t0) * 1000))
            return
    else:
        skill = _MODE_TO_SKILL.get(mode, "ask_one_contract")
        label = _SKILL_LABELS[skill]
        emit(E.route(skill, label, f"You selected {label}.", False))
    emit(E.stage("Route", f"Sent to: {label.lower()}", ms(), "done"))

    # 3. Dispatch to the chosen skill -----------------------------------------
    if skill == "draft_clause":
        _run_draft(run_query, emit, settings, ms)
    elif skill == "find_contracts":
        _run_find(run_query, emit, settings, ms)
    else:
        _run_single(run_query, doc_id, contract_hint, emit, settings, ms)

    emit(E.done(total_ms=(time.perf_counter() - t0) * 1000))


def _run_draft(q: str, emit: Emit, settings, ms) -> None:
    from rag_revops.clause_drafting import ClauseDrafter, decline_reason
    from rag_revops.clause_finder import ClauseFinder
    from rag_revops.embeddings import CohereEmbedder
    from rag_revops.vectorstore import ChromaStore

    from .clause_format import to_html, to_plain

    store = ChromaStore(settings.vectorstore)
    embedder = CohereEmbedder(settings.embeddings)
    finder = ClauseFinder(settings, store, embedder)
    drafter = ClauseDrafter(settings)

    emit(E.stage("Retrieve", "Pulling candidate clauses, reranking for precision", ms(), "running"))
    found = finder.find(q)
    top = found.options[0].score if found.options else None
    emit(E.retrieval(found.n_considered, len(found.options), top))
    emit(E.stage("Rerank", f"{len(found.options)} example passages kept", ms(), "done"))

    emit(E.stage("Draft", "Drafting clause language grounded in those passages", ms(), "running"))
    result = drafter.draft(q, found.options)
    emit(E.stage("Draft", "Clause drafted with inline citations", ms(), "done"))

    if not found.options or result.declined:
        reason = decline_reason(result.draft) if found.options else ""
        emit(E.decline("Couldn't ground a reusable clause from the corpus.",
                       reason or "The corpus had no supporting language for this clause."))
        return

    sources = [{"doc_id": o.doc_id, "score": f"{o.score:.3f}"} for o in result.citations]
    emit(E.answer(
        "draft",
        heading="Suggested clause language",
        subheading="",
        body_html=to_html(result.draft),    # formatted; [n] rendered as <sup>
        body_plain=to_plain(result.draft),  # clean, order-form-ready copy
        note="",
        disclaimer="Drafted from real contract language and grounded in the sources "
                   "below — adapt before use, not legal advice.",
        sources=sources,
    ))


def _run_find(q: str, emit: Emit, settings, ms) -> None:
    from rag_revops.analytical import AnalyticalRetriever
    from rag_revops.analytical_generation import AnalyticalGenerator
    from rag_revops.embeddings import CohereEmbedder
    from rag_revops.vectorstore import ChromaStore

    store = ChromaStore(settings.vectorstore)
    embedder = CohereEmbedder(settings.embeddings)
    retriever = AnalyticalRetriever(settings, store, embedder)
    generator = AnalyticalGenerator(settings)

    emit(E.stage("Retrieve", "Every contract in the corpus pulled in — no result cap",
                 ms(), "running"))
    matches = retriever.retrieve(q)
    emit(E.stage("Rerank", "Ordering only; paraphrased clauses score ~0.00", ms(), "done"))

    emit(E.stage("Judge", f"Reading {len(matches)} contracts for legal meaning…", ms(), "running"))
    result = generator.generate(q, matches)
    for f in result.findings:
        emit(E.judge(f.doc_id, True, "", f.reason))
    emit(E.stage("Judge", f"{len(result.findings)} of {result.n_candidates} confirmed",
                 ms(), "done"))

    if not result.findings:
        emit(E.decline(
            "No contracts in the corpus match that criterion.",
            f"Scanned {result.n_candidates} candidate contracts; the judge confirmed none.",
        ))
        return

    findings = [{"doc_id": f.doc_id, "tag": "", "reason": f.reason} for f in result.findings]
    emit(E.answer(
        "find",
        summary=f"{len(result.findings)} of {result.n_candidates} contracts qualify",
        findings=findings,
        footer="Wording varies — a keyword search finds none of these; a judge that "
               "reads for meaning finds them all.",
    ))


def _run_single(q: str, doc_id: str | None, hint: str | None, emit: Emit, settings, ms) -> None:
    from rag_revops.graph import RagPipeline
    from rag_revops.router import resolve_contract

    from .clause_format import to_html, to_plain

    pipeline = RagPipeline(settings)

    if not doc_id:
        emit(E.stage("Resolve", "Matching the named agreement in the corpus", ms(), "running"))
        doc_id = resolve_contract(hint, pipeline.store.list_contracts())
        if not doc_id:
            emit(E.stage("Resolve", "Couldn't identify the contract", ms(), "done"))
            emit(E.decline("I couldn't tell which contract you meant.",
                           "Name the agreement, or use the contract picker in the live app."))
            return
        emit(E.stage("Resolve", "Matched the named agreement", ms(), "done"))

    emit(E.stage("Retrieve", "Passages pulled from that contract only", ms(), "running"))
    result = pipeline.ask(q, doc_id=doc_id)
    emit(E.stage("Check", "Grounding an answer against the retrieved passages", ms(), "done"))

    if result.declined:
        emit(E.decline(
            "Declined — I couldn't find support for that in the documents.",
            "The retrieved passages didn't clearly answer the question, so nothing is offered.",
        ))
        return

    citations = [
        {"marker": c.marker, "doc_id": c.doc_id, "source_path": c.source_path,
         "chunk_id": c.chunk_id, "snippet": c.snippet, "score": f"{c.score:.3f}"}
        for c in result.citations
    ]
    emit(E.answer(
        "single",
        heading="Answer",
        body_html=to_html(result.answer),
        body_plain=to_plain(result.answer),
        doc_id=doc_id,
        citations=citations,
    ))
