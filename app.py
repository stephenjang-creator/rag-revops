"""Deal Desk Helper — local Streamlit UI (bring-your-own-key), three modes.

A bring-your-own-key UI over the contract-analysis pipeline. It bakes in NO API
keys: you paste your own Anthropic + Cohere keys into the sidebar. Keys live only
in Streamlit session state for the browser session — never written to disk, never
logged, never committed.

Modes:
  • Ask anything (auto)    — an LLM router reads the question and dispatches to one
                             of the three skills below; shows which it chose, and
                             declines to guess when the intent is ambiguous.
  • Draft clause language  — the reranker pulls the most on-point real passages,
                             then Claude drafts suggested language for a NEW
                             contract, grounded in and citing those passages. A
                             citation-free copy is offered for pasting into an
                             order form. This is the default mode.
  • Find contracts with clauses — cross-corpus analytical retrieval: "which
                             contracts have X", using an LLM judge that recognizes
                             clauses phrased differently from the query.
  • Ask about a contract   — pick one contract, then get a grounded answer with
                             inline citations, or an explicit decline.

Run it locally with `streamlit run app.py`. The hosted demo is the FastAPI `web/`
service deployed on Render; this Streamlit app is the zero-setup local alternative.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# `streamlit run` doesn't do an editable install of this package, so make the
# src/ layout importable directly.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

# Bridge observability config from Streamlit secrets → environment BEFORE importing
# anything that reads it. observability.py decides whether Langfuse is on at import
# time from os.environ, so these must be set first. All are optional: absent secrets
# leave the defaults (tracing off, metrics sink on). This is the seam that lets you
# turn tracing on locally via Streamlit secrets without code changes.
for _key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST",
             "RAG_METRICS_PATH", "RAG_CONFIG_VERSION"):
    try:
        _val = st.secrets.get(_key)
    except Exception:
        _val = None
    if _val:
        os.environ[_key] = str(_val)

from rag_revops.admin_panel import render_admin_panel

st.set_page_config(
    page_title="Deal Desk Helper — contract analysis", page_icon="📄", layout="wide"
)

# --- Operator aesthetic (matches the GTM portfolio site) --------------------
st.markdown(
    """
    <style>
      /* Tighten headings, monospace accent, operator feel */
      h1, h2, h3 { letter-spacing: -0.5px; font-weight: 700; }
      .dd-eyebrow {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.72rem; letter-spacing: 2px; text-transform: uppercase;
        color: #4ade80; margin-bottom: 0.2rem;
      }
      .dd-tag {
        display: inline-block; font-family: ui-monospace, monospace;
        font-size: 0.65rem; letter-spacing: 1px; text-transform: uppercase;
        padding: 2px 8px; border-radius: 3px; border: 1px solid #4ade80;
        color: #4ade80; margin-left: 8px; vertical-align: middle;
      }
      .dd-rule { height: 1px; background: #2a2f3b; margin: 0.75rem 0 1.25rem; }
      /* Primary buttons: operator green */
      .stButton > button[kind="primary"] {
        background: #4ade80; color: #0e1117; border: none; font-weight: 700;
      }
      /* Expanders / panels: subtle border */
      .streamlit-expanderHeader { font-family: ui-monospace, monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="dd-eyebrow">Operator tool · Built in Python</div>',
    unsafe_allow_html=True,
)
st.title("📄 Deal Desk Helper")
st.caption(
    "Citation-grounded analysis over **public** contract data (CUAD, CC BY 4.0). "
    "Ask in plain English and let it route automatically, or pick a mode: draft new "
    "clause language grounded in real contracts, find which contracts across the "
    "corpus contain a given clause — even when they phrase it differently — or ask "
    "about a single contract. "
    "**Human-in-the-loop by design: it cites its sources, or declines rather than guessing.**"
)
st.markdown('<div class="dd-rule"></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("🔑 Your API keys")
    st.markdown(
        "This demo runs on **your** keys and is **not stored anywhere** — they stay "
        "in this browser session only. Nothing is logged or committed."
    )
    anthropic_key = st.text_input("Anthropic API key", type="password", key="anthropic_key")
    cohere_key = st.text_input("Cohere API key", type="password", key="cohere_key")

    st.divider()
    st.markdown(
        "**Data:** public only (CUAD, CC BY 4.0). No proprietary data. "
        "See the repo's `DATA_PROVENANCE.md`."
    )

# Admin-only operational metrics (password-gated via st.secrets["admin_password"]).
# Invisible to reviewers who don't have the password; renders nothing if no
# password is configured. Metrics still collect silently either way.
render_admin_panel()

keys_ready = bool(anthropic_key and cohere_key)
if anthropic_key:
    os.environ["ANTHROPIC_API_KEY"] = anthropic_key
if cohere_key:
    os.environ["COHERE_API_KEY"] = cohere_key


# ---------------------------------------------------------------------------
# Cached builders. Keyed on a fingerprint of the keys so a new key rebuilds.
# Imports are lazy so a missing key doesn't error at module import time.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _build_single_doc(fingerprint: str):
    from rag_revops.config import load_secrets, load_settings

    load_secrets.cache_clear()
    from rag_revops.graph import RagPipeline

    pipeline = RagPipeline(load_settings())
    # Expose the store so the UI can list contracts for the picker.
    return pipeline, pipeline.store


@st.cache_resource(show_spinner=False)
def _build_analytical(fingerprint: str):
    from rag_revops.config import load_secrets, load_settings

    load_secrets.cache_clear()
    from rag_revops.analytical import AnalyticalRetriever
    from rag_revops.analytical_generation import AnalyticalGenerator
    from rag_revops.embeddings import CohereEmbedder
    from rag_revops.vectorstore import ChromaStore

    settings = load_settings()
    store = ChromaStore(settings.vectorstore)
    embedder = CohereEmbedder(settings.embeddings)
    retriever = AnalyticalRetriever(settings, store, embedder)
    generator = AnalyticalGenerator(settings)
    return retriever, generator


@st.cache_resource(show_spinner=False)
def _build_clause(fingerprint: str):
    from rag_revops.config import load_secrets, load_settings

    load_secrets.cache_clear()
    from rag_revops.clause_drafting import ClauseDrafter
    from rag_revops.clause_finder import ClauseFinder
    from rag_revops.embeddings import CohereEmbedder
    from rag_revops.vectorstore import ChromaStore

    settings = load_settings()
    store = ChromaStore(settings.vectorstore)
    embedder = CohereEmbedder(settings.embeddings)
    finder = ClauseFinder(settings, store, embedder)
    drafter = ClauseDrafter(settings)
    return finder, drafter


@st.cache_resource(show_spinner=False)
def _build_router(fingerprint: str):
    from rag_revops.config import load_secrets, load_settings

    load_secrets.cache_clear()
    from rag_revops.router import QueryRouter

    return QueryRouter(load_settings())


@st.cache_resource(show_spinner=False)
def _build_rewriter(fingerprint: str):
    from rag_revops.config import load_secrets, load_settings

    load_secrets.cache_clear()
    from rag_revops.query_rewrite import QueryRewriter

    return QueryRewriter(load_settings())


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------

MODE_AUTO = "Ask anything (auto)"
MODE_CLAUSE = "Draft clause language"
MODE_ANALYTICAL = "Find contracts with clauses"
MODE_SINGLE = "Ask about a contract"


def _set_example(input_key: str, text: str) -> None:
    """Button on_click callback: set an input widget's value. Runs before the
    widget is instantiated on the next run, so this assignment is legal (writing
    to a widget's key after it renders raises StreamlitAPIException)."""
    st.session_state[input_key] = text


def _rewrite_and_show(raw_question: str) -> str:
    """Reformulate the user's question into an ideal retrieval query, show the
    interpretation when it changed, and return the query to actually run."""
    rewriter = _build_rewriter(anthropic_key[-4:] + cohere_key[-4:])
    with st.spinner("Interpreting your question…"):
        rq = rewriter.rewrite(raw_question)
    if rq.changed:
        st.info(f"🔎 Interpreted as: **{rq.rewritten}**")
    return rq.rewritten

if not keys_ready:
    st.info("👈 Enter your Anthropic and Cohere API keys in the sidebar to begin.")
    st.stop()

# From here down, keys are present — show the full query interface.
mode = st.radio(
    "Mode",
    [MODE_AUTO, MODE_CLAUSE, MODE_ANALYTICAL, MODE_SINGLE],
    horizontal=True,
    label_visibility="collapsed",
)


# ===========================================================================
# Shared result renderers — used by both the dedicated modes and Auto mode.
# ===========================================================================

def _render_single_result(result) -> None:
    if result.declined:
        st.warning(f"**Declined:** {result.answer}")
        st.caption(
            "The system declined because retrieved passages didn't clearly "
            "support an answer — by design, it does not guess."
        )
    else:
        st.success("Answer")
        st.markdown(result.answer)
        if result.citations:
            st.subheader("Sources")
            for c in result.citations:
                with st.expander(
                    f"[{c.marker}]  {c.source_path}  ·  score {c.score:.3f}"
                ):
                    st.markdown(f"> {c.snippet}…")
                    st.caption(f"chunk id: `{c.chunk_id}`")
        else:
            st.error("Answer produced no inline citations — flagging for review.")


def _render_analytical_result(result) -> None:
    if not result.findings:
        st.warning("No contracts in the corpus match that criterion.")
        st.caption(
            f"Scanned {result.n_candidates} candidate contracts; the judge "
            "confirmed none."
        )
    else:
        st.success(
            f"{len(result.findings)} of {result.n_candidates} contracts match"
        )
        for f in result.findings:
            with st.expander(f"📄 {f.doc_id}", expanded=True):
                st.markdown(f.reason)
                st.caption(
                    f"rerank relevance {f.score:.3f} "
                    "(low is expected — the judge, not the reranker, "
                    "confirmed this match by meaning)"
                )


def _render_clause_result(found, result) -> None:
    if not found.options:
        st.warning("No passages in the corpus match that clause.")
        st.caption(
            f"Reranked {found.n_considered} candidate passages; none scored "
            "above the relevance floor, so there's nothing to draft from."
        )
    elif result.declined:
        # Passages were retrieved, but the model judged they can't ground a
        # reusable clause. Show why — but NO copy block: there's no clause
        # language to paste into an order form.
        from rag_revops.clause_drafting import decline_reason

        st.warning(
            "Couldn't draft a reusable clause grounded in the corpus for that "
            "request."
        )
        reason = decline_reason(result.draft)
        if reason:
            st.markdown(reason)
        st.caption(
            "The corpus didn't contain language that generalizes to this clause, "
            "so nothing is offered to copy rather than inventing it."
        )
        st.subheader("Passages considered")
        for i, o in enumerate(found.options, start=1):
            with st.expander(f"[{i}]  {o.doc_id}  ·  relevance {o.score:.3f}"):
                st.markdown(f"> {o.text}")
                st.caption(f"source: `{o.source_path}`  ·  chunk `{o.chunk_id}`")
    else:
        st.success("Suggested clause language")
        st.markdown(result.draft)
        st.caption(
            "Drafted from real contract language and grounded in the cited "
            "sources below — adapt before use; not legal advice."
        )

        # Clean, citation-free version to drop straight into an order form.
        # st.code renders a one-click copy button. Only shown on a real draft.
        from rag_revops.clause_drafting import strip_citations

        st.markdown("**Copy for an order form** (citations removed):")
        st.code(strip_citations(result.draft), language=None)

        if result.citations:
            st.subheader("Sources")
            # marker i maps to found.options[i-1] (the numbering the model saw)
            for i, o in enumerate(found.options, start=1):
                if o not in result.citations:
                    continue
                with st.expander(f"[{i}]  {o.doc_id}  ·  relevance {o.score:.3f}"):
                    st.markdown(f"> {o.text}")
                    st.caption(
                        f"source: `{o.source_path}`  ·  chunk `{o.chunk_id}`"
                    )
        else:
            st.error("Draft produced no inline citations — flagging for review.")


# ===========================================================================
# MODE 1 — single-document Q&A
# ===========================================================================

def render_single_doc() -> None:
    st.caption(
        "Pick a contract, then ask about it — the answer draws only from that "
        "contract and cites the passages it used, or declines rather than guessing. "
        "To ask across the whole corpus, use **Find contracts with clauses**."
    )

    # Keys are guaranteed present here (app st.stop()s earlier otherwise).
    contracts: list[str] = []
    try:
        _, store = _build_single_doc(anthropic_key[-4:] + cohere_key[-4:])
        contracts = store.list_contracts()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't load contracts: {exc}")
        return

    # Searchable single-select. Streamlit's selectbox filters as you type.
    selected: str | None = None
    if contracts:
        selected = st.selectbox(
            "Contract",
            options=contracts,
            index=0,
            help="Type to filter the list.",
        )
    else:
        st.caption("Enter your keys to load the contract list.")

    examples = [
        "Can either party terminate for convenience, and with how much notice?",
        "How is liability capped, and what damages are excluded?",
        "What are the confidentiality obligations?",
        "Which state's law governs this agreement?",
    ]

    cols = st.columns([4, 1])
    with cols[0]:
        question = st.text_input(
            "Question",
            label_visibility="collapsed",
            placeholder="e.g. Can either party terminate for convenience?",
            key="sd_input",
        )
    with cols[1]:
        ask = st.button("Ask", type="primary", use_container_width=True)

    ex_cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        # Use on_click: the callback runs BEFORE widgets are instantiated on the
        # next run, so setting the input's key here is legal (writing to it after
        # the widget renders raises StreamlitAPIException).
        ex_cols[i].button(
            f"Example {i + 1}",
            help=ex,
            use_container_width=True,
            key=f"sd_ex{i}",
            on_click=_set_example,
            args=("sd_input", ex),
        )

    if ask and question.strip():
        if not selected:
            st.warning("Pick a contract first.")
            return
        try:
            pipeline, _ = _build_single_doc(anthropic_key[-4:] + cohere_key[-4:])
            run_query = _rewrite_and_show(question.strip())
            with st.spinner(f"Retrieving from contract {selected} and grounding an answer…"):
                result = pipeline.ask(run_query, doc_id=selected)
            _render_single_result(result)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Something went wrong: {exc}")
            st.caption("Check that both API keys are valid and have available credit.")


# ===========================================================================
# MODE 2 — cross-corpus analytical
# ===========================================================================

def render_analytical() -> None:
    st.caption(
        "Ask **which contracts** across the whole corpus contain a clause. An LLM "
        "judge recognizes clauses that mean the same thing but are worded "
        "differently — e.g. it finds 'terminate without cause' when you ask about "
        "'termination for convenience'."
    )
    st.info(
        "⏳ This judges **every** contract in the corpus with an LLM judge (no cap), "
        "so nothing is missed — but it takes ~30–60s and uses more API credit than "
        "single-contract questions, scaling with the size of the corpus.",
        icon="⏳",
    )

    examples = [
        "Which contracts allow termination for convenience?",
        "Which contracts cap liability?",
        "Which contracts grant audit rights?",
        "Which contracts restrict assignment?",
    ]

    cols = st.columns([4, 1])
    with cols[0]:
        question = st.text_input(
            "Question",
            label_visibility="collapsed",
            placeholder="e.g. Which contracts allow termination for convenience?",
            key="an_input",
        )
    with cols[1]:
        find = st.button(
            "Find", type="primary", use_container_width=True
        )

    ex_cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        ex_cols[i].button(
            f"Example {i + 1}",
            help=ex,
            use_container_width=True,
            key=f"an_ex{i}",
            on_click=_set_example,
            args=("an_input", ex),
        )

    if find and question.strip():
        try:
            retriever, generator = _build_analytical(anthropic_key[-4:] + cohere_key[-4:])
            run_query = _rewrite_and_show(question.strip())
            with st.spinner("Scanning the corpus and judging each contract… (~30–60s)"):
                matches = retriever.retrieve(run_query)
                result = generator.generate(run_query, matches)
            _render_analytical_result(result)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Something went wrong: {exc}")
            st.caption("Check that both API keys are valid and have available credit.")


# ===========================================================================
# MODE 3 — clause language finder (reranker-driven precision retrieval)
# ===========================================================================

def render_clause() -> None:
    st.caption(
        "Ask for a clause and get **suggested language for a new contract** — Claude "
        "drafts it, grounded in real passages the cross-encoder reranker pulls from "
        "the corpus, and cites the source contracts with inline `[n]` markers. The "
        "reranker finds the most on-point examples; the draft turns them into "
        "reusable language you can adapt. It declines rather than inventing a clause "
        "the corpus can't support."
    )

    examples = [
        "termination for convenience",
        "mutual confidentiality / NDA",
        "limitation of liability",
        "indemnification",
    ]

    cols = st.columns([4, 1])
    with cols[0]:
        question = st.text_input(
            "Clause",
            label_visibility="collapsed",
            placeholder="e.g. termination for convenience",
            key="cl_input",
        )
    with cols[1]:
        find = st.button("Draft clause", type="primary", use_container_width=True)

    ex_cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        ex_cols[i].button(
            f"Example {i + 1}",
            help=ex,
            use_container_width=True,
            key=f"cl_ex{i}",
            on_click=_set_example,
            args=("cl_input", ex),
        )

    if find and question.strip():
        try:
            finder, drafter = _build_clause(anthropic_key[-4:] + cohere_key[-4:])
            run_query = _rewrite_and_show(question.strip())
            with st.spinner("Retrieving example language and drafting a clause…"):
                found = finder.find(run_query)
                result = drafter.draft(run_query, found.options)
            _render_clause_result(found, result)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Something went wrong: {exc}")
            st.caption("Check that both API keys are valid and have available credit.")


# ===========================================================================
# MODE 0 — Auto: route the question to the right skill
# ===========================================================================

_SKILL_LABELS = {
    "ask_one_contract": "Ask about a contract",
    "find_contracts": "Find contracts with clauses",
    "draft_clause": "Draft clause language",
}


def render_auto() -> None:
    from rag_revops.router import (
        SKILL_ASK_ONE,
        SKILL_DRAFT,
        SKILL_FIND_MANY,
        resolve_contract,
    )

    st.caption(
        "Ask in plain English — the router reads your question and picks the right "
        "skill: answer about **one** contract, **find** which contracts have a "
        "clause, or **draft** new clause language for an order form. It shows you "
        "which it chose, and you can always switch to a specific mode above."
    )

    examples = [
        "What does the Pizza Fusion franchise agreement say about termination?",
        "Which contracts allow termination for convenience?",
        "Draft a mutual NDA clause for an order form",
    ]

    cols = st.columns([4, 1])
    with cols[0]:
        question = st.text_input(
            "Question",
            label_visibility="collapsed",
            placeholder="e.g. Which contracts cap liability?",
            key="auto_input",
        )
    with cols[1]:
        ask = st.button("Ask", type="primary", use_container_width=True)

    ex_cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        ex_cols[i].button(
            f"Example {i + 1}",
            help=ex,
            use_container_width=True,
            key=f"auto_ex{i}",
            on_click=_set_example,
            args=("auto_input", ex),
        )

    if not (ask and question.strip()):
        return

    fp = anthropic_key[-4:] + cohere_key[-4:]
    try:
        router = _build_router(fp)
        with st.spinner("Routing your question to the right skill…"):
            decision = router.route(question.strip())

        label = _SKILL_LABELS.get(decision.skill, decision.skill)
        st.info(f"🧭 Routed to **{label}** — {decision.reason}")

        if decision.needs_clarification:
            st.warning(
                "This could be interpreted a few ways. Rephrase, or pick a specific "
                "mode above (e.g. ask about one contract vs. search across all)."
            )
            return

        run_query = _rewrite_and_show(question.strip())

        if decision.skill == SKILL_ASK_ONE:
            pipeline, store = _build_single_doc(fp)
            doc_id = resolve_contract(decision.contract_hint, store.list_contracts())
            if not doc_id:
                st.warning(
                    "I couldn't tell which contract you meant. Switch to **Ask about "
                    "a contract** and pick it from the list."
                )
                return
            with st.spinner(f"Retrieving from contract {doc_id} and grounding an answer…"):
                result = pipeline.ask(run_query, doc_id=doc_id)
            _render_single_result(result)

        elif decision.skill == SKILL_FIND_MANY:
            retriever, generator = _build_analytical(fp)
            with st.spinner("Scanning the corpus and judging each contract… (~30–60s)"):
                matches = retriever.retrieve(run_query)
                result = generator.generate(run_query, matches)
            _render_analytical_result(result)

        elif decision.skill == SKILL_DRAFT:
            finder, drafter = _build_clause(fp)
            with st.spinner("Retrieving example language and drafting a clause…"):
                found = finder.find(run_query)
                result = drafter.draft(run_query, found.options)
            _render_clause_result(found, result)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Something went wrong: {exc}")
        st.caption("Check that both API keys are valid and have available credit.")


if mode == MODE_AUTO:
    render_auto()
elif mode == MODE_CLAUSE:
    render_clause()
elif mode == MODE_ANALYTICAL:
    render_analytical()
else:
    render_single_doc()
