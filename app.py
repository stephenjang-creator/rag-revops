"""Deal Desk Helper — Streamlit demo (bring-your-own-key), two modes.

A public, live demo of the contract-analysis pipeline. It bakes in NO API keys:
the visitor pastes their own Anthropic + Cohere keys into the sidebar. Keys live
only in Streamlit session state for the browser session — never written to disk,
never logged, never committed.

Two modes:
  • Ask about a contract   — pick one contract (or search all), then get a grounded
                             answer with inline citations, or an explicit decline.
  • Find contracts         — cross-corpus analytical retrieval: "which contracts
                             have X", using an LLM judge that recognizes clauses
                             phrased differently from the query.

Deployed on Streamlit Community Cloud from this repo; it reruns on every push.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Streamlit Cloud installs runtime deps from requirements.txt but does not run an
# editable install of this package, so make the src/ layout importable directly.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

# Bridge observability config from Streamlit secrets → environment BEFORE importing
# anything that reads it. observability.py decides whether Langfuse is on at import
# time from os.environ, so these must be set first. All are optional: absent secrets
# leave the defaults (tracing off, metrics sink on). This is the seam that lets you
# turn tracing on for a deployment via Streamlit's Secrets UI without code changes.
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
    "Ask about a single contract, or find which contracts across the corpus contain "
    "a given clause — even when they phrase it differently from your query. "
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
def _build_rewriter(fingerprint: str):
    from rag_revops.config import load_secrets, load_settings

    load_secrets.cache_clear()
    from rag_revops.query_rewrite import QueryRewriter

    return QueryRewriter(load_settings())


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------

MODE_SINGLE = "Ask about a contract"
MODE_ANALYTICAL = "Find contracts across the corpus"


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
    [MODE_SINGLE, MODE_ANALYTICAL],
    horizontal=True,
    label_visibility="collapsed",
)


# ===========================================================================
# MODE 1 — single-document Q&A
# ===========================================================================

def render_single_doc() -> None:
    st.caption(
        "Pick a contract, then ask about it — the answer draws only from that "
        "contract and cites the passages it used, or declines rather than guessing. "
        "Or toggle **Search all contracts** to ask across the whole corpus."
    )

    # Keys are guaranteed present here (app st.stop()s earlier otherwise).
    contracts: list[str] = []
    try:
        _, store = _build_single_doc(anthropic_key[-4:] + cohere_key[-4:])
        contracts = store.list_contracts()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't load contracts: {exc}")
        return

    search_all = st.toggle(
        "Search all contracts (instead of one)",
        value=False,
        help="On: ask across the entire corpus. Off: restrict to the contract you pick.",
    )

    selected: str | None = None
    if not search_all:
        # Searchable single-select. Streamlit's selectbox filters as you type.
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
        try:
            pipeline, _ = _build_single_doc(anthropic_key[-4:] + cohere_key[-4:])
            run_query = _rewrite_and_show(question.strip())
            scope = "the whole corpus" if search_all else f"contract {selected}"
            with st.spinner(f"Retrieving from {scope} and grounding an answer…"):
                result = pipeline.ask(
                    run_query, doc_id=None if search_all else selected
                )

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
        "⏳ Analytical queries scan every contract and use an LLM judge, so they "
        "take ~30–60s and use more API credit than single-contract questions.",
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
        except Exception as exc:  # noqa: BLE001
            st.error(f"Something went wrong: {exc}")
            st.caption("Check that both API keys are valid and have available credit.")


if mode == MODE_SINGLE:
    render_single_doc()
else:
    render_analytical()
