"""PolicyLens — Streamlit demo (bring-your-own-key).

A public, live demo that runs the Phase 1 RAG pipeline. It bakes in NO API keys:
the visitor pastes their own Anthropic + Cohere keys into the sidebar. Keys live
only in Streamlit session state for the duration of the browser session — they are
never written to disk, never logged, and never committed.

Reviewers without keys still see the full UI, the corpus, the retrieved passages,
and the citation / decline behavior described in the README.

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

st.set_page_config(page_title="PolicyLens — grounded RevOps Q&A", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Keys are read from session state and pushed into the environment so the
# existing config.load_secrets() (which reads env vars) works unchanged.
# We import the pipeline lazily, AFTER keys are set, so construction doesn't
# fail on a missing-key error before the user has entered anything.
# ---------------------------------------------------------------------------

st.title("📄 PolicyLens")
st.caption(
    "Grounded, citation-enforced Q&A over **public** revenue-operations documents "
    "— SaaS agreements, SLAs, and policy language. Answers cite their source, or "
    "the system declines rather than guessing."
)

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
        "**Data:** public only (CUAD, CC BY 4.0 + published SaaS agreements). "
        "No proprietary data. See the repo's `DATA_PROVENANCE.md`."
    )
    st.markdown(
        "**How it works:** retrieve top passages → generate an answer grounded "
        "only in those passages, with inline `[n]` citations → decline if support "
        "is weak."
    )

keys_ready = bool(anthropic_key and cohere_key)

# Push keys into the environment for the pipeline's secrets loader.
if anthropic_key:
    os.environ["ANTHROPIC_API_KEY"] = anthropic_key
if cohere_key:
    os.environ["COHERE_API_KEY"] = cohere_key


@st.cache_resource(show_spinner=False)
def _build_pipeline(anthropic_fingerprint: str, cohere_fingerprint: str):
    """Construct the pipeline once per distinct key pair.

    The fingerprints (last 4 chars) are only used as a cache key so a new key
    rebuilds the cached resource; the full keys are read from the environment.
    """
    # Import here so a missing key doesn't blow up at module import time.
    from rag_revops.config import load_secrets, load_settings

    load_secrets.cache_clear()  # pick up freshly set env vars
    from rag_revops.graph import RagPipeline

    return RagPipeline(load_settings())


EXAMPLES = [
    "Can a customer terminate the agreement for convenience, and with how much notice?",
    "What are the provider's obligations to return or delete customer data on termination?",
    "How is liability capped, and what types of damages are excluded?",
    "What happens under a force majeure event?",
    "Who owns intellectual property in deliverables created for the customer?",
]

st.subheader("Ask a question")
cols = st.columns([4, 1])
with cols[0]:
    question = st.text_input(
        "Question",
        value=st.session_state.get("question", ""),
        label_visibility="collapsed",
        placeholder="e.g. Can a customer terminate for convenience?",
    )
with cols[1]:
    ask = st.button("Ask", type="primary", use_container_width=True, disabled=not keys_ready)

st.caption("Try an example:")
ex_cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    if ex_cols[i].button(f"Example {i + 1}", help=ex, use_container_width=True):
        st.session_state["question"] = ex
        st.rerun()

if not keys_ready:
    st.info("👈 Enter your Anthropic and Cohere API keys in the sidebar to run a live query.")

if ask and keys_ready and question.strip():
    try:
        pipeline = _build_pipeline(anthropic_key[-4:], cohere_key[-4:])
        with st.spinner("Retrieving passages and grounding an answer…"):
            result = pipeline.ask(question.strip())

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
                    with st.expander(f"[{c.marker}]  {c.source_path}  ·  score {c.score:.3f}"):
                        st.markdown(f"> {c.snippet}…")
                        st.caption(f"chunk id: `{c.chunk_id}`")
            else:
                st.error("Answer produced no inline citations — flagging for review.")
    except Exception as exc:  # noqa: BLE001 - surface errors to the demo user
        st.error(f"Something went wrong: {exc}")
        st.caption("Check that both API keys are valid and have available credit.")
