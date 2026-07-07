"""Fire a batch of queries across both pipeline modes to populate metrics.

Runs single-doc and analytical queries so the Langfuse dashboards and the local
metrics sink have a real distribution to work with (percentiles need spread).
Includes a couple of deliberately off-corpus questions to exercise the decline
path, so citation-coverage and decline-rate aren't trivially 100%/0%.

    python smoke_queries.py
    python -m rag_revops.metrics_report      # then read the numbers
"""

from __future__ import annotations

import time

from rag_revops.graph import RagPipeline
from rag_revops.analytical import AnalyticalRetriever
from rag_revops.analytical_generation import AnalyticalGenerator
from rag_revops.config import load_settings
from rag_revops.embeddings import CohereEmbedder
from rag_revops.vectorstore import ChromaStore

# Single-doc questions (some answerable, one likely to decline).
SINGLE_DOC = [
    "How is liability capped in this agreement?",
    "What are the payment terms?",
    "Can either party terminate for convenience?",
    "What governs confidentiality obligations?",
    "Is there an indemnification clause?",
    "What is the capital of France?",          # off-corpus -> should decline
]

# Analytical (cross-corpus) questions.
ANALYTICAL = [
    "Which contracts allow termination for convenience?",
    "Which contracts have uncapped liability?",
    "Which contracts include a most-favored-nation clause?",
    "Which contracts require 30-day payment terms?",
]


def main() -> None:
    settings = load_settings()

    print("=" * 60)
    print("SINGLE-DOC QUERIES")
    print("=" * 60)
    pipeline = RagPipeline(settings)
    for i, q in enumerate(SINGLE_DOC, 1):
        t0 = time.perf_counter()
        result = pipeline.ask(q)   # searches all contracts
        dt = time.perf_counter() - t0
        tag = "DECLINED" if result.declined else f"{len(result.citations)} citations"
        print(f"[{i}/{len(SINGLE_DOC)}] {dt:5.1f}s  {tag:14s} {q[:50]}")

    print("\n" + "=" * 60)
    print("ANALYTICAL QUERIES  (slower — many judge calls each)")
    print("=" * 60)
    store = ChromaStore(settings.vectorstore)
    embedder = CohereEmbedder(settings.embeddings)
    retriever = AnalyticalRetriever(settings, store, embedder)
    generator = AnalyticalGenerator(settings)
    for i, q in enumerate(ANALYTICAL, 1):
        t0 = time.perf_counter()
        matches = retriever.retrieve(q)
        result = generator.generate(q, matches)
        dt = time.perf_counter() - t0
        print(f"[{i}/{len(ANALYTICAL)}] {dt:5.1f}s  {len(result.findings):2d} found   {q[:50]}")

    print("\nDone. Now run:  python -m rag_revops.metrics_report")


if __name__ == "__main__":
    main()
