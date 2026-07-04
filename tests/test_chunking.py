"""Tests for the token-aware chunker: size bounds and overlap behavior."""

from __future__ import annotations

import tiktoken

from rag_revops.chunking import TokenChunker
from rag_revops.config import ChunkingConfig
from rag_revops.loaders import LoadedDocument

_ENC = tiktoken.get_encoding("cl100k_base")


def _cfg() -> ChunkingConfig:
    return ChunkingConfig(
        target_tokens=120,
        min_tokens=80,
        max_tokens=160,
        overlap_tokens=20,
        tokenizer="cl100k_base",
    )


def _make_doc(n_paragraphs: int, sentences_each: int = 6) -> LoadedDocument:
    sentence = "This clause governs the obligations of the parties under the agreement."
    paras = [" ".join([sentence] * sentences_each) for _ in range(n_paragraphs)]
    return LoadedDocument(
        doc_id="sample",
        source_path="sample.txt",
        doc_type="markdown",
        text="\n\n".join(paras),
    )


def test_chunks_respect_max_tokens():
    cfg = _cfg()
    chunker = TokenChunker(cfg)
    chunks = chunker.chunk_document(_make_doc(10))
    assert chunks, "expected at least one chunk"
    for c in chunks:
        # Allow the small overlap prefix to nudge slightly over max.
        assert c.n_tokens <= cfg.max_tokens + cfg.overlap_tokens


def test_chunk_ids_are_sequential_and_unique():
    chunker = TokenChunker(_cfg())
    chunks = chunker.chunk_document(_make_doc(8))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_overlap_carries_tail_between_chunks():
    chunker = TokenChunker(_cfg())
    chunks = chunker.chunk_document(_make_doc(12))
    assert len(chunks) >= 2, "need multiple chunks to test overlap"
    # The start of chunk n+1 should share tokens with the end of chunk n.
    prev_tail = set(_ENC.encode(chunks[0].text)[-20:])
    next_head = set(_ENC.encode(chunks[1].text)[:20])
    assert prev_tail & next_head, "expected overlapping tokens between chunks"


def test_oversized_paragraph_is_split():
    cfg = _cfg()
    chunker = TokenChunker(cfg)
    giant = " ".join(["word"] * 2000)
    doc = LoadedDocument("big", "big.txt", "markdown", giant)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1
    for c in chunks:
        assert c.n_tokens <= cfg.max_tokens + cfg.overlap_tokens
