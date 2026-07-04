"""Token-aware chunking.

Splits normalized document text into chunks of ~500-800 tokens with a 100-token
overlap. The splitter is paragraph-aware: it packs whole paragraphs together up
to the target size and only falls back to sentence/token splitting when a single
paragraph exceeds the max — so we rarely slice a clause mid-sentence.

Token counts use tiktoken (`cl100k_base`) to stay consistent with the counting
assumptions downstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken

from .config import ChunkingConfig
from .loaders import LoadedDocument

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str            # f"{doc_id}::{index}"
    doc_id: str
    index: int
    text: str
    n_tokens: int
    metadata: dict = field(default_factory=dict)


class TokenChunker:
    def __init__(self, cfg: ChunkingConfig):
        self.cfg = cfg
        self._enc = tiktoken.get_encoding(cfg.tokenizer)

    def _count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _split_oversized(self, text: str) -> list[str]:
        """Break a too-long block into <= max_tokens pieces on sentence bounds,
        falling back to a hard token window if a single sentence is still huge."""
        sentences = _SENTENCE_SPLIT.split(text)
        pieces: list[str] = []
        buf: list[str] = []
        buf_tokens = 0
        for sent in sentences:
            st = self._count(sent)
            if st > self.cfg.max_tokens:
                # Flush buffer, then hard-window the giant sentence.
                if buf:
                    pieces.append(" ".join(buf))
                    buf, buf_tokens = [], 0
                pieces.extend(self._hard_window(sent))
                continue
            if buf_tokens + st > self.cfg.max_tokens:
                pieces.append(" ".join(buf))
                buf, buf_tokens = [sent], st
            else:
                buf.append(sent)
                buf_tokens += st
        if buf:
            pieces.append(" ".join(buf))
        return pieces

    def _hard_window(self, text: str) -> list[str]:
        toks = self._enc.encode(text)
        step = self.cfg.max_tokens
        return [
            self._enc.decode(toks[i : i + step]).strip()
            for i in range(0, len(toks), step)
        ]

    def _overlap_tail(self, text: str) -> str:
        """Return the trailing `overlap_tokens` worth of `text` as a string."""
        toks = self._enc.encode(text)
        if len(toks) <= self.cfg.overlap_tokens:
            return text
        return self._enc.decode(toks[-self.cfg.overlap_tokens :]).strip()

    def chunk_document(self, doc: LoadedDocument) -> list[Chunk]:
        paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(doc.text) if p.strip()]

        # Normalize into blocks that individually fit under max_tokens.
        blocks: list[str] = []
        for para in paragraphs:
            if self._count(para) > self.cfg.max_tokens:
                blocks.extend(self._split_oversized(para))
            else:
                blocks.append(para)

        chunks: list[Chunk] = []
        buf: list[str] = []
        buf_tokens = 0
        carry = ""  # overlap text carried from the previous emitted chunk

        def emit() -> None:
            nonlocal buf, buf_tokens, carry
            if not buf:
                return
            body = " ".join(buf).strip()
            text = f"{carry} {body}".strip() if carry else body
            idx = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{idx}",
                    doc_id=doc.doc_id,
                    index=idx,
                    text=text,
                    n_tokens=self._count(text),
                    metadata={
                        "source_path": doc.source_path,
                        "doc_type": doc.doc_type,
                        **doc.metadata,
                    },
                )
            )
            carry = self._overlap_tail(text)
            buf, buf_tokens = [], 0

        for block in blocks:
            bt = self._count(block)
            if buf_tokens + bt > self.cfg.max_tokens and buf_tokens >= self.cfg.min_tokens:
                emit()
            buf.append(block)
            buf_tokens += bt
            # If we've comfortably passed the target, emit to keep sizes tight.
            if buf_tokens >= self.cfg.target_tokens:
                emit()
        emit()
        return chunks

    def chunk_documents(self, docs: list[LoadedDocument]) -> list[Chunk]:
        out: list[Chunk] = []
        for doc in docs:
            out.extend(self.chunk_document(doc))
        return out
