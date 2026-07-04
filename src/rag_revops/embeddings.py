"""Cohere embedding wrapper with rate-limit handling."""

from __future__ import annotations

import time

import cohere

from .config import EmbeddingsConfig, load_secrets

try:
    from cohere.errors import TooManyRequestsError as _RateLimitError
except Exception:
    _RateLimitError = None


def _is_rate_limit(exc: Exception) -> bool:
    if _RateLimitError is not None and isinstance(exc, _RateLimitError):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return "429" in text or "rate limit" in text or "too many requests" in text


class CohereEmbedder:
    def __init__(self, cfg: EmbeddingsConfig):
        self.cfg = cfg
        secrets = load_secrets()
        if not secrets.cohere_api_key:
            raise RuntimeError("COHERE_API_KEY is not set (see .env.example).")
        self._client = cohere.Client(api_key=secrets.cohere_api_key)

    def _embed_batch_with_retry(self, batch: list[str], input_type: str) -> list[list[float]]:
        attempt = 0
        while True:
            try:
                resp = self._client.embed(
                    texts=batch,
                    model=self.cfg.model,
                    input_type=input_type,
                )
                return resp.embeddings
            except Exception as exc:
                retryable = _is_rate_limit(exc)
                if not retryable or attempt >= self.cfg.max_retries:
                    raise
                wait = min(
                    self.cfg.backoff_base_s * (2 ** attempt),
                    self.cfg.backoff_cap_s,
                )
                attempt += 1
                print(
                    f"  [rate limit] waiting {wait:.0f}s then retrying "
                    f"(attempt {attempt}/{self.cfg.max_retries})…"
                )
                time.sleep(wait)

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        n_batches = (len(texts) + self.cfg.batch_size - 1) // self.cfg.batch_size
        for i, start in enumerate(range(0, len(texts), self.cfg.batch_size)):
            batch = texts[start : start + self.cfg.batch_size]
            vectors.extend(self._embed_batch_with_retry(batch, input_type))
            if self.cfg.inter_batch_delay_s > 0 and i < n_batches - 1:
                time.sleep(self.cfg.inter_batch_delay_s)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, self.cfg.input_type_document)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch_with_retry([text], self.cfg.input_type_query)[0]