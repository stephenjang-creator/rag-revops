"""Typed configuration loader.

Loads `config/settings.yaml` into validated Pydantic models. Secrets (API keys)
come from the environment, never from the YAML — see `Secrets`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "settings.yaml"


class ChunkingConfig(BaseModel):
    target_tokens: int = 650
    min_tokens: int = 500
    max_tokens: int = 800
    overlap_tokens: int = 100
    tokenizer: str = "cl100k_base"


class EmbeddingsConfig(BaseModel):
    provider: str = "cohere"
    model: str = "embed-english-v3.0"
    input_type_document: str = "search_document"
    input_type_query: str = "search_query"
    batch_size: int = 96
    # Rate-limit handling (Cohere trial keys are throttled per-minute).
    inter_batch_delay_s: float = 6.0   # proactive pace between embed calls
    max_retries: int = 6               # retries on a rate-limit / transient error
    backoff_base_s: float = 5.0        # first backoff wait; doubles each retry
    backoff_cap_s: float = 60.0        # ceiling for any single backoff wait


class VectorStoreConfig(BaseModel):
    provider: str = "chroma"
    collection: str = "revops_docs"
    persist_dir: str = "data/processed/chroma"
    distance: str = "cosine"


class RetrievalConfig(BaseModel):
    top_k: int = 5
    # Phase 2 hybrid retrieval:
    fetch_k: int = 20        # candidates each retriever pulls before fusion
    rrf_k: int = 60          # RRF damping constant (convention: 60)
    use_hybrid: bool = True  # False falls back to pure dense retrieval


class GenerationConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    temperature: float = 0.0


class PromptConfig(BaseModel):
    system: str
    user_template: str


class Settings(BaseModel):
    version: str
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    prompts: PromptConfig


class Secrets(BaseSettings):
    """API keys pulled from the environment / .env. Never committed."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    cohere_api_key: str = ""


@lru_cache(maxsize=1)
def load_settings(path: Path | None = None) -> Settings:
    cfg_path = path or _DEFAULT_CONFIG
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Settings.model_validate(raw)


@lru_cache(maxsize=1)
def load_secrets() -> Secrets:
    return Secrets()
