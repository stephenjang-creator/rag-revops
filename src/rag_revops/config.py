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


class RerankConfig(BaseModel):
    enabled: bool = True
    model: str = "rerank-english-v3.0"
    top_n: int = 5              # chunks kept after reranking, passed to generation
    # "Balanced" starting point. Rerank scores are ~[0,1]; below this the best
    # candidate isn't relevant enough to answer from, so we decline. Tune against
    # the Phase 3 eval set rather than by feel.
    min_relevance: float = 0.30
    max_retries: int = 5
    backoff_base_s: float = 5.0
    backoff_cap_s: float = 60.0


class AnalyticalConfig(BaseModel):
    """Cross-corpus 'which contracts have X' retrieval (contract-level).

    Membership is decided by an LLM judge (not a rerank threshold), because the
    cross-encoder scores paraphrased legal clauses ('without cause', 'sole
    discretion') near zero and would drop most genuine matches. The reranker is
    used only to ORDER which contracts the judge sees first.
    """
    # Dense pool feeding the per-contract collapse. 0 = fetch the ENTIRE
    # collection, so every contract is represented no matter how large the corpus
    # grows (Chroma caps at the collection size). A positive value caps the pool
    # for cost control, at the risk of leaving some contracts unrepresented.
    pool_size: int = 0
    chunks_per_contract: int = 4  # chunks per contract handed to the LLM judge
    # Candidate contracts passed to the LLM judge. 0 = judge EVERY contract (the
    # whole set). A positive value caps it — cheaper, but then a real match ranked
    # below the cap is never judged. Judging every contract costs more API credit
    # (one judge call per judge_batch_size contracts); that is the intended
    # trade-off for complete coverage.
    max_contracts: int = 0
    judge_batch_size: int = 10    # smaller batches = more careful per-candidate judging


class ClauseConfig(BaseModel):
    """'Show me example language for clause X' — precision retrieval that leans on
    the cross-encoder reranker.

    This is the reranker's home turf, the opposite of analytical mode: the user
    wants passages that most LITERALLY express a clause (drafting references), so
    the calibrated rerank score IS the membership signal here — a low score
    genuinely means "not good example language for this clause". Retrieval + rerank
    only; no LLM judge.
    """
    pool_size: int = 60          # dense candidates handed to the reranker
    max_options: int = 6         # distinct example clauses returned (best per contract)
    min_relevance: float = 0.30  # cross-encoder floor to count as usable language


class RewriteConfig(BaseModel):
    """Query rewriting runs on EVERY query in every mode, but it's a tiny call, so
    it can run on a cheaper provider than the main generation model.

    Defaults to Cohere's `command` model — the app already collects a Cohere key
    (for embeddings + rerank) and Cohere's trial keys are free, so this keeps query
    rewrite off the Anthropic bill with no extra key or dependency. Falls back to
    Anthropic automatically when no Cohere key is present. Each provider path uses
    its own model id so a fallback never sends a Cohere id to Anthropic.
    """
    provider: str = "cohere"                    # "cohere" | "anthropic"
    cohere_model: str = "command-r-08-2024"
    anthropic_model: str = "claude-sonnet-4-6"
    temperature: float = 0.0
    max_tokens: int = 200


class GenerationConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    temperature: float = 0.0


class EvalConfig(BaseModel):
    # Ragas uses an LLM as judge. We use Claude to keep the stack Anthropic-native
    # and avoid a second vendor. Note: generating and judging with the same model
    # family carries a mild self-preference bias — acceptable for a portfolio eval,
    # and the judge model can be swapped here for a fully independent grader.
    judge_model: str = "claude-sonnet-4-6"
    judge_max_tokens: int = 1024
    # CI regression gate: build fails if mean faithfulness on answered items drops
    # below this. Calibrated below the measured ~0.78 baseline (15-item golden set)
    # so it catches a real regression without flaking on small-sample judge variance.
    min_faithfulness: float = 0.70
    # Metrics to compute (must match names the runner knows).
    metrics: list[str] = Field(
        default_factory=lambda: ["faithfulness", "answer_relevancy", "context_precision"]
    )


class AnalyticalEvalConfig(BaseModel):
    # Set-based eval (precision/recall/F1 over contract sets). Deterministic — no
    # judge calls — so it's cheap enough for a CI gate. Gate on mean F1.
    min_f1: float = 0.60


class PromptConfig(BaseModel):
    system: str
    user_template: str
    decline_message: str = "I couldn't find support for that in the provided documents."


class Settings(BaseModel):
    version: str
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    analytical: AnalyticalConfig = Field(default_factory=AnalyticalConfig)
    clause: ClauseConfig = Field(default_factory=ClauseConfig)
    rewrite: RewriteConfig = Field(default_factory=RewriteConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    analytical_eval: AnalyticalEvalConfig = Field(default_factory=AnalyticalEvalConfig)
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
    settings = Settings.model_validate(raw)
    # Resolve the {decline_message} placeholder in the system prompt so the
    # canonical decline string lives in exactly one place. Only that placeholder
    # is substituted; other braces in the prompt (if any) are left untouched.
    settings.prompts.system = settings.prompts.system.replace(
        "{decline_message}", settings.prompts.decline_message
    )
    return settings


@lru_cache(maxsize=1)
def load_secrets() -> Secrets:
    return Secrets()
