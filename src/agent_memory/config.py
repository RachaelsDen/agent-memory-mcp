"""Application settings via environment variables (pydantic-settings)."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", frozen=True)

    DATABASE_URL: str = "postgresql://agent_memory:agent_memory@localhost:55432/agent_memory"
    MEMORY_NAMESPACE: str = "default@local"
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBED_DEVICE: str = "cpu"
    EMBED_IMPL: Literal["local", "fake"] = "local"
    PGVECTOR_DIM: int = 384
    W_REL: float = 0.45
    W_SAL: float = 0.20
    W_ENV: float = 0.15
    W_USE: float = 0.10
    W_SPREAD: float = 0.10
    TAU_ENV_H: int = 4320
    TAU_USE_H: int = 720
    SIM_FLOOR: float = 0.25
    TS_RANK_SAT: float = 0.1
    STALE_ENV_FRESH: float = 0.2
    SALIENCE_STALE: float = 0.7
    DIGEST_DIR: str = "~/.agent-memory/digest"  # expanduser at use time
    PROBE_TOPK: int = 12
    FINAL_K: int = 8
    CLUSTER_COS: float = 0.82
    DEDUP_COS: float = 0.95
    DEDUP_WINDOW_H: int = 24
    SIMILAR_LINK_COS: float = 0.75
    DUP_CLAIM_COS: float = 0.95
    CONSOLIDATE_MIN_AGE_H: int = 1
    FAKE_EMBED_OVERRIDES: str = ""  # JSON {text: [vector]} map consumed by the fake embedder


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
