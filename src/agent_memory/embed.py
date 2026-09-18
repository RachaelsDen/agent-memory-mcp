"""Pluggable embedders: local sentence-transformers and a deterministic fake (DESIGN §4, §12)."""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import TYPE_CHECKING, Protocol

from agent_memory.config import Settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    """Anything that turns texts into fixed-width, L2-normalized vectors."""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbedder:
    """SentenceTransformer embedder; the model loads on first use, never at construction."""

    def __init__(self, model: str, device: str) -> None:
        self._model_name = model
        self._device = device
        self._model: SentenceTransformer | None = None

    @property
    def dim(self) -> int:
        dimension = self._load().get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError(f"model {self._model_name!r} exposes no sentence embedding dimension")
        return dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._load().encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy: keeps torch off the import path

            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model


def _hash_vector(text: str, dim: int) -> list[float]:
    canonical = " ".join(text.lower().split())
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    vector = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if norm == 0.0:  # unreachable for dim >= 1; keeps the never-raise contract total
        vector = [1.0, *(0.0 for _ in range(dim - 1))]
        norm = 1.0
    return [value / norm for value in vector]


class FakeEmbedder:
    """Deterministic hash embedder for tests and offline runs.

    Override keys match the raw text exactly; every other text maps to the
    SHA-256-seeded PRNG vector of its lowercased, whitespace-collapsed form,
    so the same text always yields the same vector in any process.
    """

    def __init__(self, dim: int, overrides: dict[str, list[float]] | None = None) -> None:
        self._dim = dim
        self._overrides = overrides or {}

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for(text) for text in texts]

    def _vector_for(self, text: str) -> list[float]:
        if text in self._overrides:
            return [float(value) for value in self._overrides[text]]
        return _hash_vector(text, self._dim)


def load_embedder(settings: Settings) -> Embedder:
    if settings.EMBED_IMPL == "fake":
        overrides = json.loads(settings.FAKE_EMBED_OVERRIDES or "{}")
        return FakeEmbedder(dim=settings.PGVECTOR_DIM, overrides=overrides)
    return LocalEmbedder(settings.EMBED_MODEL, settings.EMBED_DEVICE)
