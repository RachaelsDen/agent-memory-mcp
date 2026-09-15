"""FakeEmbedder determinism, override precedence, and load_embedder routing."""

import json
import math
import subprocess
import sys

import pytest

from agent_memory.config import Settings
from agent_memory.embed import FakeEmbedder, LocalEmbedder, load_embedder

_CHILD = (
    "import json, sys\n"
    "from agent_memory.embed import FakeEmbedder\n"
    "dim = int(sys.argv[1])\n"
    "overrides = json.loads(sys.argv[2])\n"
    "print(json.dumps(FakeEmbedder(dim, overrides).embed([sys.argv[3]])[0]))\n"
)


def _child_vector(text: str, dim: int, overrides: dict[str, list[float]] | None = None) -> list[float]:
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, str(dim), json.dumps(overrides or {}), text],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _norm(vector: list[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in vector))


def _cosine(left: list[float], right: list[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right)) / (_norm(left) * _norm(right))


def test_hash_vectors_are_unit_norm() -> None:
    embedder = FakeEmbedder(dim=8)
    for text in ["alpha", "the quick brown fox", "", "unicode text", "x" * 500]:
        vector = embedder.embed([text])[0]
        assert len(vector) == 8
        assert _norm(vector) == pytest.approx(1.0)


def test_same_text_same_vector_across_instances() -> None:
    first = FakeEmbedder(dim=16)
    second = FakeEmbedder(dim=16)
    assert first.embed(["determinism", "rocks"]) == second.embed(["determinism", "rocks"])


def test_distinct_texts_distinct_vectors() -> None:
    vectors = FakeEmbedder(dim=8).embed(["alpha", "beta", "gamma", "delta"])
    assert len({tuple(vector) for vector in vectors}) == 4


def test_case_and_whitespace_are_canonicalized() -> None:
    embedder = FakeEmbedder(dim=8)
    variants = ["hello world", "Hello  WORLD", "\thello\n world "]
    vectors = [embedder.embed([text])[0] for text in variants]
    assert len({tuple(vector) for vector in vectors}) == 1


def test_override_takes_precedence_over_hash() -> None:
    override = [1.0, 0.0, 0.0, 0.0]
    embedder = FakeEmbedder(dim=4, overrides={"crafted text": override})
    assert embedder.embed(["crafted text"])[0] == override
    assert embedder.embed(["other text"])[0] != override


def test_override_vector_returned_verbatim() -> None:
    embedder = FakeEmbedder(dim=2, overrides={"v": [3.0, 4.0]})
    assert embedder.embed(["v"])[0] == [3.0, 4.0]


def test_embed_never_raises_on_any_text() -> None:
    embedder = FakeEmbedder(dim=4)
    vectors = embedder.embed(["", "emoji and tabs", "a\x00b", "long " * 2000])
    assert all(len(vector) == 4 for vector in vectors)
    assert embedder.embed([]) == []


def test_same_vector_in_subprocess_as_in_process() -> None:
    text = "cross process determinism"
    in_process = FakeEmbedder(dim=8).embed([text])[0]
    assert _child_vector(text, 8) == in_process


def test_one_char_difference_yields_different_unit_vectors() -> None:
    near, far = FakeEmbedder(dim=8).embed(["explorer", "explorers"])
    assert _norm(near) == pytest.approx(1.0)
    assert _norm(far) == pytest.approx(1.0)
    assert _cosine(near, far) < 0.99


def test_hash_fallback_identical_across_two_subprocesses() -> None:
    overrides = {"known text": [1.0] * 8}
    first = _child_vector("never seen before", 8, overrides)
    second = _child_vector("never seen before", 8, overrides)
    assert first == second
    assert first != overrides["known text"]


def test_load_embedder_routes_to_fake_without_overrides() -> None:
    embedder = load_embedder(Settings(EMBED_IMPL="fake", PGVECTOR_DIM=8, FAKE_EMBED_OVERRIDES=""))
    assert isinstance(embedder, FakeEmbedder)
    assert embedder.dim == 8
    text = "routing check"
    assert embedder.embed([text])[0] == FakeEmbedder(dim=8).embed([text])[0]


def test_load_embedder_routes_overrides_from_settings() -> None:
    embedder = load_embedder(
        Settings(
            EMBED_IMPL="fake",
            PGVECTOR_DIM=2,
            FAKE_EMBED_OVERRIDES='{"crafted": [1.0, 2.0]}',
        )
    )
    assert embedder.embed(["crafted"])[0] == [1.0, 2.0]


def test_load_embedder_routes_to_local_lazily() -> None:
    embedder = load_embedder(
        Settings(EMBED_IMPL="local", EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2", EMBED_DEVICE="cpu")
    )
    assert isinstance(embedder, LocalEmbedder)
    assert "sentence_transformers" not in sys.modules
