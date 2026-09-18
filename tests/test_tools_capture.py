"""Contract tests for memory_capture_episode over a real stdio ClientSession."""

import json
from contextlib import AbstractAsyncContextManager
from datetime import timedelta
from typing import Any

import psycopg
import pytest
from mcp import ClientSession
from mcp.types import CallToolResult, TextContent
from psycopg.rows import DictRow

from agent_memory.embed import FakeEmbedder
from tests.conftest import backdate

SECRET = "sk-AbCdEf0123456789AbCdEf0123456789"

AKIA_SECRET = "AKIAIOSFODNN7EXAMPLE"
JWT_SECRET = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.token-sig"

EPISODE: dict[str, Any] = {
    "goal": "ship the retry logic",
    "expectation": "tests pass on first run",
    "action": "ran uv run pytest",
    "outcome": "flake in retry backoff test",
    "surprise": 0.6,
    "state_at_encoding": {"mood": "focused", "confidence": 0.8},
    "tags": ["retries", "testing"],
    "raw_text": "raw capture notes",
}

EMBEDDED_TEXT = (
    f"{EPISODE['goal']} {EPISODE['expectation']} {EPISODE['action']} {EPISODE['outcome']}"
)


async def _capture(session: ClientSession, episode: dict[str, Any]) -> int:
    """Happy-path capture; returns the new episode id."""
    result = await session.call_tool("memory_capture_episode", episode)
    assert isinstance(result, CallToolResult)
    assert result.is_error is False
    block = result.content[0]
    assert isinstance(block, TextContent)
    payload: dict[str, object] = json.loads(block.text)
    episode_id = payload["id"]
    assert isinstance(episode_id, int)
    return episode_id


async def test_capture_round_trips_row_fields(
    db: psycopg.Connection[DictRow], client: AbstractAsyncContextManager[ClientSession]
) -> None:
    async with client as session:
        episode_id = await _capture(session, EPISODE)
    row = db.execute("SELECT * FROM episodes WHERE id = %(id)s", {"id": episode_id}).fetchone()
    assert row is not None
    assert row["namespace"] == "default@local"  # None -> settings.MEMORY_NAMESPACE
    assert row["goal"] == EPISODE["goal"]
    assert row["expectation"] == EPISODE["expectation"]
    assert row["action"] == EPISODE["action"]
    assert row["outcome"] == EPISODE["outcome"]
    assert row["surprise"] == pytest.approx(0.6)
    assert row["state_at_encoding"] == EPISODE["state_at_encoding"]
    assert row["tags"] == EPISODE["tags"]
    assert row["raw_text"] == EPISODE["raw_text"]
    assert str(row["search_tsv"]).strip() != ""
    expected = FakeEmbedder(dim=8).embed([EMBEDDED_TEXT])[0]
    assert row["embedding"].to_list() == pytest.approx(expected)


class TestStateExcludedFromEmbedding:
    """P4: the embedding must match the 4-field text, not the state-term variant."""

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        four_field = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        five_field = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return {
            EMBEDDED_TEXT: four_field,
            f"{EMBEDDED_TEXT} quantum-telemetry-alpha": five_field,
        }

    async def test_state_stored_but_not_embedded(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode = {**EPISODE, "state_at_encoding": {"signal": "quantum-telemetry-alpha"}}
        async with client as session:
            episode_id = await _capture(session, episode)
        row = db.execute(
            "SELECT embedding, state_at_encoding FROM episodes WHERE id = %(id)s",
            {"id": episode_id},
        ).fetchone()
        assert row is not None
        assert row["state_at_encoding"] == {"signal": "quantum-telemetry-alpha"}
        assert row["embedding"].to_list() == pytest.approx(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )


async def test_namespace_override_honored(
    db: psycopg.Connection[DictRow], client: AbstractAsyncContextManager[ClientSession]
) -> None:
    async with client as session:
        episode_id = await _capture(session, {**EPISODE, "namespace": "orbis@proj-hash"})
    row = db.execute(
        "SELECT namespace FROM episodes WHERE id = %(id)s", {"id": episode_id}
    ).fetchone()
    assert row is not None
    assert row["namespace"] == "orbis@proj-hash"


@pytest.mark.parametrize(("given", "stored"), [(1.7, 1.0), (-0.25, 0.0)])
async def test_surprise_clamped_to_unit_interval(
    db: psycopg.Connection[DictRow],
    client: AbstractAsyncContextManager[ClientSession],
    given: float,
    stored: float,
) -> None:
    async with client as session:
        episode_id = await _capture(session, {**EPISODE, "surprise": given})
    row = db.execute(
        "SELECT surprise FROM episodes WHERE id = %(id)s", {"id": episode_id}
    ).fetchone()
    assert row is not None
    assert row["surprise"] == pytest.approx(stored)


@pytest.mark.parametrize("field", ["outcome", "goal", "raw_text"])
async def test_secret_rejected_without_write_and_session_recovers(
    db: psycopg.Connection[DictRow], client: AbstractAsyncContextManager[ClientSession], field: str
) -> None:
    """Failure QA: isError result, zero rows, and the SAME session still works."""
    episode = {**EPISODE, field: f"context around {SECRET} embedded in {field}"}
    async with client as session:
        result = await session.call_tool("memory_capture_episode", episode)
        assert isinstance(result, CallToolResult)
        assert result.is_error is True
        block = result.content[0]
        assert isinstance(block, TextContent)
        error_text = block.text
        assert field in error_text
        assert SECRET not in error_text

        episode_id = await _capture(session, EPISODE)

    count = db.execute("SELECT count(*) AS n FROM episodes").fetchone()
    assert count is not None and count["n"] == 1
    assert episode_id >= 1


class TestStateSecretScreening:
    """Issue #13: state_at_encoding JSON is screened recursively — every
    string value at any nesting depth, against the shared pattern list."""

    async def test_secret_nested_in_dict_rejected_without_write_and_session_recovers(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        """dict-in-dict secret → isError, field named, no echo, no row, same
        session still captures."""
        episode = {**EPISODE, "state_at_encoding": {"mood": {"note": f"leak {SECRET} here"}}}
        async with client as session:
            result = await session.call_tool("memory_capture_episode", episode)
            assert isinstance(result, CallToolResult)
            assert result.is_error is True
            block = result.content[0]
            assert isinstance(block, TextContent)
            error_text = block.text
            assert "state_at_encoding" in error_text
            assert SECRET not in error_text

            episode_id = await _capture(session, EPISODE)

        count = db.execute("SELECT count(*) AS n FROM episodes").fetchone()
        assert count is not None and count["n"] == 1
        assert episode_id >= 1

    @pytest.mark.parametrize(
        "secret", [SECRET, AKIA_SECRET, JWT_SECRET], ids=["sk-", "AKIA", "jwt"]
    )
    async def test_secret_inside_list_value_rejected(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
        secret: str,
    ) -> None:
        """Strings inside list values are screened at any depth, for multiple
        credential patterns."""
        episode = {**EPISODE, "state_at_encoding": {"history": [f"ping {secret} failed"]}}
        async with client as session:
            result = await session.call_tool("memory_capture_episode", episode)
            assert isinstance(result, CallToolResult)
            assert result.is_error is True
            block = result.content[0]
            assert isinstance(block, TextContent)
            error_text = block.text
            assert "state_at_encoding" in error_text
            assert secret not in error_text

        count = db.execute("SELECT count(*) AS n FROM episodes").fetchone()
        assert count is not None and count["n"] == 0

    async def test_deep_clean_state_round_trips(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        """Numbers/bools/None/lists nested at depth pass the screen unchanged."""
        state = {
            "mood": "focused",
            "confidence": 0.8,
            "flags": {"caffeinated": True, "interrupted": False},
            "stack": [{"depth": 3, "labels": ["a", "b"]}, "tail"],
            "absent": None,
        }
        async with client as session:
            episode_id = await _capture(session, {**EPISODE, "state_at_encoding": state})
        row = db.execute(
            "SELECT state_at_encoding FROM episodes WHERE id = %(id)s", {"id": episode_id}
        ).fetchone()
        assert row is not None
        assert row["state_at_encoding"] == state


async def test_backdate_shifts_created_at(
    db: psycopg.Connection[DictRow], client: AbstractAsyncContextManager[ClientSession]
) -> None:
    """Smoke for the shared time-control helper later tasks depend on."""
    async with client as session:
        episode_id = await _capture(session, EPISODE)
    before = db.execute(
        "SELECT created_at FROM episodes WHERE id = %(id)s", {"id": episode_id}
    ).fetchone()
    assert before is not None
    backdate("episodes", episode_id, hours=-2)
    after = db.execute(
        "SELECT created_at FROM episodes WHERE id = %(id)s", {"id": episode_id}
    ).fetchone()
    assert after is not None
    assert after["created_at"] == pytest.approx(
        before["created_at"] - timedelta(hours=2), abs=timedelta(seconds=1)
    )
