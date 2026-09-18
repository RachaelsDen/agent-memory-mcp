"""Issue #4: session-scoped namespace + clientInfo-derived agent default.

Every test spawns a module-local stdio server (conftest's shape) so each can
pin its own clientInfo name and env tier; SQL asserts ride the ``db``
fixture, never trusting tool payloads alone. Fixture-order contract: request
``db`` before ``client`` — the ``db`` fixture performs the per-test TRUNCATE.

Precedence under test (Issue #4):
    tool param  >  session-set (memory_set_namespace)  >  env (incl. the
    --namespace flag, which sets env)  >  derived default
where the derived default is ``<sanitized clientInfo name>@local`` when
MEMORY_NAMESPACE is unset, else the env value, else ``default@local`` when no
clientInfo was observed.
"""

import json
import shutil
import sys
import tempfile
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

import psycopg
import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult, Implementation, TextContent
from psycopg.rows import DictRow


def _spawn(
    pg: str,
    *,
    client_name: str | None = None,
    env_extra: dict[str, str] | None = None,
    mcp_args: list[str] | None = None,
) -> AbstractAsyncContextManager[ClientSession]:
    """Module-local server spawn mirroring conftest's client, with a pinned
    clientInfo name (the derive tier), extra env (the env tier), and extra
    CLI args (the --namespace flag tier)."""
    digest_dir = tempfile.mkdtemp(prefix="agent-memory-digest-")
    client_info = (
        None if client_name is None else Implementation(name=client_name, version="9.9")
    )
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agent_memory", *(mcp_args or [])],
        env={
            "DATABASE_URL": pg,
            "EMBED_IMPL": "fake",
            "PGVECTOR_DIM": "8",
            "FAKE_EMBED_OVERRIDES": "{}",
            "DIGEST_DIR": digest_dir,
            **(env_extra or {}),
        },
    )

    @asynccontextmanager
    async def enter() -> AsyncIterator[ClientSession]:
        try:
            async with stdio_client(parameters) as (read_stream, write_stream):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=60.0,
                    client_info=client_info,
                ) as session:
                    await session.initialize()
                    yield session
        finally:
            shutil.rmtree(digest_dir, ignore_errors=True)

    return enter()


def _payload(result: CallToolResult) -> dict[str, Any]:
    """Decode a successful tool result's JSON text block."""
    assert result.is_error is False, result.content
    block = result.content[0]
    assert isinstance(block, TextContent)
    payload: dict[str, Any] = json.loads(block.text)
    return payload


def _error_text(result: CallToolResult) -> str:
    assert result.is_error is True
    if not result.content:
        return ""
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


def _episode_namespaces(db: psycopg.Connection[DictRow]) -> list[str]:
    rows = db.execute("SELECT namespace FROM episodes ORDER BY id").fetchall()
    return [str(row["namespace"]) for row in rows]


async def test_session_set_honored_by_capture_and_probe(
    db: psycopg.Connection[DictRow], pg: str
) -> None:
    """Given no env namespace, When the session sets one and capture + probe
    run without a namespace param, Then episodes and the retrieval_event row
    both land in the session namespace (SQL-verified)."""
    server = _spawn(pg)
    async with server as session:
        set_result = await session.call_tool(
            "memory_set_namespace", {"namespace": "session-ns@proj"}
        )
        assert _payload(set_result)["namespace"] == "session-ns@proj"
        capture = await session.call_tool(
            "memory_capture_episode", {"goal": "session default applies"}
        )
        probe = await session.call_tool("memory_probe", {"current_goal": "session ns"})

    episode_id = _payload(capture)["id"]
    event_id = _payload(probe)["retrieval_event_id"]
    episode_ns = db.execute(
        "SELECT namespace FROM episodes WHERE id = %(id)s", {"id": episode_id}
    ).fetchone()
    event_ns = db.execute(
        "SELECT namespace FROM retrieval_events WHERE id = %(id)s", {"id": event_id}
    ).fetchone()
    assert episode_ns is not None and episode_ns["namespace"] == "session-ns@proj"
    assert event_ns is not None and event_ns["namespace"] == "session-ns@proj"


async def test_tool_param_beats_session_set(
    db: psycopg.Connection[DictRow], pg: str
) -> None:
    """Given a session-set namespace, When capture passes an explicit
    namespace param, Then the param wins; a bare capture still uses the
    session default."""
    server = _spawn(pg)
    async with server as session:
        await session.call_tool(
            "memory_set_namespace", {"namespace": "session-ns@proj"}
        )
        await session.call_tool(
            "memory_capture_episode",
            {"goal": "param wins", "namespace": "param-ns@proj"},
        )
        await session.call_tool(
            "memory_capture_episode", {"goal": "session default"}
        )

    assert _episode_namespaces(db) == ["param-ns@proj", "session-ns@proj"]


async def test_env_wins_when_session_unset_and_flag_beats_env(
    db: psycopg.Connection[DictRow], pg: str
) -> None:
    """Given MEMORY_NAMESPACE in env (and the flag overriding it) but no
    session-set, When a bare capture runs, Then it lands in the flag's
    namespace — env+flag behave exactly as before the session tier existed."""
    server = _spawn(
        pg,
        mcp_args=["--namespace", "flag-b@proj"],
        env_extra={"MEMORY_NAMESPACE": "env-a@proj"},
    )
    async with server as session:
        await session.call_tool("memory_capture_episode", {"goal": "no session set"})

    assert _episode_namespaces(db) == ["flag-b@proj"]


async def test_session_set_beats_env(
    db: psycopg.Connection[DictRow], pg: str
) -> None:
    """Given MEMORY_NAMESPACE in env, When the session sets a different
    namespace and capture runs bare, Then the session value outranks env."""
    server = _spawn(pg, env_extra={"MEMORY_NAMESPACE": "env-a@proj"})
    async with server as session:
        await session.call_tool(
            "memory_set_namespace", {"namespace": "session-ns@proj"}
        )
        await session.call_tool("memory_capture_episode", {"goal": "session > env"})

    assert _episode_namespaces(db) == ["session-ns@proj"]


async def test_precedence_matrix_param_session_env(
    db: psycopg.Connection[DictRow], pg: str
) -> None:
    """The full matrix in one session — param > session > env — with the
    env tier itself active (MEMORY_NAMESPACE set at spawn)."""
    server = _spawn(pg, env_extra={"MEMORY_NAMESPACE": "env-a@proj"})
    async with server as session:
        await session.call_tool(
            "memory_set_namespace", {"namespace": "session-ns@proj"}
        )
        await session.call_tool(
            "memory_capture_episode",
            {"goal": "param tier", "namespace": "param-ns@proj"},
        )
        await session.call_tool(
            "memory_capture_episode", {"goal": "session tier"}
        )
    assert _episode_namespaces(db) == ["param-ns@proj", "session-ns@proj"]


async def test_global_rejected_as_session_default_same_session_recovery(
    db: psycopg.Connection[DictRow], pg: str
) -> None:
    """Given the promotion-only invariant, When the session tries to set
    'global' as its default, Then the call errors naming the invariant and
    the SAME session recovers by setting a legal namespace that subsequent
    captures honor."""
    server = _spawn(pg)
    async with server as session:
        rejected = await session.call_tool(
            "memory_set_namespace", {"namespace": "global"}
        )
        assert rejected.is_error is True
        assert "promotion-only" in _error_text(rejected)
        # session state must be unchanged by the rejected call, and the
        # session stays usable (ToolError never exits the process)
        recovered = await session.call_tool(
            "memory_set_namespace", {"namespace": "recovered@proj"}
        )
        assert _payload(recovered)["namespace"] == "recovered@proj"
        await session.call_tool(
            "memory_capture_episode", {"goal": "after recovery"}
        )

    assert _episode_namespaces(db) == ["recovered@proj"]


async def test_whitespace_only_session_namespace_rejected(
    pg: str,
) -> None:
    """Given minimal validation, When the session sets an empty or
    whitespace-only namespace, Then each call is an error and no session
    state changes (a later capture still uses the pre-call default)."""
    server = _spawn(pg)
    async with server as session:
        for bad in ("", "   "):
            rejected = await session.call_tool(
                "memory_set_namespace", {"namespace": bad}
            )
            assert rejected.is_error is True, bad


async def test_clientinfo_derived_default_when_env_unset(
    db: psycopg.Connection[DictRow], pg: str
) -> None:
    """Given MEMORY_NAMESPACE unset and a clientInfo name of 'Quokka Host',
    When a bare capture runs, Then it lands in the sanitized derived default
    'quokka-host@local' (SQL-verified), not 'default@local'."""
    server = _spawn(pg, client_name="Quokka Host")
    async with server as session:
        capture = await session.call_tool(
            "memory_capture_episode", {"goal": "derived default"}
        )
        stats_result = await session.call_tool("memory_stats", {})

    episode_id = _payload(capture)["id"]
    row = db.execute(
        "SELECT namespace FROM episodes WHERE id = %(id)s", {"id": episode_id}
    ).fetchone()
    assert row is not None and row["namespace"] == "quokka-host@local"
    assert _payload(stats_result)["namespace"] == "quokka-host@local"


async def test_env_set_beats_clientinfo(
    db: psycopg.Connection[DictRow], pg: str
) -> None:
    """Given MEMORY_NAMESPACE set AND a clientInfo name, When a bare capture
    runs, Then the env value wins — explicit config always beats the derived
    default."""
    server = _spawn(
        pg,
        client_name="Quokka Host",
        env_extra={"MEMORY_NAMESPACE": "env-ns@proj"},
    )
    async with server as session:
        await session.call_tool("memory_capture_episode", {"goal": "env wins"})

    assert _episode_namespaces(db) == ["env-ns@proj"]


async def test_no_clientinfo_falls_back_to_default_local(
    db: psycopg.Connection[DictRow], pg: str
) -> None:
    """Given env unset and a client that sends no usable clientInfo name,
    When a bare capture runs, Then the derived default degrades to exactly
    'default@local' (today's behavior)."""
    server = _spawn(pg, client_name="   ")  # sanitizes to empty -> fallback
    async with server as session:
        await session.call_tool("memory_capture_episode", {"goal": "fallback"})

    assert _episode_namespaces(db) == ["default@local"]


def test_session_state_module_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit-level contract of the session_state module: resolve precedence
    and validation are exact over the module state, and the ToolError texts
    name the violated rule. Runs in-process (no server) — the module state
    is per-process by design, and this process never served a session."""
    from mcp.server.mcpserver.exceptions import ToolError

    from agent_memory.config import Settings
    from agent_memory.session_state import (
        resolve_namespace,
        set_session_namespace,
    )

    monkeypatch.setenv("MEMORY_NAMESPACE", "env-ns@proj")
    settings = Settings(MEMORY_NAMESPACE="env-ns@proj")
    # param > (unset session) > env
    assert resolve_namespace(settings, "param-ns@proj") == "param-ns@proj"
    assert resolve_namespace(settings, None) == "env-ns@proj"
    # session-set slots between them
    assert set_session_namespace("session-ns@proj") == "session-ns@proj"
    assert resolve_namespace(settings, None) == "session-ns@proj"
    assert resolve_namespace(settings, "param-ns@proj") == "param-ns@proj"

    with pytest.raises(ToolError, match="promotion-only"):
        set_session_namespace("global")
    assert resolve_namespace(settings, None) == "session-ns@proj"
    with pytest.raises(ToolError):
        set_session_namespace("   ")
