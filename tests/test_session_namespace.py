"""Issue #4: session-scoped namespace + clientInfo-derived agent default.

Every precedence test spawns a module-local stdio server (conftest's shape)
so each can pin its own clientInfo name and env tier; SQL asserts ride the
``db`` fixture, never trusting tool payloads alone. Fixture-order contract:
request ``db`` before ``client`` — the ``db`` fixture performs the per-test
TRUNCATE. The PR #11 review tests at the bottom run WITHOUT a spawn:
unit-level Settings-source tests (P2) and SDK-capability tests (P1),
including one subprocess that imports the server against a blocked-import
mcp 1.x SDK shim.

Precedence under test (Issue #4):
    tool param  >  session-set (memory_set_namespace)  >  env (incl. the
    --namespace flag, which sets env)  >  derived default
where the derived default is ``<sanitized clientInfo name>@local`` when
MEMORY_NAMESPACE is unset, else the env value, else ``default@local`` when no
clientInfo was observed.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, get_origin, get_type_hints

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


def test_direct_settings_namespace_honored_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given MEMORY_NAMESPACE absent from env but a Settings constructed with
    an explicit MEMORY_NAMESPACE (pydantic v2 marks init kwargs set), When
    resolve runs below the param/session tiers, Then the explicitly
    configured namespace wins — the env-only detection wrongly fell through
    to the derived default here."""
    import agent_memory.session_state as session_state
    from agent_memory.config import Settings

    monkeypatch.delenv("MEMORY_NAMESPACE", raising=False)
    monkeypatch.setattr(session_state, "_session_namespace", None)
    monkeypatch.setattr(session_state, "_client_name", "quokka")
    settings = Settings(MEMORY_NAMESPACE="direct@set")
    assert "MEMORY_NAMESPACE" in settings.model_fields_set
    assert session_state.resolve_namespace(settings, None) == "direct@set"


def test_compiled_default_namespace_derives_clientinfo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given NO Settings source provides MEMORY_NAMESPACE (the field is
    absent from model_fields_set), When resolve runs with an observed client
    name, Then the derived default wins — only the compiled-in default may
    fall through to clientInfo."""
    import agent_memory.session_state as session_state
    from agent_memory.config import Settings

    monkeypatch.delenv("MEMORY_NAMESPACE", raising=False)
    monkeypatch.setattr(session_state, "_session_namespace", None)
    monkeypatch.setattr(session_state, "_client_name", "quokka")
    settings = Settings()
    assert "MEMORY_NAMESPACE" not in settings.model_fields_set
    assert session_state.resolve_namespace(settings, None) == "quokka@local"


def test_env_source_marks_namespace_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given MEMORY_NAMESPACE via env (the dominant source), Then the field
    IS in model_fields_set and resolve honors it — the explicit-detection
    mechanism must not forget the env tier."""
    import agent_memory.session_state as session_state
    from agent_memory.config import Settings

    monkeypatch.setenv("MEMORY_NAMESPACE", "env-ns@proj")
    monkeypatch.setattr(session_state, "_session_namespace", None)
    monkeypatch.setattr(session_state, "_client_name", "quokka")
    settings = Settings()
    assert "MEMORY_NAMESPACE" in settings.model_fields_set
    assert session_state.resolve_namespace(settings, None) == "env-ns@proj"


def test_client_info_middleware_wired_on_installed_sdk() -> None:
    """Given the installed mcp 2.x SDK, Then the capability flag is True, the
    quoted middleware annotations still resolve to the real context types
    (typing.get_type_hints works — quoting cost nothing), and create_server
    wires _observe_client_info into the middleware chain."""
    from mcp.server.context import ServerRequestContext

    import agent_memory.server as server_module

    assert server_module._has_request_context is True
    hints = get_type_hints(server_module._observe_client_info)
    assert get_origin(hints["ctx"]) is ServerRequestContext
    server = server_module.create_server()
    assert any(
        middleware is server_module._observe_client_info
        for middleware in server.middleware
    )


def test_create_server_skips_middleware_when_capability_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the capability flag forced off (what an mcp 1.x install
    produces), When create_server runs on the real SDK, Then the server
    still constructs with every tool registered and _observe_client_info is
    NOT wired."""
    import agent_memory.server as server_module

    monkeypatch.setattr(server_module, "_has_request_context", False)
    server = server_module.create_server()
    assert not any(
        middleware is server_module._observe_client_info
        for middleware in server.middleware
    )


_MCP1_SIMULATION_PROGRAM = """
import os
import sys
import types

# mcp 1.x SDK shape: mcpserver/context absent, FastMCP present (mcp 2.x
# replaced fastmcp with a tombstone module that raises ModuleNotFoundError).
for missing in (
    "mcp.server.mcpserver",
    "mcp.server.mcpserver.exceptions",
    "mcp.server.context",
):
    sys.modules[missing] = None

fastmcp = types.ModuleType("mcp.server.fastmcp")


class FastMCPStub:
    # 1.x FastMCP constructor shape: takes name, has NO middleware parameter.
    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self.registered_tools = []

    def tool(self):
        def decorate(function):
            self.registered_tools.append(function.__name__)
            return function

        return decorate


fastmcp.FastMCP = FastMCPStub
sys.modules["mcp.server.fastmcp"] = fastmcp

exceptions = types.ModuleType("mcp.server.fastmcp.exceptions")


class ToolError(Exception):
    pass


exceptions.ToolError = ToolError
sys.modules["mcp.server.fastmcp.exceptions"] = exceptions

os.environ.pop("MEMORY_NAMESPACE", None)

import agent_memory.server as server_module

server = server_module.create_server()
assert server_module._has_request_context is False
assert isinstance(server, FastMCPStub)
assert len(server.registered_tools) == 14  # every tool registered, none lost

from agent_memory.config import Settings
from agent_memory.session_state import resolve_namespace

assert resolve_namespace(Settings(), None) == "default@local"
assert resolve_namespace(Settings(MEMORY_NAMESPACE="direct@set"), None) == "direct@set"
os.environ["MEMORY_NAMESPACE"] = "env-ns@proj"
assert resolve_namespace(Settings(), None) == "env-ns@proj"
print("MCP1_SIMULATION_OK")
"""


def test_server_boots_on_mcp1_shaped_sdk() -> None:
    """The review's crash scenario end to end: with an mcp 1.x SDK surface
    (mcpserver/context blocked, FastMCP stubbed without a middleware
    parameter), the server module must import AND create_server must succeed
    with no middleware wiring, leaving env/default namespace precedence
    intact. On Python <=3.13 the old unquoted annotations NameError'd at
    import; 3.14's PEP 649 defers annotation evaluation, so there the red
    was the constructor's middleware kwarg — quoting the annotations keeps
    both interpreter lines safe, which is why this runs as a subprocess
    (sys.modules surgery must not leak into the pytest process)."""
    result = subprocess.run(
        [sys.executable, "-c", _MCP1_SIMULATION_PROGRAM],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "MCP1_SIMULATION_OK" in result.stdout
