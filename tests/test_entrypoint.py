"""Task-15 entrypoint contract: 13 tools over the REAL ``python -m agent_memory``
path, generated inputSchema integrity, and the CLI subcommands.

Every client test spawns the real dispatcher (``python -m agent_memory``, no
args -> serve()); every CLI test spawns the same module with subcommand args.
SQL asserts ride the ``db`` fixture (misleading_success_output guard) — stdout
and tool payloads are never trusted alone. Deterministic throughout: no
wall-clock dependence (dead-DB refusal is immediate), per-test truncate, and
bounded subprocess timeouts (hung_commands).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any

import pgvector
import psycopg
import pytest
import yaml
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult, TextContent
from psycopg.rows import DictRow

from tests.conftest import backdate

EXPECTED_TOOLS = frozenset(
    {
        "memory_capture_episode",
        "memory_probe",
        "memory_search",
        "memory_report_usage",
        "memory_consolidate_scan",
        "memory_write_lesson",
        "memory_corroborate",
        "memory_contradict",
        "memory_promote",
        "memory_demote",
        "memory_dispute",
        "memory_digest",
        "memory_stats",
    }
)

# Keyword-only no-default params that MUST surface as required in the
# generated inputSchema, alongside each tool's other required args.
REQUIRED_BY_TOOL: dict[str, set[str]] = {
    "memory_write_lesson": {"claim", "because", "evidence"},
    "memory_promote": {"lesson_id", "reason"},
    "memory_demote": {"lesson_id", "reason"},
    "memory_dispute": {"lesson_id", "reason"},
}

CLI_ENV_NAMESPACE = "env-ns@proj"


def _payload(result: CallToolResult) -> dict[str, Any]:
    """Decode a successful tool result's JSON text block."""
    assert result.is_error is False
    block = result.content[0]
    assert isinstance(block, TextContent)
    payload: dict[str, Any] = json.loads(block.text)
    return payload


def _spawn(
    pg: str, *, mcp_args: list[str], env_extra: dict[str, str]
) -> AbstractAsyncContextManager[ClientSession]:
    """Module-local server spawn mirroring conftest's client, with extra CLI
    args (the --namespace channel) and extra env (the MEMORY_NAMESPACE env
    tier of the precedence chain)."""
    digest_dir = tempfile.mkdtemp(prefix="agent-memory-digest-")
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agent_memory", *mcp_args],
        env={
            "DATABASE_URL": pg,
            "EMBED_IMPL": "fake",
            "PGVECTOR_DIM": "8",
            "FAKE_EMBED_OVERRIDES": "{}",
            "DIGEST_DIR": digest_dir,
            **env_extra,
        },
    )

    @asynccontextmanager
    async def enter() -> AsyncIterator[ClientSession]:
        try:
            async with stdio_client(parameters) as (read_stream, write_stream):
                async with ClientSession(
                    read_stream, write_stream, read_timeout_seconds=60.0
                ) as session:
                    await session.initialize()
                    yield session
        finally:
            shutil.rmtree(digest_dir, ignore_errors=True)

    return enter()


def _run_cli(
    pg: str, *args: str, env_extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the real entrypoint with CLI args; env carries MEMORY_NAMESPACE so
    every call exercises the env tier of the precedence chain."""
    env = {
        **os.environ,
        "DATABASE_URL": pg,
        "MEMORY_NAMESPACE": CLI_ENV_NAMESPACE,
        **(env_extra or {}),
    }
    return subprocess.run(
        [sys.executable, "-m", "agent_memory", *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        stdin=subprocess.DEVNULL,
    )


def _insert_episode(
    db: psycopg.Connection[DictRow], namespace: str, goal: str
) -> int:
    """Direct-SQL episode fixture (task-7 recipe: raw_text NOT NULL, no default)."""
    row = db.execute(
        """
        INSERT INTO episodes (namespace, goal, raw_text, embedding)
        VALUES (%(namespace)s, %(goal)s, '', %(embedding)s)
        RETURNING id
        """,
        {
            "namespace": namespace,
            "goal": goal,
            "embedding": pgvector.Vector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        },
    ).fetchone()
    assert row is not None
    return int(row["id"])


async def test_list_tools_exactly_thirteen(
    db: psycopg.Connection[DictRow], client: AbstractAsyncContextManager[ClientSession]
) -> None:
    """The REAL entrypoint serves exactly the 13 plan tools — no extras, none
    missing (set equality, not a subset check)."""
    async with client as session:
        listing = await session.list_tools()
    names = {tool.name for tool in listing.tools}
    assert names == EXPECTED_TOOLS


async def test_every_input_schema_parses_and_marks_required(
    db: psycopg.Connection[DictRow], client: AbstractAsyncContextManager[ClientSession]
) -> None:
    """Each generated inputSchema is JSON-clean, is an object schema, carries
    the trailing namespace param as optional, and marks the keyword-only
    required params (evidence, reason) REQUIRED."""
    async with client as session:
        listing = await session.list_tools()
    schemas: dict[str, dict[str, Any]] = {}
    for tool in listing.tools:
        # Round-trip proves the schema is JSON-serializable and re-parsable.
        schema: dict[str, Any] = json.loads(json.dumps(tool.input_schema))
        assert schema["type"] == "object"
        assert "namespace" in schema["properties"]
        assert "namespace" not in schema.get("required", [])
        schemas[tool.name] = schema
    for name, expected_required in REQUIRED_BY_TOOL.items():
        assert set(schemas[name].get("required", [])) == expected_required, name


async def test_namespace_precedence_env_flag_tool_param(
    db: psycopg.Connection[DictRow], pg: str
) -> None:
    """env < --namespace < per-tool param, end to end through the real
    dispatcher: server launched with MEMORY_NAMESPACE=A and --namespace B;
    a tool call with namespace=C lands in C, a call without lands in B, and
    nothing ever lands in A (SQL-verified)."""
    server = _spawn(
        pg,
        mcp_args=["--namespace", "flag-b@proj"],
        env_extra={"MEMORY_NAMESPACE": "env-a@proj"},
    )
    async with server as session:
        with_param = await session.call_tool(
            "memory_capture_episode",
            {"goal": "tool param wins", "namespace": "tool-c@proj"},
        )
        without_param = await session.call_tool(
            "memory_capture_episode", {"goal": "flag beats env"}
        )
        stats_result = await session.call_tool("memory_stats", {})
    id_c = _payload(with_param)["id"]
    id_b = _payload(without_param)["id"]
    stats_payload = _payload(stats_result)

    rows = db.execute(
        "SELECT id, namespace FROM episodes ORDER BY id"
    ).fetchall()
    assert [(row["id"], row["namespace"]) for row in rows] == [
        (id_c, "tool-c@proj"),
        (id_b, "flag-b@proj"),
    ]
    assert stats_payload["namespace"] == "flag-b@proj"
    assert stats_payload["episode_count"] == 1


async def test_cli_stats_reflects_namespace_flag(
    db: psycopg.Connection[DictRow], pg: str
) -> None:
    """`agent-memory --namespace X stats` reflects X; without the flag the
    env namespace wins and sees none of X's episodes."""
    _insert_episode(db, "flag-ns@proj", "the only episode")

    with_flag = _run_cli(pg, "--namespace", "flag-ns@proj", "stats")
    assert with_flag.returncode == 0, with_flag.stderr
    flagged = json.loads(with_flag.stdout)
    assert flagged["namespace"] == "flag-ns@proj"
    assert flagged["episode_count"] == 1

    without_flag = _run_cli(pg, "stats")
    assert without_flag.returncode == 0, without_flag.stderr
    env_only = json.loads(without_flag.stdout)
    assert env_only["namespace"] == CLI_ENV_NAMESPACE
    assert env_only["episode_count"] == 0


async def test_cli_digest_prints_path(
    db: psycopg.Connection[DictRow], pg: str, tmp_path: Path
) -> None:
    """`agent-memory digest` renders the file and prints exactly its path."""
    digest_dir = tmp_path / "digests"
    ran = _run_cli(
        pg,
        "--namespace",
        "cli-digest@proj",
        "digest",
        env_extra={"DIGEST_DIR": str(digest_dir)},
    )
    assert ran.returncode == 0, ran.stderr
    printed = ran.stdout.strip()
    assert printed.count("\n") == 0  # the path and nothing else
    path = Path(printed)
    assert path.is_file()
    assert path.parent == digest_dir

    frontmatter, _, _ = path.read_text().partition("\n---\n")
    meta: dict[str, Any] = yaml.safe_load(frontmatter.removeprefix("---\n"))
    assert meta["namespace"] == "cli-digest@proj"  # flag beat the env tier


async def test_cli_consolidate_scan_emits_clusters_json(
    db: psycopg.Connection[DictRow], pg: str
) -> None:
    """`agent-memory consolidate-scan` (the §8 cron entrypoint) prints the
    scan payload as JSON with clusters + rederivation_groups keys."""
    for goal in ("first near-duplicate", "second near-duplicate"):
        episode_id = _insert_episode(db, "scan-ns@proj", goal)
        backdate("episodes", episode_id, hours=-2)  # past the age gate

    ran = _run_cli(pg, "--namespace", "scan-ns@proj", "consolidate-scan")
    assert ran.returncode == 0, ran.stderr
    payload = json.loads(ran.stdout)
    assert set(payload) == {"pool", "namespace", "clusters", "rederivation_groups"}
    assert payload["namespace"] == "scan-ns@proj"
    assert payload["pool"] == "fresh"
    assert payload["rederivation_groups"] == []
    assert len(payload["clusters"]) == 1
    members = payload["clusters"][0]["episodes"]
    assert len(members) == 2  # identical vectors -> one 2-episode cluster


def test_server_startup_fails_fast_on_dead_database() -> None:
    """Failure QA: a dead DATABASE_URL kills the entrypoint at startup with a
    one-line stderr message and a nonzero exit — never a hang or a silent 0."""
    dead_url = "postgresql://agent_memory:x@127.0.0.1:1/agent_memory"
    started = time.monotonic()
    try:
        ran = subprocess.run(
            [sys.executable, "-m", "agent_memory"],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "DATABASE_URL": dead_url},
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("server did not fail fast on a dead DATABASE_URL (hung >30s)")
    elapsed = time.monotonic() - started
    assert elapsed < 20, f"startup failure took {elapsed:.1f}s"
    assert ran.returncode != 0
    assert "agent-memory:" in ran.stderr
    assert "OperationalError" in ran.stderr
    assert "Traceback" not in ran.stderr
