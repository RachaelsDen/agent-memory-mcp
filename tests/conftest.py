"""Contract-test harness: testcontainer Postgres + a live stdio MCP server subprocess.

Fixture order is contractual: per test — truncate (`db`) → spawn + initialize
(`client`) → tear the session down → the next test truncates. Tests MUST
request `db` before `client` (pytest instantiates fixtures in signature
order); `client` deliberately does not depend on `db` so later tasks can
compose them freely.

`backdate` is the ONLY time-control mechanism in tests; production code never
uses it. `SET hnsw.ef_search = 100` rides the server's connect path already
(`agent_memory.db.connect`).
"""

import json
import os
import shutil
import sys
import tempfile
from collections.abc import AsyncIterator, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import datetime, timedelta, timezone

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import DictRow
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Implementation
from testcontainers.community.postgres import PostgresContainer

import agent_memory.db as agent_db
from agent_memory.config import get_settings

_TRUNCATE_SQL = (
    "TRUNCATE episodes, lessons, lesson_evidence, lesson_links, "
    "retrieval_events, usage_reports RESTART IDENTITY CASCADE"
)

_BACKDATE_COLUMNS: dict[str, frozenset[str]] = {
    "episodes": frozenset({"created_at"}),
    "lessons": frozenset({"created_at", "last_evidence_at", "updated_at"}),
}


@pytest.fixture(scope="session")
def pg() -> Iterator[str]:
    """Fresh pgvector Postgres per session, migrated at dim=8; yields the libpq URL."""
    started = datetime.now(tz=timezone.utc)
    with PostgresContainer("pgvector/pgvector:pg16") as container:
        url = container.get_connection_url(driver=None)
        os.environ["DATABASE_URL"] = url
        get_settings.cache_clear()  # this process must now see the container URL
        agent_db.migrate(dim=8)
        elapsed = (datetime.now(tz=timezone.utc) - started).total_seconds()
        print(f"\n[pg] testcontainer ready + migrated in {elapsed:.1f}s: {url}")
        yield url


@pytest.fixture()
def db(pg: str) -> Iterator[psycopg.Connection[DictRow]]:
    """Application connection over the testcontainer; truncates every table first."""
    conn = agent_db.connect()
    try:
        conn.execute(_TRUNCATE_SQL)
        yield conn
    finally:
        conn.close()


@pytest.fixture()
def fake_embed_overrides() -> dict[str, list[float]]:
    """Vectors the spawned server's FakeEmbedder returns verbatim, by exact raw text.

    Override this fixture per test/module to craft vectors; keys must equal the
    server-side embedded text EXACTLY (goal + " " + expectation + " " + action
    + " " + outcome for capture).
    """
    return {}


@pytest.fixture()
def client(
    pg: str, fake_embed_overrides: dict[str, list[float]]
) -> AbstractAsyncContextManager[ClientSession]:
    """Real MCP server over stdio: [sys.executable, "-m", "agent_memory"].

    Returns a per-test async context manager — enter it INSIDE the test body
    (``async with client as session:``). It cannot be an async-generator
    fixture yielding the live session: pytest-asyncio finalizes such fixtures
    in a different task than setup, and anyio forbids exiting stdio_client's
    task group outside the task that entered it.
    """
    digest_dir = tempfile.mkdtemp(prefix="agent-memory-digest-")
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agent_memory"],
        env={
            "DATABASE_URL": pg,
            "EMBED_IMPL": "fake",
            "PGVECTOR_DIM": "8",
            "FAKE_EMBED_OVERRIDES": json.dumps(fake_embed_overrides),
            "DIGEST_DIR": digest_dir,
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
                    # Issue #4: the shared fixture session presents as the
                    # generic "default" client so the clientInfo-derived
                    # namespace stays default@local for every existing test.
                    client_info=Implementation(name="default", version="0.0.0"),
                ) as session:
                    await session.initialize()
                    yield session
        finally:
            shutil.rmtree(digest_dir, ignore_errors=True)

    return enter()


def backdate(
    target: str, row_id: int, *, at: datetime | None = None, **timedelta_kwargs: float
) -> None:
    """Rewind one timestamp column of one row (test-only time control).

    `target` is "<table>" (defaults to created_at) or "<table>.<column>"; the
    legal targets are episodes.created_at and lessons.created_at /
    last_evidence_at / updated_at. Exactly one of the tz-aware absolute `at=`
    (required whenever dates must be deterministic against DB now()) and the
    relative timedelta kwargs (days=, hours=, ...) may be given.
    """
    table, _, column = target.partition(".")
    column = column or "created_at"
    if table not in _BACKDATE_COLUMNS or column not in _BACKDATE_COLUMNS[table]:
        raise ValueError(f"unsupported backdate target: {target!r}")
    if (at is None) == (not timedelta_kwargs):
        raise ValueError("exactly one of at= and the relative kwargs must be provided")
    if at is not None:
        if at.tzinfo is None:
            raise ValueError("at= must be tz-aware")
        moment = at
    else:
        moment = datetime.now(timezone.utc) + timedelta(**timedelta_kwargs)
    conn = agent_db.connect()
    try:
        query = sql.SQL("UPDATE {} SET {} = %(moment)s WHERE id = %(row_id)s").format(
            sql.Identifier(table), sql.Identifier(column)
        )
        conn.execute(query, {"moment": moment, "row_id": row_id})
    finally:
        conn.close()
