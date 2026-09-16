"""F2 fix contract: db.connect() must not leak its connection on setup failure.

connect() performs two post-open steps (pgvector adapter registration, the
``SET hnsw.ef_search`` session command) BEFORE returning the connection to
the caller. If either raises, the caller never receives the connection, so
closing it is connect()'s own responsibility — verified here with NO
database involved: ``psycopg.Connection.connect`` is monkeypatched to return
a stub whose setup step raises, and the assertions watch the stub's
``close()``. The runtime generic alias ``psycopg.Connection[DictRow]``
resolves ``.connect`` through the origin class, so the class-level patch is
what db.connect() itself observes.
"""

import pgvector.psycopg
import psycopg
import pytest

from agent_memory import db
from agent_memory.config import Settings


class _SetupFailed(Exception):
    """The injected post-open failure."""


class _StubConnection:
    """Stand-in for the opened connection; records close(), fails the SET."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def execute(self, query: object, params: object = None) -> None:
        raise _SetupFailed("session SET failed")


def _patch_open(monkeypatch: pytest.MonkeyPatch, stub: _StubConnection) -> None:
    # Hermetic Settings: no get_settings() cache or DATABASE_URL env involved.
    monkeypatch.setattr(
        db, "get_settings", lambda: Settings(DATABASE_URL="postgresql://leak-stub")
    )
    monkeypatch.setattr(psycopg.Connection, "connect", lambda *args, **kwargs: stub)


def test_register_vector_failure_closes_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _StubConnection()
    _patch_open(monkeypatch, stub)

    def register_boom(conn: object) -> None:
        raise _SetupFailed("register_vector failed")

    monkeypatch.setattr(pgvector.psycopg, "register_vector", register_boom)

    with pytest.raises(_SetupFailed):
        db.connect()

    assert stub.closed is True


def test_ef_search_set_failure_closes_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _StubConnection()
    _patch_open(monkeypatch, stub)
    monkeypatch.setattr(pgvector.psycopg, "register_vector", lambda conn: None)

    with pytest.raises(_SetupFailed):
        db.connect()

    assert stub.closed is True
