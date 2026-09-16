"""Postgres connections and the migration runner.

Two connection flavors, on purpose:

- ``connect()``: application connections. Registers the pgvector adapter
  (type registration alone does NOT adapt plain Python lists; every vector
  value, column or parameter, must be wrapped explicitly as
  ``pgvector.Vector(...)``) and raises ``hnsw.ef_search`` for recall.
- ``connect_bootstrap()``: migrations ONLY. No vector registration and no
  ef_search, because the ``vector`` type does not exist until migration 001
  creates the extension (registering an adapter for a missing type fails on
  a cold database). Opened with EXPLICIT ``autocommit=True``: in psycopg's
  default transaction mode the advisory-lock SELECT would silently start an
  outer transaction, demoting every ``with conn.transaction():`` below to a
  savepoint and rolling all migrations back when the connection closes.
"""

from pathlib import Path

import pgvector.psycopg
import psycopg
from psycopg.rows import DictRow, dict_row

from agent_memory.config import get_settings

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_LOCK_KEY = "agent-memory-migrations"


def connect() -> psycopg.Connection[DictRow]:
    """Application connection: dict rows, pgvector adapter, ef_search=100."""
    conn = psycopg.Connection[DictRow].connect(
        get_settings().DATABASE_URL, autocommit=True, row_factory=dict_row
    )
    try:
        pgvector.psycopg.register_vector(conn)
        conn.execute("SET hnsw.ef_search = 100")
    except BaseException:
        # The caller never sees this connection, so a failed setup step must
        # not leak it (F2): close before re-raising.
        conn.close()
        raise
    return conn


def connect_bootstrap() -> psycopg.Connection:
    """Migration connection: plain rows, no vector adapter, no ef_search."""
    return psycopg.connect(get_settings().DATABASE_URL, autocommit=True)


def migrate(dim: int) -> list[str]:
    """Apply pending migrations; returns the filenames applied (in order).

    Holds a session-level advisory lock for the whole run, OUTSIDE each
    per-migration transaction, so concurrent migrators serialize. State is a
    filename comparison against the bookkeeping table, never a mere
    table-existence check.
    """
    conn = connect_bootstrap()
    try:
        conn.execute("SELECT pg_advisory_lock(hashtext(%(key)s))", {"key": _LOCK_KEY})
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_memory_migrations (
                name TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        applied = {
            row[0] for row in conn.execute("SELECT name FROM agent_memory_migrations").fetchall()
        }
        pending = [
            path for path in sorted(MIGRATIONS_DIR.glob("*.sql")) if path.name not in applied
        ]
        for path in pending:
            # psycopg's Query type only admits literal strings or bytes, and
            # sql.SQL()'s own input is typed LiteralString too; file SQL is
            # computed text, so it must cross as bytes (same simple-protocol
            # path, still multi-statement capable).
            migration_sql = path.read_text().replace("__DIM__", str(dim)).encode()
            with conn.transaction():
                conn.execute(migration_sql)
                conn.execute(
                    "INSERT INTO agent_memory_migrations (name) VALUES (%(name)s)",
                    {"name": path.name},
                )
        return [path.name for path in pending]
    finally:
        # Unlock first, but ALWAYS close: an abandoned open connection still
        # holding the session lock would block migrate() from every other host.
        try:
            conn.execute("SELECT pg_advisory_unlock(hashtext(%(key)s))", {"key": _LOCK_KEY})
        finally:
            conn.close()
