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

Migration 003's claim-embedding backfill runs as a PYTHON step inside
migrate(), not as SQL (design choice, Issue #6): computing embeddings needs
the embedder, which SQL cannot invoke. The step is gated on 003 being in the
bookkeeping (just applied or earlier) and touches only rows whose
claim_embedding is still NULL, so it is idempotent and a no-op on fresh
databases. It runs while the migration advisory lock is held, so concurrent
migrators cannot double-embed; a crashed run resumes on the next migrate().
"""

from pathlib import Path

import pgvector.psycopg
import psycopg
from psycopg.rows import DictRow, dict_row

from agent_memory.config import get_settings
from agent_memory.embed import load_embedder

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_LOCK_KEY = "agent-memory-migrations"
CLAIM_EMBEDDING_MIGRATION = "003_claim_embedding.sql"

_CLAIM_BACKFILL_SELECT_SQL = """
    SELECT id, claim FROM lessons WHERE claim_embedding IS NULL ORDER BY id
"""

_CLAIM_BACKFILL_UPDATE_SQL = """
    UPDATE lessons SET claim_embedding = %(vector)s WHERE id = %(id)s
"""


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


def _backfill_claim_embeddings(conn: psycopg.Connection) -> None:
    """Embed `claim` for lessons still lacking a claim_embedding (003 step)."""
    rows = conn.execute(_CLAIM_BACKFILL_SELECT_SQL).fetchall()
    if not rows:
        return
    # The vector type now exists (001 applied), so the adapter is safe to
    # register on this bootstrap connection for the UPDATE parameters.
    pgvector.psycopg.register_vector(conn)
    embedder = load_embedder(get_settings())
    vectors = embedder.embed([str(row[1]) for row in rows])
    for row, vector in zip(rows, vectors, strict=True):
        conn.execute(
            _CLAIM_BACKFILL_UPDATE_SQL,
            {"id": row[0], "vector": pgvector.Vector(vector)},
        )


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
        if CLAIM_EMBEDDING_MIGRATION in applied | {path.name for path in pending}:
            _backfill_claim_embeddings(conn)
        return [path.name for path in pending]
    finally:
        # Unlock first, but ALWAYS close: an abandoned open connection still
        # holding the session lock would block migrate() from every other host.
        try:
            conn.execute("SELECT pg_advisory_unlock(hashtext(%(key)s))", {"key": _LOCK_KEY})
        finally:
            conn.close()
