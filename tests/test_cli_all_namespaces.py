"""CLI ``--all-namespaces`` contract for digest / consolidate-scan (cron coverage).

Every test drives the real dispatcher (``python -m agent_memory``, same module
as the console script) as a subprocess against the testcontainer database,
mirroring test_entrypoint's ``_run_cli`` pattern. Enumeration under test is
the SQL union of distinct namespaces in episodes and lessons plus the literal
``global`` whenever any memory exists anywhere — so a lessons-only namespace
enumerates, and ``global`` digests even while it holds no rows of its own,
while a fully empty database enumerates nothing (graceful empty-DB cron).
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pgvector
import psycopg
import yaml
from psycopg.rows import DictRow

from tests.conftest import backdate

CLI_ENV_NAMESPACE = "env-ns@proj"

ALPHA = "alpha-ns@proj"  # episodes + a lesson
BETA = "beta-ns@proj"  # episodes only
GAMMA = "gamma-lessons@proj"  # lesson only: proves the lessons arm of the union
# Fixed under both C and en_US collations: alpha < beta < gamma < global.
EXPECTED_NAMESPACES = [ALPHA, BETA, GAMMA, "global"]


def _run_cli(
    pg: str, *args: str, env_extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Real entrypoint with CLI args; MEMORY_NAMESPACE is pinned so every call
    also proves the env tier never leaks into the enumeration."""
    env = {
        **os.environ,
        "DATABASE_URL": pg,
        "EMBED_IMPL": "fake",
        "PGVECTOR_DIM": "8",
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


def _insert_episode(db: psycopg.Connection[DictRow], namespace: str, goal: str) -> int:
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


def _insert_lesson(db: psycopg.Connection[DictRow], namespace: str, claim: str) -> int:
    """Direct-SQL lesson fixture (embedding is not read by digest or the scan)."""
    row = db.execute(
        """
        INSERT INTO lessons (namespace, claim, because, embedding)
        VALUES (%(namespace)s, %(claim)s, 'because fixture', %(embedding)s)
        RETURNING id
        """,
        {
            "namespace": namespace,
            "claim": claim,
            "embedding": pgvector.Vector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        },
    ).fetchone()
    assert row is not None
    return int(row["id"])


def _seed(db: psycopg.Connection[DictRow]) -> None:
    """Three namespaces: alpha (2 backdated near-duplicate episodes + a lesson),
    beta (one lone episode), gamma (lesson only — no episodes at all)."""
    for goal in ("first near-duplicate", "second near-duplicate"):
        episode_id = _insert_episode(db, ALPHA, goal)
        backdate("episodes", episode_id, hours=-2)  # past the age gate
    _insert_episode(db, BETA, "lone episode")
    _insert_lesson(db, ALPHA, "alpha lesson claim")
    _insert_lesson(db, GAMMA, "gamma lesson claim")


def test_digest_all_namespaces_renders_one_file_per_namespace(
    db: psycopg.Connection[DictRow], pg: str, tmp_path: Path
) -> None:
    """One '<ns>\t<path>\t<flagged_count>' stdout line per enumerated namespace,
    a real digest file per line, and no env-namespace leakage (exact list)."""
    _seed(db)
    digest_dir = tmp_path / "digests"

    ran = _run_cli(
        pg, "digest", "--all-namespaces", env_extra={"DIGEST_DIR": str(digest_dir)}
    )
    assert ran.returncode == 0, ran.stderr

    lines = ran.stdout.splitlines()
    assert [line.split("\t")[0] for line in lines] == EXPECTED_NAMESPACES
    for line in lines:
        namespace, path_text, flagged = line.split("\t")
        path = Path(path_text)
        assert path.is_file(), path
        assert path.parent == digest_dir
        assert flagged.isdigit()
        frontmatter, _, _ = path.read_text(encoding="utf-8").partition("\n---\n")
        meta: dict[str, Any] = yaml.safe_load(frontmatter.removeprefix("---\n"))
        assert meta["namespace"] == namespace


def test_consolidate_scan_all_namespaces_single_json(
    db: psycopg.Connection[DictRow], pg: str
) -> None:
    """One JSON object {"namespaces": [per-namespace scan payloads]}; each entry
    carries the single-namespace payload shape keyed by its namespace."""
    _seed(db)

    ran = _run_cli(pg, "consolidate-scan", "--all-namespaces")
    assert ran.returncode == 0, ran.stderr

    payload = json.loads(ran.stdout)
    assert set(payload) == {"namespaces"}
    entries = payload["namespaces"]
    assert [entry["namespace"] for entry in entries] == EXPECTED_NAMESPACES
    for entry in entries:
        assert set(entry) == {"pool", "namespace", "clusters", "rederivation_groups"}
        assert entry["pool"] == "fresh"
    alpha = entries[EXPECTED_NAMESPACES.index(ALPHA)]
    assert len(alpha["clusters"]) == 1  # identical vectors -> one cluster
    assert len(alpha["clusters"][0]["episodes"]) == 2
    assert entries[EXPECTED_NAMESPACES.index("global")]["clusters"] == []


def test_all_namespaces_and_namespace_flag_are_mutually_exclusive(
    db: psycopg.Connection[DictRow], pg: str
) -> None:
    """--namespace + --all-namespaces is a usage error on both subcommands."""
    for subcommand in ("digest", "consolidate-scan"):
        ran = _run_cli(pg, "--namespace", "any@proj", subcommand, "--all-namespaces")
        assert ran.returncode != 0, subcommand
        assert "usage:" in ran.stderr, subcommand
        assert "mutually exclusive" in ran.stderr, subcommand


def test_all_namespaces_on_empty_database_is_graceful(
    db: psycopg.Connection[DictRow], pg: str, tmp_path: Path
) -> None:
    """Zero namespaces enumerated: digest prints nothing and exits 0; the scan
    prints {"namespaces": []}."""
    digest_dir = tmp_path / "digests"

    ran = _run_cli(
        pg, "digest", "--all-namespaces", env_extra={"DIGEST_DIR": str(digest_dir)}
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == ""

    scan = _run_cli(pg, "consolidate-scan", "--all-namespaces")
    assert scan.returncode == 0, scan.stderr
    assert json.loads(scan.stdout) == {"namespaces": []}


def test_digest_all_namespaces_reports_per_namespace_errors_and_exits_nonzero(
    db: psycopg.Connection[DictRow], pg: str, tmp_path: Path
) -> None:
    """An unusable DIGEST_DIR fails every namespace individually: one ERROR line
    each, the loop keeps going, and the final exit is nonzero."""
    _seed(db)
    blocker = tmp_path / "blocker"
    blocker.write_text("a regular file, so the mkdir under it fails")

    ran = _run_cli(
        pg,
        "digest",
        "--all-namespaces",
        env_extra={"DIGEST_DIR": str(blocker / "digests")},
    )
    assert ran.returncode == 1
    lines = ran.stdout.splitlines()
    assert [line.split("\t")[0] for line in lines] == EXPECTED_NAMESPACES
    assert all("\tERROR:" in line for line in lines)
