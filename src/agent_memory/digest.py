"""Markdown digest renderer for memory_digest (DESIGN S9, S10).

Renders ``<DIGEST_DIR>/<slug>-<sha256(namespace)[:8]>-<YYYY-MM-DD>.md`` where
``slug`` maps every character outside ``[A-Za-z0-9._-]`` to ``_``. Collisions
on slug+hash (rare but real) never overwrite a file whose stored frontmatter
namespace differs: the name deterministically extends with a retry hash
sha256(``namespace + '#' + n``)[:8] while the ``<slug>-<basehash>`` prefix is
never rewritten, so this namespace's files (collision-resolved ones included)
stay discoverable by the prefix scan that locates the LATEST VALID PRIOR
DIGEST across ALL dates — same-day included, so the weekly cron never
regresses to the fallback just because the date rolled over.

Section 2 flags lessons whose live confidence < snapshot - 1e-6 — REAL
storage precision: the snapshot serializes the float64 of the stored float32
with full repr precision (a floor-adjacent 0.055 -> 0.05 drop of 0.005 still
flags), and the snapshot is READ before the replacement digest is written.
With no valid snapshot anywhere, the fallback lists lessons carrying any
``contradict`` edge. Sections 3-5 and the backlog reuse ``oversight.stats``
queries verbatim (single source of thresholds). Frontmatter is YAML via
``yaml.safe_dump`` — never f-string interpolated — and the resolved path is
verified to remain under DIGEST_DIR.

The rendered document is screened for credential patterns before the
write (Issue #9 defense in depth): a hit names 'digest content' (never
the matched text) and NOTHING is written.
"""

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, LiteralString

import psycopg
import yaml
from psycopg.rows import DictRow

try:  # mcp>=2 renamed FastMCP to MCPServer; keep both import spellings working
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # fallback targets mcp 1.x, which is not installed here
    from mcp.server.fastmcp.exceptions import ToolError  # pyright: ignore[reportMissingImports]

from agent_memory import db
from agent_memory.config import Settings
from agent_memory.oversight import stats as oversight_stats
from agent_memory.secrets import screen_document
from agent_memory.session_state import resolve_namespace

EXCERPT_CHARS = 140
SNAPSHOT_TOLERANCE = 1e-6

_SLUG_SUBSTITUTE = "_"
_SLUG_PATTERN = re.compile(r"[^A-Za-z0-9._-]")
_REMAINDER_PATTERN = re.compile(r"^(?:[0-9a-f]{8}-)?(\d{4}-\d{2}-\d{2})\.md$")

DISPUTED_SQL = """
    SELECT id, claim, dispute_reason
    FROM lessons
    WHERE namespace = %(ns)s AND disputed
    ORDER BY id
"""

DISPUTED_EXCERPTS_SQL = """
    SELECT le.lesson_id, le.episode_id, le.relation, e.outcome
    FROM lesson_evidence le
    JOIN episodes e ON e.id = le.episode_id
    WHERE le.lesson_id = ANY(%(ids)s)
    ORDER BY le.lesson_id, le.episode_id
"""

LIVE_LESSONS_SQL = """
    SELECT id, claim, confidence
    FROM lessons
    WHERE namespace = %(ns)s
    ORDER BY id
"""

CONTRADICT_EDGE_SQL = """
    SELECT DISTINCT l.id, l.claim
    FROM lessons l
    JOIN lesson_evidence le
      ON le.lesson_id = l.id AND le.relation = 'contradict'
    WHERE l.namespace = %(ns)s
    ORDER BY l.id
"""


def _hash8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _read_frontmatter(path: Path) -> dict[str, Any] | None:
    """Parse the leading YAML block; None when missing or unparseable."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for end in range(1, len(lines)):
        if lines[end].strip() == "---":
            try:
                data = yaml.safe_load("\n".join(lines[1:end]))
            except yaml.YAMLError:
                return None
            return data if isinstance(data, dict) else None
    return None


def _resolve_target(root: Path, namespace: str, today: str) -> Path:
    """Base name for this namespace today, retry-hash extended on collision
    with a different namespace's file; never rewrites the stem prefix."""
    stem = f"{_SLUG_PATTERN.sub(_SLUG_SUBSTITUTE, namespace)}-{_hash8(namespace)}"
    candidate = root / f"{stem}-{today}.md"
    attempt = 0
    while candidate.exists():
        data = _read_frontmatter(candidate)
        if data is not None and data.get("namespace") == namespace:
            return candidate
        attempt += 1
        retry_hash = _hash8(f"{namespace}#{attempt}")
        candidate = root / f"{stem}-{retry_hash}-{today}.md"
    return candidate


def _find_snapshot(root: Path, prefix: str, namespace: str) -> dict[int, float] | None:
    """Confidence snapshot from the latest VALID prior digest (any date).

    Candidates are every file carrying this namespace's slug-hash prefix —
    which by construction also matches collision-resolved retry files — and a
    candidate is valid only when its frontmatter namespace matches and its
    snapshot parses.
    """
    dated: list[tuple[str, str]] = []
    for entry in sorted(root.iterdir()):
        if not entry.name.startswith(prefix):
            continue
        matched = _REMAINDER_PATTERN.match(entry.name[len(prefix) :])
        if matched is not None and entry.is_file():
            dated.append((matched.group(1), entry.name))
    for _, name in sorted(dated, reverse=True):
        data = _read_frontmatter(root / name)
        if data is None or data.get("namespace") != namespace:
            continue
        raw = data.get("confidence_snapshot")
        if not isinstance(raw, dict):
            continue
        try:
            return {int(key): float(value) for key, value in raw.items()}
        except (TypeError, ValueError):
            continue
    return None


def _section(number: int, title: str, entries: list[str]) -> list[str]:
    body = entries if entries else ["(none)"]
    return [f"## {number}. {title}", "", *body, ""]


def digest(settings: Settings, *, namespace: str | None = None) -> dict[str, Any]:
    """Render the six-section audit digest; None namespace -> resolved default."""
    effective_ns = resolve_namespace(settings, namespace)
    root = Path(settings.DIGEST_DIR).expanduser()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ToolError(
            f"DIGEST_DIR is not a usable directory: {settings.DIGEST_DIR!r}"
            f" ({error.strerror or error})"
        ) from None

    stem = f"{_SLUG_PATTERN.sub(_SLUG_SUBSTITUTE, effective_ns)}-{_hash8(effective_ns)}"
    today = datetime.now(timezone.utc).date().isoformat()
    target = _resolve_target(root, effective_ns, today)
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError:
        raise ToolError(f"digest path escapes DIGEST_DIR: {target}") from None

    # Read BEFORE the replacement digest is written (the same-day target may
    # be exactly the file being replaced).
    snapshot = _find_snapshot(root, f"{stem}-", effective_ns)

    connection: psycopg.Connection[DictRow] = db.connect()
    try:
        disputed_rows = connection.execute(DISPUTED_SQL, {"ns": effective_ns}).fetchall()
        excerpts: dict[int, list[tuple[int, str, str]]] = {}
        if disputed_rows:
            edge_rows = connection.execute(
                DISPUTED_EXCERPTS_SQL, {"ids": [int(row["id"]) for row in disputed_rows]}
            ).fetchall()
            for edge in edge_rows:
                lesson_id = int(edge["lesson_id"])
                excerpt = (edge["outcome"] or "")[:EXCERPT_CHARS]
                excerpts.setdefault(lesson_id, []).append(
                    (int(edge["episode_id"]), str(edge["relation"]), excerpt)
                )
        live_rows = connection.execute(LIVE_LESSONS_SQL, {"ns": effective_ns}).fetchall()
        fallback_rows = (
            None
            if snapshot is not None
            else connection.execute(CONTRADICT_EDGE_SQL, {"ns": effective_ns}).fetchall()
        )
    finally:
        connection.close()
    stats_payload = oversight_stats(settings, namespace=effective_ns)

    flagged: list[str] = []
    section1: list[str] = []
    for row in disputed_rows:
        lesson_id = int(row["id"])
        reason = row["dispute_reason"] or ""
        flagged.append(f"disputed [[lesson:{lesson_id}]] {row['claim']} - reason: {reason}")
        section1.append(f"- [[lesson:{lesson_id}]] {row['claim']} - reason: {reason}")
        for episode_id, relation, excerpt in excerpts.get(lesson_id, []):
            section1.append(f"  - [[episode:{episode_id}]] {relation}: {excerpt}")

    section2: list[str] = []
    if snapshot is not None:
        for row in live_rows:
            lesson_id = int(row["id"])
            live_confidence = float(row["confidence"])
            prior = snapshot.get(lesson_id)
            if prior is not None and live_confidence < prior - SNAPSHOT_TOLERANCE:
                entry = (
                    f"contradicted [[lesson:{lesson_id}]] {row['claim']}"
                    f" - confidence {prior:.4f} -> {live_confidence:.4f}"
                )
                flagged.append(entry)
                section2.append(f"- {entry}")
    else:
        for row in fallback_rows or []:
            lesson_id = int(row["id"])
            entry = (
                f"contradicted [[lesson:{lesson_id}]] {row['claim']}"
                " - carries contradict evidence"
            )
            flagged.append(entry)
            section2.append(f"- {entry}")

    section3 = [
        f"- popular-but-shaky [[{entry['id']}]] {entry['claim']}"
        f" - confidence {entry['confidence']}, access_count {entry['access_count']}"
        for entry in stats_payload["popular_but_shaky"]
    ]
    flagged.extend(line[2:] for line in section3)
    section4 = [
        f"- rare-critical-stale [[{entry['id']}]] {entry['claim']}"
        f" - confidence {entry['confidence']}, last evidence {entry['last_evidence_at']}"
        for entry in stats_payload["rare_critical_stale"]
    ]
    flagged.extend(line[2:] for line in section4)
    section5 = [
        f"- demoted [[{entry['id']}]] {entry['claim']} - reason: {entry['demotion_reason']}"
        for entry in stats_payload["demoted_promotions"]
    ]
    flagged.extend(line[2:] for line in section5)

    section6 = [f"{stats_payload['unconsolidated_backlog']} episode(s) awaiting consolidation.", ""]
    section6.append("### Lessons")
    for row in live_rows:
        section6.append(
            f"- [[lesson:{row['id']}]] {row['claim']}"
            f" (confidence {float(row['confidence']):.2f})"
        )

    frontmatter = {
        "namespace": effective_ns,
        "confidence_snapshot": {str(row["id"]): float(row["confidence"]) for row in live_rows},
    }
    document = [
        "---",
        yaml.safe_dump(frontmatter, default_flow_style=False, sort_keys=False).rstrip("\n"),
        "---",
        "",
        f"# Memory digest - {effective_ns}",
        "",
        f"Generated: {today}",
        "",
        *_section(1, "Disputed", section1),
        *_section(2, "Recently contradicted", section2),
        *_section(3, "Popular-but-shaky", section3),
        *_section(4, "Rare-critical-stale", section4),
        *_section(5, "Recently demoted promotions", section5),
        *_section(6, "Unconsolidated backlog", section6),
    ]
    text = "\n".join(document) + "\n"
    screen_document(text)
    try:
        target.write_text(text, encoding="utf-8")
    except OSError as error:
        raise ToolError(
            f"cannot write digest file {target} ({error.strerror or error})"
        ) from None
    return {"path": str(target), "flagged": flagged, "flagged_count": len(flagged)}
