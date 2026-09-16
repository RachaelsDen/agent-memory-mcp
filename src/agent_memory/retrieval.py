"""DB-facing retrieval pipeline for memory_probe / memory_search (DESIGN S7).

Owns typed-ref assembly, the activation neighbor expansion, the
retrieval_events exposure row, and the provenance payload. Every piece of
ranking math lives in ``agent_memory.retrieve`` — this module only feeds it
Candidate/Edge values and never reimplements a formula; the SQL lives in
``agent_memory.retrieval_sql``. Access stats are never written here (P3).
"""

from datetime import datetime, timezone
from typing import Any, Literal

import pgvector
import psycopg
from psycopg.rows import DictRow

from agent_memory import db
from agent_memory.config import Settings
from agent_memory.embed import load_embedder
from agent_memory.retrieve import (
    Candidate,
    Edge,
    Scored,
    ScoringParams,
    activation_hop,
    rel_norm,
    rrf,
)
from agent_memory.retrieval_sql import (
    EPISODE_CONTENT_SQL,
    EVIDENCE_SQL,
    INSERT_EVENT_SQL,
    KEYWORD_SQL,
    LESSON_CONTENT_SQL,
    NEIGHBOR_SQL,
    VECTOR_SQL,
)

EXCERPT_CHARS = 140


def _candidate(row: DictRow) -> Candidate:
    """One channel row -> Candidate; both channel queries compute both signals."""
    return Candidate(
        record_type=row["record_type"],
        record_id=int(row["id"]),
        salience=float(row["salience"]),
        cosine=float(row["cosine"]),
        ts_rank=float(row["rank"]),
        evidence_ts=row["evidence_ts"],
        last_accessed=row["last_accessed"],
    )


def _scoring_params(settings: Settings) -> ScoringParams:
    return ScoringParams(
        w_rel=settings.W_REL,
        w_sal=settings.W_SAL,
        w_env=settings.W_ENV,
        w_use=settings.W_USE,
        w_spread=settings.W_SPREAD,
        sim_floor=settings.SIM_FLOOR,
        ts_rank_sat=settings.TS_RANK_SAT,
        tau_env_h=settings.TAU_ENV_H,
        tau_use_h=settings.TAU_USE_H,
        probe_topk=settings.PROBE_TOPK,
    )


def run_retrieval(
    settings: Settings,
    *,
    tool: Literal["probe", "search"],
    query_text: str,
    k: int | None,
    namespace: str | None,
) -> dict[str, Any]:
    """Full hybrid pipeline; the kwargs mirror the two tool signatures verbatim."""
    effective_ns = settings.MEMORY_NAMESPACE if namespace is None else namespace
    final_k = settings.FINAL_K if k is None else k
    params = _scoring_params(settings)
    query_vector = pgvector.Vector(load_embedder(settings).embed([query_text])[0])
    bind = {"q": query_text, "ns": effective_ns, "qvec": query_vector, "topk": params.probe_topk}

    connection: psycopg.Connection[DictRow] = db.connect()
    try:
        keyword_rows = connection.execute(KEYWORD_SQL, bind).fetchall()
        vector_rows = connection.execute(VECTOR_SQL, bind).fetchall()

        candidates: dict[str, Candidate] = {}
        for row in keyword_rows + vector_rows:
            candidate = _candidate(row)
            candidates[candidate.ref] = candidate
        keyword_ranking = [f"{row['record_type']}:{row['id']}" for row in keyword_rows]
        vector_ranking = [f"{row['record_type']}:{row['id']}" for row in vector_rows]
        relevance = rel_norm(rrf([keyword_ranking, vector_ranking]))

        edges: list[Edge] = []
        lesson_ids = [
            candidate.record_id
            for candidate in candidates.values()
            if candidate.record_type == "lesson"
        ]
        if lesson_ids:
            for row in connection.execute(
                NEIGHBOR_SQL, {**bind, "lesson_ids": lesson_ids}
            ).fetchall():
                edges.append(
                    Edge(
                        source=f"lesson:{row['source_id']}",
                        target=f"lesson:{row['target_id']}",
                        weight=float(row["weight"]),
                    )
                )
                target_ref = f"lesson:{row['target_id']}"
                if target_ref not in candidates:
                    candidates[target_ref] = _candidate(row)

        finals = activation_hop(
            candidates, relevance, edges, params, datetime.now(timezone.utc)
        )[:final_k]
        refs = [item.ref for item in finals]

        event = connection.execute(
            INSERT_EVENT_SQL,
            {"ns": effective_ns, "tool": tool, "qc": query_text, "ids": refs},
        ).fetchone()
        assert event is not None  # INSERT ... RETURNING always yields exactly one row
        results = _result_records(connection, finals, candidates, settings)
    finally:
        connection.close()
    return {"retrieval_event_id": event["id"], "results": results}


def _result_records(
    connection: psycopg.Connection[DictRow],
    finals: list[Scored],
    candidates: dict[str, Candidate],
    settings: Settings,
) -> list[dict[str, Any]]:
    """Content + provenance for the final records, in final order."""
    episode_ids = [item.record_id for item in finals if item.record_type == "episode"]
    lesson_ids = [item.record_id for item in finals if item.record_type == "lesson"]
    episodes = (
        {
            row["id"]: row
            for row in connection.execute(EPISODE_CONTENT_SQL, {"ids": episode_ids})
        }
        if episode_ids
        else {}
    )
    lessons = (
        {
            row["id"]: row
            for row in connection.execute(LESSON_CONTENT_SQL, {"ids": lesson_ids})
        }
        if lesson_ids
        else {}
    )
    evidence_by_lesson: dict[int, list[DictRow]] = dict.fromkeys(lesson_ids, [])
    if lesson_ids:
        for row in connection.execute(EVIDENCE_SQL, {"ids": lesson_ids}):
            evidence_by_lesson[int(row["lesson_id"])].append(row)

    records: list[dict[str, Any]] = []
    for item in finals:
        if item.record_type == "episode":
            records.append(_episode_record(episodes[item.record_id], item))
        else:
            records.append(
                _lesson_record(lessons[item.record_id], item, evidence_by_lesson[item.record_id])
            )
        salience = candidates[item.ref].salience
        if item.env_fresh < settings.STALE_ENV_FRESH and salience > settings.SALIENCE_STALE:
            records[-1]["staleness_note"] = (
                f"high-salience record with stale evidence: env_fresh {item.env_fresh:.3f} "
                f"< STALE_ENV_FRESH {settings.STALE_ENV_FRESH} while salience {salience:.2f} "
                f"> SALIENCE_STALE {settings.SALIENCE_STALE}"
            )
    return records


def _episode_record(row: DictRow, item: Scored) -> dict[str, Any]:
    return {
        "id": f"episode:{row['id']}",
        "record_type": "episode",
        "namespace": row["namespace"],
        "goal": row["goal"],
        "expectation": row["expectation"],
        "action": row["action"],
        "outcome": row["outcome"],
        "surprise": row["surprise"],
        "tags": row["tags"],
        "created_at": row["created_at"].isoformat(),
        "match_strength": item.match_strength,
    }


def _lesson_record(row: DictRow, item: Scored, evidence_rows: list[DictRow]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": f"lesson:{row['id']}",
        "record_type": "lesson",
        "namespace": row["namespace"],
        "claim": row["claim"],
        "because": row["because"],
        "holds_when": row["holds_when"],
        "fails_when": row["fails_when"],
        "confidence": row["confidence"],
        "created_at": row["created_at"].isoformat(),
        "last_evidence_at": row["last_evidence_at"].isoformat(),
        "match_strength": item.match_strength,
        "evidence": [
            {
                "episode": f"episode:{edge['episode_id']}",
                "relation": edge["relation"],
                "reason": edge["reason"],
                "excerpt": (edge["outcome"] or "")[:EXCERPT_CHARS],
            }
            for edge in evidence_rows
        ],
    }
    if row["disputed"]:
        record["disputed"] = True
        record["dispute_reason"] = row["dispute_reason"]
    return record
