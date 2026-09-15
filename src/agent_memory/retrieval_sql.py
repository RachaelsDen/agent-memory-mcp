"""SQL contracts for the retrieval pipeline (DESIGN S7, task-2 query contracts).

Every channel/visibility/provenance query used by agent_memory.retrieval.
Shared visibility fragment: episodes are scoped ``namespace = ns``; lessons
union in ``namespace = 'global'`` copies but ONLY with
``promotion_status = 'active'`` — a demoted copy is invisible even when the
request IS the 'global' namespace.

ts_rank() floors at 1e-20 even for NON-matching rows, which would leak
through the pure gate's ``ts_rank > 0``; the full-text match predicate must
therefore gate the hydrated rank itself (plan task 7).
"""

LessonVisibility = "((namespace = %(ns)s OR namespace = 'global') AND promotion_status = 'active')"

MatchedRank = """CASE WHEN search_tsv @@ websearch_to_tsquery('english', %(q)s)
                   THEN ts_rank(search_tsv, websearch_to_tsquery('english', %(q)s))
                   ELSE 0 END"""

KEYWORD_SQL = f"""
SELECT record_type, id, salience, cosine, rank, evidence_ts, last_accessed FROM (
    (
        SELECT 'episode' AS record_type, id, surprise AS salience,
               1 - (embedding <=> %(qvec)s) AS cosine,
               ts_rank(search_tsv, websearch_to_tsquery('english', %(q)s)) AS rank,
               created_at AS evidence_ts, last_accessed
        FROM episodes
        WHERE namespace = %(ns)s
          AND embedding IS NOT NULL
          AND search_tsv @@ websearch_to_tsquery('english', %(q)s)
          AND ts_rank(search_tsv, websearch_to_tsquery('english', %(q)s)) > 0
    )
    UNION ALL
    (
        SELECT 'lesson' AS record_type, id, confidence AS salience,
               1 - (embedding <=> %(qvec)s) AS cosine,
               ts_rank(search_tsv, websearch_to_tsquery('english', %(q)s)) AS rank,
               last_evidence_at AS evidence_ts, last_accessed
        FROM lessons
        WHERE {LessonVisibility}
          AND embedding IS NOT NULL
          AND search_tsv @@ websearch_to_tsquery('english', %(q)s)
          AND ts_rank(search_tsv, websearch_to_tsquery('english', %(q)s)) > 0
    )
) merged
ORDER BY rank DESC, record_type ASC, id ASC
LIMIT %(topk)s
"""

VECTOR_SQL = f"""
SELECT record_type, id, salience, cosine, rank, evidence_ts, last_accessed FROM (
    (
        SELECT 'episode' AS record_type, id, surprise AS salience,
               1 - (embedding <=> %(qvec)s) AS cosine,
               {MatchedRank} AS rank,
               created_at AS evidence_ts, last_accessed
        FROM episodes
        WHERE namespace = %(ns)s AND embedding IS NOT NULL
        ORDER BY embedding <=> %(qvec)s ASC
        LIMIT %(topk)s
    )
    UNION ALL
    (
        SELECT 'lesson' AS record_type, id, confidence AS salience,
               1 - (embedding <=> %(qvec)s) AS cosine,
               {MatchedRank} AS rank,
               last_evidence_at AS evidence_ts, last_accessed
        FROM lessons
        WHERE {LessonVisibility} AND embedding IS NOT NULL
        ORDER BY embedding <=> %(qvec)s ASC
        LIMIT %(topk)s
    )
) merged
ORDER BY cosine DESC, record_type ASC, id ASC
LIMIT %(topk)s
"""

NEIGHBOR_SQL = f"""
SELECT ll.lesson_id AS source_id, ll.related_lesson_id AS target_id, ll.weight,
       l.confidence AS salience,
       1 - (l.embedding <=> %(qvec)s) AS cosine,
       CASE WHEN l.search_tsv @@ websearch_to_tsquery('english', %(q)s)
            THEN ts_rank(l.search_tsv, websearch_to_tsquery('english', %(q)s))
            ELSE 0 END AS rank,
       l.last_evidence_at AS evidence_ts, l.last_accessed
FROM lesson_links ll
JOIN lessons l ON l.id = ll.related_lesson_id
WHERE ll.lesson_id = ANY(%(lesson_ids)s)
  AND ((l.namespace = %(ns)s OR l.namespace = 'global') AND l.promotion_status = 'active')
  AND l.embedding IS NOT NULL
"""

EPISODE_CONTENT_SQL = """
SELECT id, namespace, goal, expectation, action, outcome, surprise, tags, created_at
FROM episodes WHERE id = ANY(%(ids)s)
"""

LESSON_CONTENT_SQL = """
SELECT id, namespace, claim, because, holds_when, fails_when, confidence,
       disputed, created_at, last_evidence_at
FROM lessons WHERE id = ANY(%(ids)s)
"""

EVIDENCE_SQL = """
SELECT lesson_id, episode_id, relation, reason, e.outcome
FROM lesson_evidence le JOIN episodes e ON e.id = le.episode_id
WHERE le.lesson_id = ANY(%(ids)s)
ORDER BY episode_id
"""

INSERT_EVENT_SQL = """
INSERT INTO retrieval_events (namespace, tool, query_context, returned_ids)
VALUES (%(ns)s, %(tool)s, %(qc)s, %(ids)s)
RETURNING id
"""
