-- Migration 001: initial schema (DESIGN.md §5, verbatim column names/comments).
-- __DIM__ is replaced with settings.PGVECTOR_DIM by the migration runner.
-- Sanctioned deviations from §5 (review-fixed):
--   * trailing comma after episodes.last_accessed removed (DDL would fail)
--   * retrieval_events.returned_ids is TEXT[] and usage_reports.record_id is
--     TEXT, storing typed refs "episode:<id>"/"lesson:<id>" — a BIGINT[]
--     cannot disambiguate lesson 5 from episode 5 in one array; record_type
--     stays as a derived convenience.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE episodes (
  id               BIGSERIAL PRIMARY KEY,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  namespace        TEXT NOT NULL,               -- agent × project scope (see §11)
  goal             TEXT,                        -- what I was trying to do
  expectation      TEXT,                        -- what I believed would happen
  action           TEXT,                        -- what I did
  outcome          TEXT,                        -- what actually happened
  surprise         REAL NOT NULL DEFAULT 0.0,   -- 0..1 prediction-error magnitude (P2)
  state_at_encoding JSONB,                      -- mood/context signals; stored, NEVER matched (P4)
  tags             TEXT[] NOT NULL DEFAULT '{}',
  raw_text         TEXT NOT NULL,               -- fallback / free-form capture
  embedding        vector(__DIM__),             -- embedded from goal+expectation+action+outcome ONLY
  search_tsv       tsvector GENERATED ALWAYS AS
                   (to_tsvector('english', coalesce(goal,'') || ' ' || coalesce(expectation,'')
                     || ' ' || coalesce(action,'') || ' ' || coalesce(outcome,''))) STORED,
  access_count     INT NOT NULL DEFAULT 0,
  last_accessed    TIMESTAMPTZ                  -- updated only on reported usage (P3)
);
CREATE INDEX ON episodes USING gin (search_tsv);
CREATE INDEX ON episodes USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON episodes (namespace, created_at DESC);

CREATE TABLE lessons (
  id               BIGSERIAL PRIMARY KEY,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  namespace        TEXT NOT NULL,
  claim            TEXT NOT NULL,               -- the rule
  because          TEXT NOT NULL,               -- causal gist — the transferable part (P2)
  holds_when       TEXT,                        -- deliberately fuzzy boundary conditions
  fails_when       TEXT,
  confidence       REAL NOT NULL DEFAULT 0.5,   -- evidence-driven only (P3, P5)
  last_evidence_at TIMESTAMPTZ NOT NULL DEFAULT now(),  -- denormalized: max(created_at) over lesson_evidence episodes; evidence clock (world drift), never moved by usage
  embedding        vector(__DIM__),             -- claim + because + holds_when
  search_tsv       tsvector GENERATED ALWAYS AS
                   (to_tsvector('english', claim || ' ' || because
                     || ' ' || coalesce(holds_when,''))) STORED,
  access_count     INT NOT NULL DEFAULT 0,
  usefulness       REAL NOT NULL DEFAULT 0.0,   -- running (helped − harmed) signal
  last_accessed    TIMESTAMPTZ,
  disputed         BOOLEAN NOT NULL DEFAULT FALSE,
  -- promotion metadata (global copies only; NULL for originals — §11)
  promoted_from_lesson_id BIGINT REFERENCES lessons(id),
  promotion_reason TEXT,                        -- why the human graduated this lesson
  promoted_at      TIMESTAMPTZ,
  promotion_seed_confidence REAL,               -- confidence inherited at promotion; frozen for audit
  promotion_status TEXT NOT NULL DEFAULT 'active' CHECK (promotion_status IN ('active','demoted')),
  demoted_at       TIMESTAMPTZ,                 -- tombstone, never a delete
  demotion_reason  TEXT
);
CREATE INDEX ON lessons (namespace, promotion_status)
  WHERE promoted_from_lesson_id IS NOT NULL;    -- global retrieval union filters to active
CREATE INDEX ON lessons USING gin (search_tsv);
CREATE INDEX ON lessons USING hnsw (embedding vector_cosine_ops);

CREATE TABLE lesson_evidence (                  -- relational provenance (P1): evidence is a first-class edge
  lesson_id        BIGINT NOT NULL REFERENCES lessons(id) ON DELETE CASCADE,
  episode_id       BIGINT NOT NULL REFERENCES episodes(id),
  relation         TEXT NOT NULL CHECK (relation IN ('support','contradict','refine')),
  reason           TEXT,
  added_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (lesson_id, episode_id)          -- one AGGREGATE verdict per pair — see §5 note
);
CREATE INDEX ON lesson_evidence (episode_id);   -- reverse traversal: which lessons cite this episode (dispute tracing)

CREATE TABLE lesson_links (                     -- lesson↔lesson graph: spreading activation + P9 contradiction links
                                               -- (evidence edges above are lesson↔episode — two graphs, two jobs)
  lesson_id        BIGINT NOT NULL REFERENCES lessons(id),
  related_lesson_id BIGINT NOT NULL REFERENCES lessons(id),
  kind             TEXT NOT NULL CHECK (kind IN ('similar','contradicts','refines')),
  weight           REAL NOT NULL DEFAULT 0.5,   -- cosine similarity at write time
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (lesson_id, related_lesson_id, kind)
);

CREATE TABLE retrieval_events (                 -- exposure log: written by probe/search at query time
  id               BIGSERIAL PRIMARY KEY,
  ts               TIMESTAMPTZ NOT NULL DEFAULT now(),
  namespace        TEXT NOT NULL,
  tool             TEXT NOT NULL CHECK (tool IN ('probe','search')),
  query_context    TEXT NOT NULL,               -- goal+approach as sent
  returned_ids     TEXT[] NOT NULL              -- typed refs "episode:<id>"/"lesson:<id>"; what was SHOWN; record stats untouched (P3)
);

CREATE TABLE usage_reports (                    -- outcomes: only what the caller explicitly reports
  id               BIGSERIAL PRIMARY KEY,
  retrieval_event_id BIGINT NOT NULL REFERENCES retrieval_events(id),
  record_id        TEXT NOT NULL,               -- typed ref "episode:<id>"/"lesson:<id>"
  record_type      TEXT NOT NULL CHECK (record_type IN ('episode','lesson')),
  outcome          TEXT NOT NULL CHECK (outcome IN ('used','helped','harmed','ignored')),
  reported_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (retrieval_event_id, record_id)        -- one verdict per shown record
);
