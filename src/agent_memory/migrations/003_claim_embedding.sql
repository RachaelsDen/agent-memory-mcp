-- Migration 003: claim-identity embedding for the two-threshold duplicate
-- guard (Issue #6; DESIGN S8). Additive only: one nullable vector column on
-- lessons plus an HNSW index mirroring the existing embedding index. NULL
-- marks "not yet embedded": db.migrate()'s Python backfill step fills it for
-- pre-existing rows right after this file applies (SQL cannot compute
-- embeddings); the write path populates it for every new lesson.
ALTER TABLE lessons ADD COLUMN claim_embedding vector(__DIM__);
CREATE INDEX ON lessons USING hnsw (claim_embedding vector_cosine_ops);
