-- Migration 004: general namespace index on lessons (PR #17 review fix).
-- 001's only lessons(namespace) index is PARTIAL (promoted copies only), but
-- digest, stats, consolidate-scan, and write-lesson's duplicate-claim guard
-- filter lessons by namespace across ALL rows — each was a sequential scan.
CREATE INDEX IF NOT EXISTS idx_lessons_namespace ON lessons (namespace);
