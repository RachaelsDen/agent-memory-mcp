-- Migration 002: dispute reason storage (plan task 13; DESIGN S9 memory_dispute).
-- Additive only: one column on lessons; 001 is never touched. The runner's
-- __DIM__ replacement pass is a harmless no-op on this file.
ALTER TABLE lessons ADD COLUMN dispute_reason TEXT;
