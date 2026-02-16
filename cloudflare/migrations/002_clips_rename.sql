-- Migration: Rename credits to clips
-- STORYCUT D1 Database - Credits → Clips column rename
-- Apply after deploying updated worker.js code
-- SQLite (D1) supports ALTER TABLE RENAME COLUMN since 3.25.0

-- 1. Rename users.credits → users.clips
ALTER TABLE users RENAME COLUMN credits TO clips;

-- 2. Rename users.monthly_credits → users.monthly_clips
ALTER TABLE users RENAME COLUMN monthly_credits TO monthly_clips;

-- Note: Table names (credit_transactions, credit_packs) cannot be renamed in SQLite
-- without recreating tables. The code maps these internally.
-- Future migration can recreate tables if needed.
