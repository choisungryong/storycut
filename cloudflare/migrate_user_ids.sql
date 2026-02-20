-- ============================================================
-- 유저 ID 마이그레이션: 정수 → 이메일
-- 대상 계정:
--   twinspa0713@gmail.com (기존 정수 ID → 이메일로 변경)
--   neopioneer0713@gmail.com (기존 정수 ID → 이메일로 변경)
-- 나머지 계정: 삭제
-- ============================================================

-- Step 1: 기존 유저 확인 (실행 전 먼저 SELECT로 확인)
-- SELECT id, email, credits, plan_id FROM users;

-- Step 2: 모든 FK 테이블의 user_id를 이메일로 업데이트
-- (twinspa0713@gmail.com)
UPDATE credit_transactions SET user_id = 'twinspa0713@gmail.com' WHERE user_id IN (SELECT id FROM users WHERE email = 'twinspa0713@gmail.com');
UPDATE projects SET user_id = 'twinspa0713@gmail.com' WHERE user_id IN (SELECT id FROM users WHERE email = 'twinspa0713@gmail.com');
UPDATE payments SET user_id = 'twinspa0713@gmail.com' WHERE user_id IN (SELECT id FROM users WHERE email = 'twinspa0713@gmail.com');
UPDATE subscriptions SET user_id = 'twinspa0713@gmail.com' WHERE user_id IN (SELECT id FROM users WHERE email = 'twinspa0713@gmail.com');
UPDATE credit_packs SET user_id = 'twinspa0713@gmail.com' WHERE user_id IN (SELECT id FROM users WHERE email = 'twinspa0713@gmail.com');
UPDATE notifications SET user_id = 'twinspa0713@gmail.com' WHERE user_id IN (SELECT id FROM users WHERE email = 'twinspa0713@gmail.com');

-- (neopioneer0713@gmail.com)
UPDATE credit_transactions SET user_id = 'neopioneer0713@gmail.com' WHERE user_id IN (SELECT id FROM users WHERE email = 'neopioneer0713@gmail.com');
UPDATE projects SET user_id = 'neopioneer0713@gmail.com' WHERE user_id IN (SELECT id FROM users WHERE email = 'neopioneer0713@gmail.com');
UPDATE payments SET user_id = 'neopioneer0713@gmail.com' WHERE user_id IN (SELECT id FROM users WHERE email = 'neopioneer0713@gmail.com');
UPDATE subscriptions SET user_id = 'neopioneer0713@gmail.com' WHERE user_id IN (SELECT id FROM users WHERE email = 'neopioneer0713@gmail.com');
UPDATE credit_packs SET user_id = 'neopioneer0713@gmail.com' WHERE user_id IN (SELECT id FROM users WHERE email = 'neopioneer0713@gmail.com');
UPDATE notifications SET user_id = 'neopioneer0713@gmail.com' WHERE user_id IN (SELECT id FROM users WHERE email = 'neopioneer0713@gmail.com');

-- Step 3: 대상 외 계정의 FK 데이터 삭제
DELETE FROM credit_transactions WHERE user_id NOT IN ('twinspa0713@gmail.com', 'neopioneer0713@gmail.com');
DELETE FROM projects WHERE user_id NOT IN ('twinspa0713@gmail.com', 'neopioneer0713@gmail.com');
DELETE FROM payments WHERE user_id NOT IN ('twinspa0713@gmail.com', 'neopioneer0713@gmail.com');
DELETE FROM subscriptions WHERE user_id NOT IN ('twinspa0713@gmail.com', 'neopioneer0713@gmail.com');
DELETE FROM credit_packs WHERE user_id NOT IN ('twinspa0713@gmail.com', 'neopioneer0713@gmail.com');
DELETE FROM notifications WHERE user_id NOT IN ('twinspa0713@gmail.com', 'neopioneer0713@gmail.com');

-- Step 4: users 테이블 마이그레이션
-- 기존 유저의 데이터를 보존하면서 id를 이메일로 변경
-- SQLite는 UPDATE PK를 지원하므로 직접 변경
UPDATE users SET id = email WHERE email IN ('twinspa0713@gmail.com', 'neopioneer0713@gmail.com');

-- Step 5: 나머지 유저 삭제
DELETE FROM users WHERE id NOT IN ('twinspa0713@gmail.com', 'neopioneer0713@gmail.com');

-- Step 6: 검증
-- SELECT id, email, credits, plan_id FROM users;
-- SELECT COUNT(*) FROM credit_transactions;
-- SELECT COUNT(*) FROM projects;
