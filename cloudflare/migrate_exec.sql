PRAGMA foreign_keys = OFF;

-- Step 1: 새 users 테이블 생성 (id = TEXT)
CREATE TABLE users_new (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    username TEXT,
    credits INTEGER DEFAULT 5,
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    last_login_at INTEGER,
    plan_id TEXT DEFAULT 'free',
    plan_expires_at INTEGER,
    monthly_credits INTEGER DEFAULT 0,
    stripe_subscription_id TEXT,
    google_id TEXT,
    stripe_customer_id TEXT,
    api_token TEXT,
    updated_at TEXT
);

-- Step 2: 대상 유저 2명만 복사 (id = email)
INSERT INTO users_new (id, email, password_hash, username, credits, created_at, last_login_at, plan_id, plan_expires_at, monthly_credits, stripe_subscription_id, google_id, stripe_customer_id, api_token, updated_at)
  SELECT email, email, password_hash, username, credits, created_at, last_login_at, plan_id, plan_expires_at, monthly_credits, stripe_subscription_id, google_id, stripe_customer_id, api_token, updated_at
  FROM users WHERE email IN ('twinspa0713@gmail.com', 'neopioneer0713@gmail.com');

-- Step 3: FK 테이블 마이그레이션
UPDATE credit_transactions SET user_id = 'twinspa0713@gmail.com' WHERE user_id = '1';
UPDATE credit_transactions SET user_id = 'neopioneer0713@gmail.com' WHERE user_id = '3';
UPDATE projects SET user_id = 'twinspa0713@gmail.com' WHERE user_id = '1';
UPDATE projects SET user_id = 'neopioneer0713@gmail.com' WHERE user_id = '3';
UPDATE payments SET user_id = 'twinspa0713@gmail.com' WHERE user_id = '1';
UPDATE payments SET user_id = 'neopioneer0713@gmail.com' WHERE user_id = '3';
UPDATE subscriptions SET user_id = 'twinspa0713@gmail.com' WHERE user_id = '1';
UPDATE subscriptions SET user_id = 'neopioneer0713@gmail.com' WHERE user_id = '3';
UPDATE credit_packs SET user_id = 'twinspa0713@gmail.com' WHERE user_id = '1';
UPDATE credit_packs SET user_id = 'neopioneer0713@gmail.com' WHERE user_id = '3';
UPDATE notifications SET user_id = 'twinspa0713@gmail.com' WHERE user_id = '1';
UPDATE notifications SET user_id = 'neopioneer0713@gmail.com' WHERE user_id = '3';

-- Step 4: 나머지 유저 FK 데이터 삭제
DELETE FROM credit_transactions WHERE user_id NOT IN ('twinspa0713@gmail.com', 'neopioneer0713@gmail.com');
DELETE FROM projects WHERE user_id NOT IN ('twinspa0713@gmail.com', 'neopioneer0713@gmail.com');
DELETE FROM payments WHERE user_id NOT IN ('twinspa0713@gmail.com', 'neopioneer0713@gmail.com');
DELETE FROM subscriptions WHERE user_id NOT IN ('twinspa0713@gmail.com', 'neopioneer0713@gmail.com');
DELETE FROM credit_packs WHERE user_id NOT IN ('twinspa0713@gmail.com', 'neopioneer0713@gmail.com');
DELETE FROM notifications WHERE user_id NOT IN ('twinspa0713@gmail.com', 'neopioneer0713@gmail.com');

-- Step 5: 테이블 교체
DROP TABLE users;
ALTER TABLE users_new RENAME TO users;

-- Step 6: 인덱스 재생성
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_stripe_customer_id ON users(stripe_customer_id);

PRAGMA foreign_keys = ON;
