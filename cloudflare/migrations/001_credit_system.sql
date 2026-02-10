-- Migration: Credit System (DeeVid model)
-- Run this on existing D1 databases to add credit system tables and columns

-- 1. users 테이블 새 컬럼 추가
ALTER TABLE users ADD COLUMN plan_id TEXT DEFAULT 'free';
ALTER TABLE users ADD COLUMN plan_expires_at DATETIME;
ALTER TABLE users ADD COLUMN monthly_credits INTEGER DEFAULT 0;
ALTER TABLE users ADD COLUMN stripe_customer_id TEXT;
ALTER TABLE users ADD COLUMN stripe_subscription_id TEXT;

-- 2. credit_transactions 테이블 새 컬럼
ALTER TABLE credit_transactions ADD COLUMN action TEXT;

-- 3. payments 테이블 새 컬럼
ALTER TABLE payments ADD COLUMN payment_type TEXT DEFAULT 'one_time';
ALTER TABLE payments ADD COLUMN stripe_checkout_session_id TEXT;

-- 4. 구독 이력 테이블
CREATE TABLE IF NOT EXISTS subscriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    stripe_subscription_id TEXT,
    stripe_price_id TEXT,
    status TEXT NOT NULL,
    current_period_start DATETIME,
    current_period_end DATETIME,
    cancelled_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 5. 크레딧 팩 테이블
CREATE TABLE IF NOT EXISTS credit_packs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    pack_type TEXT NOT NULL,
    credits INTEGER NOT NULL,
    amount_usd REAL NOT NULL,
    stripe_payment_id TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 6. 새 인덱스
CREATE INDEX IF NOT EXISTS idx_subscriptions_user_id ON subscriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_status ON subscriptions(status);
CREATE INDEX IF NOT EXISTS idx_credit_packs_user_id ON credit_packs(user_id);
CREATE INDEX IF NOT EXISTS idx_users_stripe_customer_id ON users(stripe_customer_id);

-- 7. 기존 사용자 크레딧 초기화 (free 플랜 20cr)
-- UPDATE users SET credits = 20, plan_id = 'free' WHERE plan_id IS NULL;
