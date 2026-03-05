-- 쿠폰 마스터
CREATE TABLE IF NOT EXISTS coupons (
    code TEXT PRIMARY KEY,            -- "PREMIUM2026" 등
    type TEXT NOT NULL,               -- 'plan' | 'credits'
    plan_id TEXT,                     -- plan 타입: lite/pro/premium
    duration_days INTEGER,            -- plan 타입: 유효 기간(일)
    credits INTEGER,                  -- credits 타입: 지급량
    max_uses INTEGER NOT NULL DEFAULT 1,
    usage_count INTEGER DEFAULT 0,
    expiry_date DATETIME,             -- NULL = 무기한
    active INTEGER DEFAULT 1,
    created_by TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 사용 내역 (유저당 1회 제한)
CREATE TABLE IF NOT EXISTS coupon_redemptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    coupon_code TEXT NOT NULL,
    user_id TEXT NOT NULL,
    redeemed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(coupon_code, user_id),
    FOREIGN KEY (coupon_code) REFERENCES coupons(code),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_coupon_redemptions_user ON coupon_redemptions(user_id);
CREATE INDEX IF NOT EXISTS idx_coupon_redemptions_code ON coupon_redemptions(coupon_code);
