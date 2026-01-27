-- 사용자 테이블 생성
DROP TABLE IF EXISTS users;
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    username TEXT,
    credits INTEGER DEFAULT 5, -- 가입 시 무료 크레딧 5개
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    last_login_at INTEGER
);

-- 크레딧 사용 내역 (나중을 위해 미리 생성)
DROP TABLE IF EXISTS credit_logs;
CREATE TABLE IF NOT EXISTS credit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    amount INTEGER NOT NULL, -- 사용(-) 또는 충전(+)
    description TEXT,
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (user_id) REFERENCES users(id)
);
