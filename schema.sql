-- 사용자 테이블 (기존 데이터 보존)
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    username TEXT,
    credits INTEGER DEFAULT 5,
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    last_login_at INTEGER
);

-- 크레딧 사용 내역
CREATE TABLE IF NOT EXISTS credit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    amount INTEGER NOT NULL,
    description TEXT,
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 프로젝트(영상) 테이블 (신규 추가)
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    status TEXT DEFAULT 'queued',
    input_data TEXT,
    output_url TEXT,
    error_message TEXT,
    created_at TEXT,
    completed_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
