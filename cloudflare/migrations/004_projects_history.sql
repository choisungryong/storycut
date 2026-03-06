-- 004: projects 테이블에 히스토리용 컬럼 추가
ALTER TABLE projects ADD COLUMN type TEXT DEFAULT 'video';
ALTER TABLE projects ADD COLUMN thumbnail_url TEXT;
ALTER TABLE projects ADD COLUMN video_url TEXT;
ALTER TABLE projects ADD COLUMN download_url TEXT;
ALTER TABLE projects ADD COLUMN scene_count INTEGER DEFAULT 0;
ALTER TABLE projects ADD COLUMN duration_sec REAL;
ALTER TABLE projects ADD COLUMN genre TEXT;
ALTER TABLE projects ADD COLUMN style TEXT;

-- 복합 인덱스: user_id + created_at (히스토리 조회 최적화)
CREATE INDEX IF NOT EXISTS idx_projects_user_created ON projects(user_id, created_at DESC);
