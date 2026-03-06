// STORYCUT v2.0 - 프론트엔드 로직 (완전 재작성)

// ─── 콘텐츠 유형 2단계 분류 ───
const CONTENT_CATEGORIES = {
    "fiction": {
        label: "창작 픽션",
        types: [
            { value: "fiction", label: "자유 창작" },
            { value: "mystery", label: "미스터리/스릴러" },
            { value: "romance", label: "로맨스/감성" },
            { value: "sf", label: "SF" },
            { value: "horror", label: "호러/공포" },
        ]
    },
    "traditional": {
        label: "전통/설화",
        types: [
            { value: "folktale", label: "전래동화" },
            { value: "myth", label: "신화/전설" },
        ]
    },
    "nonfiction": {
        label: "역사/교양",
        types: [
            { value: "historical", label: "역사 이야기" },
            { value: "economy", label: "경제/비즈니스" },
            { value: "documentary", label: "실화/다큐" },
        ]
    },
    "children": {
        label: "아동/동화",
        types: [
            { value: "fairytale", label: "동화" },
            { value: "educational", label: "교육 콘텐츠" },
        ]
    }
};

// content_type → genre 자동 매핑 (백엔드 호환)
const CONTENT_TYPE_TO_GENRE = {
    "folktale": "fantasy", "myth": "fantasy",
    "historical": "drama", "economy": "drama", "documentary": "drama",
    "fiction": "emotional", "mystery": "mystery", "romance": "romance",
    "sf": "fantasy", "horror": "horror",
    "fairytale": "comedy", "educational": "drama",
};

// 콘텐츠 소분류 드롭다운 갱신
function updateContentTypeOptions(categoryValue) {
    const typeSelect = document.getElementById('content_type');
    const genreSelect = document.getElementById('genre');
    if (!typeSelect) return;

    typeSelect.innerHTML = '';
    const category = CONTENT_CATEGORIES[categoryValue];
    if (category) {
        category.types.forEach((t, idx) => {
            const opt = document.createElement('option');
            opt.value = t.value;
            opt.textContent = t.label;
            if (idx === 0) opt.selected = true;
            typeSelect.appendChild(opt);
        });
    }

    // genre 숨겨진 필드 자동 매핑
    if (genreSelect && typeSelect.value) {
        genreSelect.value = CONTENT_TYPE_TO_GENRE[typeSelect.value] || 'emotional';
    }
}

// 초기화: 대분류 변경 이벤트 + 최초 소분류 세팅
document.addEventListener('DOMContentLoaded', function() {
    const catSelect = document.getElementById('content_category');
    const typeSelect = document.getElementById('content_type');
    const genreSelect = document.getElementById('genre');

    if (catSelect) {
        catSelect.addEventListener('change', function() {
            updateContentTypeOptions(this.value);
        });
        // 초기 소분류 세팅
        updateContentTypeOptions(catSelect.value);
    }

    if (typeSelect) {
        typeSelect.addEventListener('change', function() {
            // genre 숨겨진 필드 자동 매핑
            if (genreSelect) {
                genreSelect.value = CONTENT_TYPE_TO_GENRE[this.value] || 'emotional';
            }
        });
    }
});

// 서버 에러 코드 → 한국어 매핑
const ERROR_MESSAGES = {
    'project_not_found': '프로젝트를 찾을 수 없습니다.',
    'video_not_found': '영상을 찾을 수 없습니다.',
    'scene_not_found': '씬을 찾을 수 없습니다.',
    'image_not_found': '이미지를 찾을 수 없습니다.',
    'character_not_found': '캐릭터를 찾을 수 없습니다.',
    'asset_not_found': '에셋을 찾을 수 없습니다.',
    'invalid_project_id': '잘못된 프로젝트 ID입니다.',
    'invalid_scene_id': '잘못된 씬 ID입니다.',
    'invalid_voice_id': '잘못된 음성 ID입니다.',
    'invalid_filename': '잘못된 파일명입니다.',
    'invalid_asset_type': '잘못된 에셋 타입입니다.',
    'invalid_webhook_url': '잘못된 웹훅 URL입니다.',
    'file_too_large': '파일이 너무 큽니다 (최대 50MB).',
    'request_body_too_large': '요청이 너무 큽니다.',
    'tts_generation_failed': 'TTS 생성에 실패했습니다.',
    'scene_regeneration_failed': '씬 재생성에 실패했습니다.',
    'video_composition_failed': '영상 합성에 실패했습니다.',
    'character_generation_failed': '캐릭터 생성에 실패했습니다.',
    'access_denied': '접근이 거부되었습니다.',
    'email_and_password_required': '이메일과 비밀번호를 입력하세요.',
    'username_required': '사용자명을 입력하세요.',
    'no_video_clips_to_compose': '합성할 비디오 클립이 없습니다.',
    'generation_already_in_progress': '이미 생성이 진행 중입니다.',
    'composition_already_in_progress': '이미 합성이 진행 중입니다.',
    'no_lyrics_in_project': '이 프로젝트에 가사가 없습니다.',
    'project_id_required': '프로젝트 ID가 필요합니다.',
    'not_found': '요청한 리소스를 찾을 수 없습니다.',
    'internal_server_error': '서버 오류가 발생했습니다.',
};
function getErrorMessage(code) {
    if (!code) return '알 수 없는 오류가 발생했습니다.';
    // Handle compound codes like "scene_not_found: 3"
    const baseCode = code.split(':')[0].trim();
    return ERROR_MESSAGES[baseCode] || code;
}

// [보안] HTML 이스케이프 유틸리티 — XSS 방지
function escapeHtml(str) {
    if (str == null) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

class StorycutApp {
    constructor() {
        this.projectId = null;
        this.websocket = null;
        this.serverUrl = null;

        // Review State
        this.currentStoryData = null;
        this.currentRequestParams = null;

        // Progress tracking
        this.pollingInterval = null;
        this.isGenerating = false;
        this.pollingFailCount = 0;
        this.mvPollingFailCount = 0;

        // Regeneration tracking (prevent double-click)
        this._regeneratingScenes = new Set();

        // I2V progress banner timers
        this._i2vTimers = {};

        // Input mode: 'ai' or 'script'
        this._inputMode = 'ai';

        // Multi-speaker voice selection
        this._detectedSpeakers = [];
        this._availableVoices = null;  // cached voice list
        this._characterVoices = {};    // speaker -> {voice_id, voice_name}

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.updateDurationDisplay();
        this._ensureToastContainer();
        this._setupNavigationGuard();
    }

    // ===== 브라우저 뒤로가기/새로고침 방어 =====
    _setupNavigationGuard() {
        // beforeunload: 탭 닫기/새로고침 방어
        window.addEventListener('beforeunload', (e) => {
            if (this.isGenerating) {
                e.preventDefault();
                e.returnValue = '';
            }
        });

        // popstate: 브라우저 뒤로가기 방어
        history.pushState(null, '', location.href);
        window.addEventListener('popstate', (e) => {
            if (this.isGenerating) {
                history.pushState(null, '', location.href);
                this.showToast('영상 생성 중에는 뒤로가기를 사용할 수 없습니다.', 'warning');
            }
        });
    }

    // ===== Toast Notification (alert 대체) =====
    _ensureToastContainer() {
        if (!document.querySelector('.toast-container')) {
            const container = document.createElement('div');
            container.className = 'toast-container';
            document.body.appendChild(container);
        }
    }

    showToast(message, type = 'info') {
        const container = document.querySelector('.toast-container');
        if (!container) return;
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        container.appendChild(toast);
        setTimeout(() => toast.remove(), 5000);
    }

    // ===== 브라우저 푸시 알림 =====
    _requestNotificationPermission() {
        if ('Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission();
        }
    }

    _sendCompletionNotification(title, body) {
        // 탭이 이미 포커스 중이면 토스트만으로 충분
        if (document.hasFocus()) return;
        if (!('Notification' in window) || Notification.permission !== 'granted') return;
        const n = new Notification(title, {
            body: body || '영상이 완성되었습니다. 클릭해서 확인하세요.',
            icon: '/static/icon.png',
            tag: 'storycut-complete',  // 중복 알림 방지
        });
        n.onclick = () => { window.focus(); n.close(); };
    }

    setupEventListeners() {
        // 1단계: 스토리 생성 (폼 제출)
        const form = document.getElementById('generate-form');
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            if (this._inputMode === 'script') {
                this.startScriptGeneration();
            } else {
                this.startStoryGeneration();
            }
        });

        // 입력 모드 토글
        const modeAiBtn = document.getElementById('mode-ai-story');
        const modeScriptBtn = document.getElementById('mode-direct-script');
        if (modeAiBtn && modeScriptBtn) {
            modeAiBtn.addEventListener('click', () => this._setInputMode('ai'));
            modeScriptBtn.addEventListener('click', () => this._setInputMode('script'));
        }

        // 2단계: 영상 생성 시작 (리뷰 후 확정)
        const startBtn = document.getElementById('start-video-generation-btn');
        if (startBtn) {
            startBtn.addEventListener('click', () => {
                this.startFinalGeneration();
            });
        }

        // 다시 설정하기 (입력 폼으로 복귀)
        const backBtn = document.getElementById('back-to-input-btn');
        if (backBtn) {
            backBtn.addEventListener('click', () => {
                this.showSection('input');
            });
        }

        // 슬라이더
        const durationSlider = document.getElementById('duration');
        durationSlider.addEventListener('input', () => {
            this.updateDurationDisplay();
        });

        // 이미지 모델 선택 시 Gemini 안내
        const imageModelSelect = document.getElementById('image_model');
        if (imageModelSelect) {
            imageModelSelect.addEventListener('change', () => {
                this.updateImageModelHint();
            });
            // 초기 상태 업데이트 (로그인 후)
            setTimeout(() => this.updateImageModelHint(), 500);
        }

        // 로그 클리어
        const clearLogBtn = document.getElementById('clear-log-btn');
        clearLogBtn.addEventListener('click', () => {
            document.getElementById('log-content').innerHTML = '';
        });

        // 새 영상 만들기
        const newVideoBtn = document.getElementById('new-video-btn');
        newVideoBtn.addEventListener('click', () => {
            this.resetUI();
        });

        // 쇼츠 선택 시 duration 슬라이더 자동 조정
        document.querySelectorAll('input[name="platform"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                const slider = document.getElementById('duration');
                const display = document.getElementById('duration-display');
                if (e.target.value === 'youtube_shorts') {
                    slider.max = 60;
                    if (parseInt(slider.value) > 60) {
                        slider.value = 30;
                        display.textContent = '30';
                    }
                } else {
                    slider.max = 300;
                }
            });
        });

        // 유튜브 업로드 버튼 (준비중)
        const ytUploadBtn = document.getElementById('youtube-upload-btn');
        if (ytUploadBtn) {
            ytUploadBtn.addEventListener('click', () => {
                this.showToast('유튜브 업로드 기능은 준비 중입니다.', 'info');
            });
        }

        // Hook text 편집
        const hookEditBtn = document.getElementById('shorts-hook-edit-btn');
        if (hookEditBtn) {
            hookEditBtn.addEventListener('click', () => {
                const hookArea = document.getElementById('shorts-hook-area');
                const display = hookArea.querySelector('.shorts-hook-display');
                const currentText = document.getElementById('shorts-hook-text').textContent;
                display.innerHTML = `<input type="text" value="${escapeHtml(currentText)}" maxlength="15" /><button class="btn-icon-small" title="저장"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg></button>`;
                const input = display.querySelector('input');
                const saveBtn = display.querySelector('button');
                input.focus();
                const save = () => {
                    const newText = input.value.trim() || currentText;
                    display.innerHTML = `<span id="shorts-hook-text">${escapeHtml(newText)}</span><button id="shorts-hook-edit-btn" class="btn-icon-small" title="편집"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg></button>`;
                    // Re-bind edit button
                    display.querySelector('#shorts-hook-edit-btn').addEventListener('click', () => hookEditBtn.click());
                };
                saveBtn.addEventListener('click', save);
                input.addEventListener('keydown', (e) => { if (e.key === 'Enter') save(); });
            });
        }

        // 네비게이션
        document.getElementById('nav-create').addEventListener('click', (e) => {
            e.preventDefault();
            this.showSection('input');
            this.setNavActive('nav-create');
        });

        document.getElementById('nav-history').addEventListener('click', (e) => {
            e.preventDefault();
            this.loadHistory();
            this.showSection('history');
            this.setNavActive('nav-history');
        });

        document.getElementById('nav-board').addEventListener('click', (e) => {
            e.preventDefault();
            this.loadBoard();
            this.showSection('board');
            this.setNavActive('nav-board');
        });

        // 하단 모바일 탭바 이벤트
        document.querySelectorAll('.mobile-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                e.preventDefault();
                const navId = tab.dataset.nav;
                // 상단 탭 클릭을 트리거하여 기존 로직 재활용
                document.getElementById(navId)?.click();
            });
        });

        // 게시판 글쓰기 버튼
        const writePostBtn = document.getElementById('write-post-btn');
        if (writePostBtn) {
            writePostBtn.addEventListener('click', () => this.openWritePostModal());
        }

        // 게시판 등록 버튼
        const postSubmitBtn = document.getElementById('post-submit-btn');
        if (postSubmitBtn) {
            postSubmitBtn.addEventListener('click', () => this.submitPost());
        }

        // 보관함 새로고침 버튼
        const refreshHistoryBtn = document.getElementById('refresh-history-btn');
        if (refreshHistoryBtn) {
            refreshHistoryBtn.addEventListener('click', () => this.loadHistory());
        }

        // 영상 재합성 버튼
        const recomposeBtn = document.getElementById('recompose-btn');
        if (recomposeBtn) {
            recomposeBtn.addEventListener('click', () => {
                if (this.projectId) {
                    this.recomposeVideo(this.projectId);
                }
            });
        }

        // 이미지 프리뷰 섹션 네비게이션
        const backToReviewBtn = document.getElementById('back-to-review-btn');
        if (backToReviewBtn) {
            backToReviewBtn.addEventListener('click', () => {
                this.showSection('review');
            });
        }

        const approveImagesBtn = document.getElementById('approve-images-btn');
        if (approveImagesBtn) {
            approveImagesBtn.addEventListener('click', () => {
                this.startFinalGenerationAfterImageReview();
            });
        }

        // 이미지만 먼저 생성 버튼 → 캐릭터 캐스팅 먼저 진행
        const generateImagesBtn = document.getElementById('generate-images-btn');
        if (generateImagesBtn) {
            generateImagesBtn.addEventListener('click', () => {
                this.startCharacterCasting();
            });
        }
    }

    setNavActive(navId) {
        // 상단 탭 동기화
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        document.getElementById(navId)?.classList.add('active');

        // 하단 모바일 탭 동기화
        document.querySelectorAll('.mobile-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.nav === navId);
        });
    }

    updateDurationDisplay() {
        const sec = parseInt(document.getElementById('duration').value);
        const min = Math.floor(sec / 60);
        const remSec = sec % 60;
        let text;
        if (min === 0) {
            text = `${sec}초`;
        } else if (remSec === 0) {
            text = `${min}분`;
        } else {
            text = `${min}분 ${remSec}초`;
        }
        document.getElementById('duration-display').textContent = text;
    }

    getApiBaseUrl() {
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return '';
        }
        // 모든 API 요청은 Worker 경유 (클립 차감 + 플랜 제한 적용)
        return 'https://storycut-worker.twinspa0713.workers.dev';
    }

    getWorkerUrl() {
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return '';
        }
        return 'https://storycut-worker.twinspa0713.workers.dev';
    }

    getAuthHeaders(extra = {}) {
        const headers = { 'Content-Type': 'application/json', ...extra };
        const token = localStorage.getItem('token');
        if (token) headers['Authorization'] = `Bearer ${token}`;
        return headers;
    }

    getMediaBaseUrl() {
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return '';
        }
        // 미디어 파일(이미지, 영상)도 Worker 경유 (R2 직접 서빙 + Railway 폴백)
        return 'https://storycut-worker.twinspa0713.workers.dev';
    }

    _shouldShowWatermark() {
        try {
            const user = JSON.parse(localStorage.getItem('user') || '{}');
            const paidPlans = ['lite', 'pro', 'premium'];
            return !paidPlans.includes(user.plan_id);
        } catch {
            return true;
        }
    }

    // ==================== Step 1: 스토리 생성 ====================
    async startStoryGeneration() {
        // 크레딧 사전 확인
        if (typeof checkCreditsBeforeAction === 'function') {
            const ok = await checkCreditsBeforeAction('video');
            if (!ok) return;
        }

        const formData = new FormData(document.getElementById('generate-form'));

        const btn = document.getElementById('generate-story-btn');
        const originalBtnText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = '<span class="btn-icon">⏳</span> 스토리 생성 중...';

        // Build topic with dialogue hint if enabled
        let topic = formData.get('topic') || null;
        const includeDialogue = document.getElementById('include_dialogue')?.checked || false;
        if (includeDialogue && topic) {
            topic = topic + ' (반드시 남녀 캐릭터 간 대화를 포함할 것. 최소 2명 이상의 화자)';
        } else if (includeDialogue && !topic) {
            topic = '남녀 캐릭터 간 대화가 풍부한 드라마 시나리오 (최소 2명 이상의 화자, 나레이터 + 남성 + 여성)';
        }

        const selectedContentType = document.getElementById('content_type')?.value || 'fiction';
        const requestData = {
            topic: topic,
            content_type: selectedContentType,
            genre: CONTENT_TYPE_TO_GENRE[selectedContentType] || formData.get('genre') || 'emotional',
            mood: formData.get('mood'),
            style: formData.get('style'),
            voice: formData.get('voice'),
            duration: parseInt(formData.get('duration')),
            platform: formData.get('platform'),
            character_ethnicity: formData.get('character_ethnicity') || 'auto',
            include_dialogue: includeDialogue,

            // Feature Flags - 기본값 (리뷰 섹션에서 최종 업데이트)
            hook_scene1_video: false,
            ffmpeg_kenburns: true,
            ffmpeg_audio_ducking: false,
            subtitle_burn_in: true,
            context_carry_over: true,
            optimization_pack: false,
            film_look: false,
        };

        this.currentRequestParams = requestData;

        // 원샷 모드 확인
        const isOneShot = document.getElementById('auto-compose')?.checked || false;

        try {
            // 즉시 progress 화면으로 전환
            btn.disabled = false;
            btn.innerHTML = originalBtnText;
            this.showSection('progress');

            if (isOneShot) {
                // ===== 원샷 모드: 스토리+영상 한 번에 =====
                this.updateStepStatus('story', '원샷 생성 중...');
                document.getElementById('status-message').textContent = '스토리 생성 후 자동으로 영상을 만듭니다...';
                document.getElementById('progress-percentage').textContent = '5%';
                document.getElementById('progress-bar').style.width = '5%';
                const progressTitle = document.getElementById('progress-title');
                if (progressTitle) progressTitle.textContent = '⏳ 원샷 생성 중...';

                const workerUrl = this.getWorkerUrl();
                const railwayUrl = this.getApiBaseUrl();
                let response;

                try {
                    const controller = new AbortController();
                    const timeoutId = setTimeout(() => controller.abort(), 30000);
                    response = await fetch(`${workerUrl}/api/generate/one-shot`, {
                        method: 'POST',
                        headers: this.getAuthHeaders(),
                        body: JSON.stringify(requestData),
                        signal: controller.signal
                    });
                    clearTimeout(timeoutId);
                } catch (workerError) {
                    console.warn('[OneShot] Worker 실패, Railway 폴백:', workerError.message);
                    response = await fetch(`${railwayUrl}/api/generate/one-shot`, {
                        method: 'POST',
                        headers: this.getAuthHeaders(),
                        body: JSON.stringify(requestData)
                    });
                }

                if (!response.ok) {
                    if (typeof handleApiError === 'function' && await handleApiError(response.clone(), 'video')) {
                        return;
                    }
                    let errorMsg = 'One-shot generation failed';
                    try {
                        const error = await response.json();
                        errorMsg = error.detail || error.error || errorMsg;
                    } catch (e) { }
                    throw new Error(errorMsg);
                }

                const result = await response.json();
                this.projectId = result.project_id;
                this.isGenerating = true;
                this._requestNotificationPermission();

                // 크레딧 차감 반영
                if (typeof deductLocalCredits === 'function') deductLocalCredits('video');

                this.addLog('INFO', `✅ 원샷 생성 시작됨 (Project ID: ${this.projectId})`);
                this.addLog('INFO', '⏳ 스토리 생성 → 영상 생성 자동 진행 중...');

                // WebSocket + Polling
                this.connectWebSocket(this.projectId);
                setTimeout(() => {
                    if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
                        this.addLog('INFO', 'Polling으로 상태 확인 중...');
                        this.startPolling(this.projectId);
                    }
                }, 2000);

            } else {
                // ===== 기존 2단계 모드 =====
                this.updateStepStatus('story', '스토리 생성 중...');
                document.getElementById('status-message').textContent = 'AI가 스토리를 구상하고 있습니다...';
                document.getElementById('progress-percentage').textContent = '5%';
                document.getElementById('progress-bar').style.width = '5%';

                // 가짜 진행률 애니메이션 (체감 속도 개선)
                const storyMessages = [
                    { pct: 8, msg: '장르와 분위기를 분석하고 있습니다...' },
                    { pct: 15, msg: '캐릭터와 세계관을 설계하고 있습니다...' },
                    { pct: 22, msg: '스토리 구조를 잡고 있습니다...' },
                    { pct: 30, msg: '기승전결 아크를 설계하고 있습니다...' },
                    { pct: 38, msg: '장면별 내러티브를 작성하고 있습니다...' },
                    { pct: 45, msg: '대사와 나레이션을 다듬고 있습니다...' },
                    { pct: 52, msg: '비주얼 프롬프트를 생성하고 있습니다...' },
                    { pct: 58, msg: '카메라 워크를 설정하고 있습니다...' },
                    { pct: 64, msg: '스토리 일관성을 검증하고 있습니다...' },
                    { pct: 70, msg: '화자(Speaker)를 분석하고 있습니다...' },
                    { pct: 75, msg: '시나리오를 최종 정리하고 있습니다...' },
                    { pct: 78, msg: '거의 완료 - 화자 확인 & 음성 선택 화면으로 이동합니다...' },
                ];
                let msgIndex = 0;
                const progressInterval = setInterval(() => {
                    if (msgIndex < storyMessages.length) {
                        const { pct, msg } = storyMessages[msgIndex];
                        this.updateProgress(pct, msg);
                        msgIndex++;
                    }
                }, 2500);

                // 스토리 생성: Worker 먼저 시도, 타임아웃 시 Railway 폴백
                const workerUrl = this.getWorkerUrl();
                const railwayUrl = this.getApiBaseUrl();
                const token = localStorage.getItem('token');
                let response;

                try {
                    const controller = new AbortController();
                    const timeoutId = setTimeout(() => controller.abort(), 120000); // 2분 타임아웃
                    response = await fetch(`${workerUrl}/api/generate/story`, {
                        method: 'POST',
                        headers: this.getAuthHeaders(),
                        body: JSON.stringify(requestData),
                        signal: controller.signal
                    });
                    clearTimeout(timeoutId);
                } catch (workerError) {
                    console.warn('[Story] Worker 실패, Railway 폴백:', workerError.message);
                    this.updateProgress(40, 'Worker 타임아웃 — 백엔드로 재시도 중...');
                    response = await fetch(`${railwayUrl}/api/generate/story`, {
                        method: 'POST',
                        headers: this.getAuthHeaders(),
                        body: JSON.stringify(requestData)
                    });
                }

                clearInterval(progressInterval);

                if (!response.ok) {
                    if (typeof handleApiError === 'function' && await handleApiError(response.clone(), 'video')) {
                        return;
                    }
                    let errorMsg = 'Story generation failed';
                    try {
                        const error = await response.json();
                        errorMsg = error.detail || error.error || errorMsg;
                    } catch (e) { }
                    throw new Error(errorMsg);
                }

                // 완료 애니메이션
                this.updateProgress(90, '스토리 생성 완료! 결과를 불러오는 중...');

                const result = await response.json();

                if (result.story_data) {
                    this.updateProgress(100, '스토리가 완성되었습니다!');
                    this.currentStoryData = result.story_data;
                    this.currentRequestParams = requestData;

                    // Store detected speakers from API response
                    if (result.detected_speakers) {
                        this.currentStoryData.detected_speakers = result.detected_speakers;
                    }

                    // 크레딧 차감 반영
                    if (typeof deductLocalCredits === 'function') deductLocalCredits('video');

                    // 짧은 딜레이로 100% 표시 후 전환
                    await new Promise(r => setTimeout(r, 500));
                    this.renderStoryReview(this.currentStoryData);
                    this.showSection('review');
                    this.setNavActive('nav-create');
                } else {
                    throw new Error('잘못된 응답 형식');
                }
            }

        } catch (error) {
            console.error('스토리 생성 실패:', error);
            this.showToast('스토리 생성 중 오류가 발생했습니다. 다시 시도해주세요.', 'error');
            this.showSection('input');
            this.isGenerating = false;
        } finally {
            btn.disabled = false;
            btn.innerHTML = originalBtnText;
        }
    }

    // ===== 입력 모드 토글 =====
    _setInputMode(mode) {
        this._inputMode = mode;
        const aiBtn = document.getElementById('mode-ai-story');
        const scriptBtn = document.getElementById('mode-direct-script');
        const aiInputs = document.getElementById('ai-story-inputs');
        const scriptInputs = document.getElementById('direct-script-inputs');
        const submitBtn = document.getElementById('generate-story-btn');

        if (mode === 'script') {
            aiBtn.style.background = 'rgba(255,255,255,0.05)';
            aiBtn.style.color = 'rgba(255,255,255,0.5)';
            scriptBtn.style.background = 'rgba(139,92,246,0.3)';
            scriptBtn.style.color = '#c4b5fd';
            if (aiInputs) aiInputs.style.display = 'none';
            if (scriptInputs) scriptInputs.style.display = '';
            if (submitBtn) submitBtn.innerHTML = '<span class="btn-icon">📋</span> 씬 분할 + 프롬프트 생성 (1단계)';
        } else {
            aiBtn.style.background = 'rgba(139,92,246,0.3)';
            aiBtn.style.color = '#c4b5fd';
            scriptBtn.style.background = 'rgba(255,255,255,0.05)';
            scriptBtn.style.color = 'rgba(255,255,255,0.5)';
            if (aiInputs) aiInputs.style.display = '';
            if (scriptInputs) scriptInputs.style.display = 'none';
            if (submitBtn) submitBtn.innerHTML = '<span class="btn-icon">📝</span> 스토리 생성 (1단계)';
        }
    }

    // ===== 스크립트 직접 입력 → 씬 분할 + 이미지 프롬프트 생성 =====
    async startScriptGeneration() {
        // 크레딧 사전 확인
        if (typeof checkCreditsBeforeAction === 'function') {
            const ok = await checkCreditsBeforeAction('script_video');
            if (!ok) return;
        }

        const scriptText = document.getElementById('direct-script')?.value?.trim();
        if (!scriptText) {
            this.showToast('스크립트를 입력해주세요.', 'error');
            return;
        }

        const formData = new FormData(document.getElementById('generate-form'));
        const btn = document.getElementById('generate-story-btn');
        const originalBtnText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = '<span class="btn-icon">⏳</span> 씬 분할 중...';

        const selectedContentType2 = document.getElementById('content_type')?.value || 'fiction';
        const requestData = {
            script: scriptText,
            content_type: selectedContentType2,
            genre: CONTENT_TYPE_TO_GENRE[selectedContentType2] || formData.get('genre') || 'emotional',
            mood: formData.get('mood'),
            style: formData.get('style'),
            voice: formData.get('voice'),
            duration: parseInt(formData.get('duration')),
            platform: formData.get('platform'),
            character_ethnicity: formData.get('character_ethnicity') || 'auto',
            hook_scene1_video: document.getElementById('hook_scene1_video')?.checked || false,
            ffmpeg_kenburns: document.getElementById('ffmpeg_kenburns')?.checked || true,
            ffmpeg_audio_ducking: document.getElementById('ffmpeg_audio_ducking')?.checked || false,
            subtitle_burn_in: document.getElementById('subtitle_burn_in')?.checked || true,
            context_carry_over: document.getElementById('context_carry_over')?.checked || true,
            optimization_pack: document.getElementById('optimization_pack')?.checked || true,
        };

        this.currentRequestParams = requestData;

        try {
            this.showSection('progress');
            this.updateStepStatus('story', '스크립트를 분석하고 있습니다...');
            document.getElementById('status-message').textContent = '스크립트를 씬으로 분할하고 이미지 프롬프트를 생성합니다...';
            document.getElementById('progress-percentage').textContent = '10%';
            document.getElementById('progress-bar').style.width = '10%';

            const scriptMessages = [
                { pct: 15, msg: '스크립트를 분석하고 있습니다...' },
                { pct: 30, msg: '장면을 분할하고 있습니다...' },
                { pct: 50, msg: 'AI가 이미지 프롬프트를 생성하고 있습니다...' },
                { pct: 65, msg: '비주얼 일관성을 확인하고 있습니다...' },
                { pct: 78, msg: '거의 완료되었습니다...' },
            ];
            let msgIndex = 0;
            const progressInterval = setInterval(() => {
                if (msgIndex < scriptMessages.length) {
                    const { pct, msg } = scriptMessages[msgIndex];
                    this.updateProgress(pct, msg);
                    msgIndex++;
                }
            }, 3000);

            const workerUrl = this.getWorkerUrl();
            const railwayUrl = this.getApiBaseUrl();
            let response;

            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 120000);
                response = await fetch(`${workerUrl}/api/generate/from-script`, {
                    method: 'POST',
                    headers: this.getAuthHeaders(),
                    body: JSON.stringify(requestData),
                    signal: controller.signal
                });
                clearTimeout(timeoutId);
                if (!response.ok) throw new Error(`Worker returned ${response.status}`);
            } catch (workerError) {
                console.warn('[Script] Worker 실패, Railway 폴백:', workerError.message);
                this.updateProgress(40, 'Worker 타임아웃 - 백엔드로 재시도 중...');
                response = await fetch(`${railwayUrl}/api/generate/from-script`, {
                    method: 'POST',
                    headers: this.getAuthHeaders(),
                    body: JSON.stringify(requestData)
                });
            }

            clearInterval(progressInterval);

            if (!response.ok) {
                if (typeof handleApiError === 'function' && await handleApiError(response.clone(), 'script_video')) {
                    return;
                }
                let errorMsg = '스크립트 처리 실패';
                try {
                    const error = await response.json();
                    errorMsg = error.detail || error.error || errorMsg;
                } catch (e) { }
                throw new Error(errorMsg);
            }

            this.updateProgress(90, '씬 분할 완료! 결과를 불러오는 중...');
            const result = await response.json();

            if (result.story_data) {
                this.updateProgress(100, '스크립트 분석이 완료되었습니다!');
                this.currentStoryData = result.story_data;
                this.currentRequestParams = requestData;

                // Store detected speakers
                if (result.detected_speakers) {
                    this.currentStoryData.detected_speakers = result.detected_speakers;
                }

                // 크레딧 차감 반영
                if (typeof deductLocalCredits === 'function') deductLocalCredits('script_video');

                await new Promise(r => setTimeout(r, 500));
                this.renderStoryReview(this.currentStoryData);
                this.showSection('review');
                this.setNavActive('nav-create');
            } else {
                throw new Error('잘못된 응답 형식');
            }

        } catch (error) {
            console.error('스크립트 처리 실패:', error);
            this.showToast('스토리 생성 중 오류가 발생했습니다. 다시 시도해주세요.', 'error');
            this.showSection('input');
        } finally {
            btn.disabled = false;
            btn.innerHTML = originalBtnText;
        }
    }

    // 스토리 생성 완료 폴링
    async pollStoryStatus(projectId, workerUrl, maxAttempts = 60) {
        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            try {
                const response = await fetch(`${workerUrl}/api/status/${projectId}`);

                if (!response.ok) {
                    console.warn(`Polling attempt ${attempt + 1} failed`);
                    await this.sleep(2000);
                    continue;
                }

                const status = await response.json();

                if (status.status === 'story_ready' && status.video_url) {
                    // video_url에 스토리 데이터가 JSON 문자열로 저장됨
                    try {
                        return JSON.parse(status.video_url);
                    } catch (e) {
                        console.error('Story data parse error:', e);
                        return null;
                    }
                } else if (status.status === 'failed') {
                    throw new Error(status.error_message || '스토리 생성 실패');
                }

                // 아직 처리 중이면 2초 대기
                await this.sleep(2000);

            } catch (error) {
                console.error(`Polling error (attempt ${attempt + 1}):`, error);
                await this.sleep(2000);
            }
        }

        return null; // 타임아웃
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }


    renderStoryReview(storyData) {
        const grid = document.getElementById('review-scene-grid');
        grid.innerHTML = '';

        // Clear voice selection slot
        const voiceSlot = document.getElementById('voice-selection-slot');
        if (voiceSlot) voiceSlot.innerHTML = '';

        document.getElementById('review-title').value = storyData.title;

        // Detect speakers
        this._detectedSpeakers = storyData.detected_speakers || ['narrator'];

        // STORYCUT 토큰 → 실제 캐릭터 이름 매핑 (character_sheet 기반)
        const tokenToName = {};
        const charSheet = storyData.character_sheet || {};
        Object.entries(charSheet).forEach(([token, data]) => {
            if (data && data.name) tokenToName[token] = data.name;
        });
        const replaceTokens = (text) => {
            if (!text) return '';
            return text.replace(/STORYCUT_\w+/g, (tok) => tokenToName[tok] || tok);
        };

        storyData.scenes.forEach((scene, index) => {
            const card = document.createElement('div');
            card.className = 'review-card';
            card.dataset.sceneId = scene.scene_id;

            // Format narration with speaker highlighting
            const narrationText = scene.narration || scene.tts_script || scene.sentence || '';
            const highlightedHtml = this._highlightSpeakerTags(narrationText);

            // Visual prompt: image_prompt 우선, 없으면 visual_description/prompt
            // STORYCUT 토큰은 실제 캐릭터 이름으로 치환하여 표시
            const rawVisual = scene.image_prompt || scene.visual_description || scene.prompt || '';
            const visualText = replaceTokens(rawVisual);

            card.innerHTML = `
                <div class="review-card-header">
                    <span>Scene ${scene.scene_id}</span>
                    <span>${scene.duration_sec}s</span>
                </div>

                <label>Narration / Dialogue</label>
                ${highlightedHtml ? `<div class="dialogue-preview">${highlightedHtml}</div>` : ''}
                <textarea class="review-textarea narration-input" data-idx="${index}">${escapeHtml(narrationText)}</textarea>

                <label>Visual Prompt</label>
                <textarea class="review-textarea visual-textarea visual-input" data-idx="${index}">${escapeHtml(visualText)}</textarea>
            `;
            grid.appendChild(card);
        });

        // Render voice selection UI if multiple speakers detected
        this._renderVoiceSelectionUI();
    }

    _highlightSpeakerTags(text) {
        if (!text) return '';
        const speakerColors = {
            'narrator': '#9ca3af',
            'male_1': '#60a5fa', 'male_2': '#38bdf8', 'male_3': '#22d3ee',
            'female_1': '#f472b6', 'female_2': '#fb7185', 'female_3': '#e879f9',
        };

        const hasTags = text.includes('[');
        return text.split('\n').map(line => {
            const match = line.match(/^\[([^\]]+)\](?:\(([^)]*)\))?\s*(.*)/);
            if (match) {
                const speaker = escapeHtml(match[1]);
                const emotion = escapeHtml(match[2] || '');
                const dialogue = escapeHtml(match[3]);
                const color = speakerColors[match[1]] || '#a78bfa';
                const emotionBadge = emotion ? `<span style="color:${color};opacity:0.6;font-size:11px">(${emotion})</span>` : '';
                return `<span style="color:${color};font-weight:600">[${speaker}]</span>${emotionBadge} ${dialogue}`;
            }
            // Show plain text lines as narrator (gray) if no tags in entire text
            if (!hasTags && line.trim()) {
                return `<span style="color:#9ca3af">${escapeHtml(line)}</span>`;
            }
            return escapeHtml(line);
        }).filter(l => l.trim()).join('<br>');
    }

    async _renderVoiceSelectionUI() {
        // Load available voices
        if (!this._availableVoices) {
            try {
                const baseUrl = this.getApiBaseUrl();
                const resp = await fetch(`${baseUrl}/api/voices`);
                if (resp.ok) {
                    const data = await resp.json();
                    this._availableVoices = data.voices || [];
                }
            } catch (e) {
                console.warn('Failed to load voices:', e);
                this._availableVoices = [];
            }
        }

        // Render into dedicated slot
        const slot = document.getElementById('voice-selection-slot');
        if (!slot) return;
        slot.innerHTML = '';

        const speakerColors = {
            'narrator': '#9ca3af',
            'male_1': '#60a5fa', 'male_2': '#38bdf8', 'male_3': '#22d3ee',
            'female_1': '#f472b6', 'female_2': '#fb7185', 'female_3': '#e879f9',
        };

        const speakerCount = this._detectedSpeakers.length;
        const hasMultiple = speakerCount > 1;

        // Build panel
        const panel = document.createElement('div');
        panel.className = 'app-card voice-selection-panel';

        // Header with speaker badges
        const speakerDisplayNames = {
            'narrator': '내레이터',
            'male_1': '남성 1', 'male_2': '남성 2', 'male_3': '남성 3',
            'female_1': '여성 1', 'female_2': '여성 2', 'female_3': '여성 3',
        };
        const badgesHtml = this._detectedSpeakers.map(s => {
            const color = speakerColors[s] || '#a78bfa';
            const displayName = speakerDisplayNames[s] || s.replace('_', ' ');
            return `<span class="speaker-badge" style="--badge-color:${color}">${displayName}</span>`;
        }).join('');

        panel.innerHTML = `
            <div class="voice-panel-header">
                <div class="voice-panel-title">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z"/><path d="M19 10v2a7 7 0 01-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
                    <span>음성 배정</span>
                </div>
                <div class="speaker-badges">${badgesHtml}</div>
                ${hasMultiple
                ? '<p class="voice-panel-hint">AI가 여러 화자를 감지했습니다. 각 화자에 맞는 음성을 선택하세요.</p>'
                : '<p class="voice-panel-hint">내레이터 음성을 선택하세요. 미리듣기 버튼으로 확인할 수 있습니다.</p>'}
            </div>
            <div class="voice-assignments" id="voice-assignments"></div>
        `;
        slot.appendChild(panel);

        const assignmentsDiv = panel.querySelector('#voice-assignments');
        const defaultVoice = document.getElementById('voice')?.value || 'uyVNoMrnUku1dZyVEXwD';

        // Smart default mapping: auto-assign gender-appropriate voices
        const maleDefaults = ['s07IwTCOrCDCaETjUVjx', 'm5qndnI7u4OAdXhH0Mr5', 'UgBBYS2sOqTuMpoF3BR0', '3MTvEr8xCMCC2mL9ujrI', '8jHHF8rMqMlg8if2mOUe'];
        const femaleDefaults = ['uyVNoMrnUku1dZyVEXwD', 'sf8Bpb1IU97NI9BHSMRf', '19STyYD15bswVz51nqLf', 'p4w8j6zCUDJ0nGJ3okKs', 'ajOR9IDAaubDK5qtLUqQ'];
        let maleIdx = 0, femaleIdx = 0;

        // Curated voice list (always use this, not full API list)
        const curatedVoices = [
            // ── 여성 (Female) ──
            { id: 'uyVNoMrnUku1dZyVEXwD', name: 'Anna Kim - 차분하고 따뜻한', gender: 'F' },
            { id: 'sf8Bpb1IU97NI9BHSMRf', name: 'Rosa Oh - 침착하고 세련된', gender: 'F' },
            { id: '19STyYD15bswVz51nqLf', name: 'Samara X - 또렷하고 따뜻한', gender: 'F' },
            // ── 남성 (Male) ──
            { id: 's07IwTCOrCDCaETjUVjx', name: 'Hyunbin - 차분하고 명확한', gender: 'M' },
            { id: 'UgBBYS2sOqTuMpoF3BR0', name: 'Mark - 자연스럽고 편안한', gender: 'M' },
            { id: '3MTvEr8xCMCC2mL9ujrI', name: 'June - 젊고 활기찬', gender: 'M' },
        ];

        // Always use curated list (API returns too many unfiltered voices)
        const voiceList = curatedVoices;

        for (const speaker of this._detectedSpeakers) {
            const color = speakerColors[speaker] || '#a78bfa';
            const displayName = speakerDisplayNames[speaker] || speaker.replace('_', ' ');

            // Determine smart default voice for this speaker
            let smartDefault = defaultVoice;
            if (speaker.startsWith('male')) {
                smartDefault = maleDefaults[maleIdx % maleDefaults.length];
                maleIdx++;
            } else if (speaker.startsWith('female')) {
                smartDefault = femaleDefaults[femaleIdx % femaleDefaults.length];
                femaleIdx++;
            }

            const row = document.createElement('div');
            row.className = 'voice-assignment-row';

            // Speaker label with color dot
            const labelDiv = document.createElement('div');
            labelDiv.className = 'voice-speaker-label';
            labelDiv.innerHTML = `<span class="voice-speaker-dot" style="background:${color}"></span><span>${displayName}</span>`;
            row.appendChild(labelDiv);

            // Voice select with gender groups
            const select = document.createElement('select');
            select.className = 'form-select voice-assignment-select';
            select.dataset.speaker = speaker;

            const femaleGroup = document.createElement('optgroup');
            femaleGroup.label = '-- 여성 --';
            const maleGroup = document.createElement('optgroup');
            maleGroup.label = '-- 남성 --';

            for (const v of voiceList) {
                const opt = document.createElement('option');
                opt.value = v.id;
                opt.textContent = v.name;
                if (v.id === smartDefault) opt.selected = true;
                if (v.gender === 'F') femaleGroup.appendChild(opt);
                else maleGroup.appendChild(opt);
            }
            select.appendChild(femaleGroup);
            select.appendChild(maleGroup);

            select.addEventListener('change', () => {
                const selectedOpt = select.options[select.selectedIndex];
                this._characterVoices[speaker] = {
                    speaker, voice_id: select.value, voice_name: selectedOpt?.textContent || '',
                };
            });
            // Set initial value
            const initialOpt = select.options[select.selectedIndex];
            this._characterVoices[speaker] = {
                speaker, voice_id: select.value, voice_name: initialOpt?.textContent || '',
            };
            row.appendChild(select);

            // Preview button
            const previewBtn = document.createElement('button');
            previewBtn.type = 'button';
            previewBtn.className = 'btn btn-secondary btn-small voice-preview-btn';
            previewBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg> 미리듣기';
            previewBtn.addEventListener('click', async () => {
                const voiceId = select.value;
                previewBtn.disabled = true;
                previewBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="10" y1="15" x2="10" y2="9"/><line x1="14" y1="15" x2="14" y2="9"/></svg> 재생 중...';
                try {
                    const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
                    const base = isLocal ? '' : 'https://web-production-bb6bf.up.railway.app';
                    const resp = await fetch(`${base}/api/sample-voice/${voiceId}?t=${Date.now()}`);
                    if (!resp.ok) throw new Error(`${resp.status}`);
                    const blob = await resp.blob();
                    const url = URL.createObjectURL(blob);
                    const audio = new Audio(url);
                    await audio.play();
                    audio.onended = () => {
                        previewBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg> 미리듣기';
                        previewBtn.disabled = false;
                        URL.revokeObjectURL(url);
                    };
                } catch (e) {
                    console.error('Preview failed:', e);
                    previewBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg> 미리듣기';
                    previewBtn.disabled = false;
                }
            });
            row.appendChild(previewBtn);

            assignmentsDiv.appendChild(row);
        }
    }

    updateImageModelHint() {
        const hint = document.getElementById('image_model_hint');
        if (hint) hint.style.display = 'none';
    }

    // ==================== Step 2: 영상 생성 시작 ====================
    // 리뷰 섹션의 영상 생성 옵션을 currentRequestParams에 반영
    _syncGenerationOptions() {
        if (!this.currentRequestParams) return;
        this.currentRequestParams.ffmpeg_kenburns = document.getElementById('ffmpeg_kenburns')?.checked ?? true;
        this.currentRequestParams.ffmpeg_audio_ducking = document.getElementById('ffmpeg_audio_ducking')?.checked ?? false;
        this.currentRequestParams.subtitle_burn_in = document.getElementById('subtitle_burn_in')?.checked ?? true;
        this.currentRequestParams.context_carry_over = document.getElementById('context_carry_over')?.checked ?? true;
        this.currentRequestParams.optimization_pack = document.getElementById('optimization_pack')?.checked ?? false;
        this.currentRequestParams.film_look = document.getElementById('film_look')?.checked ?? false;
        // image_model은 select
        const imageModel = document.getElementById('image_model')?.value || 'standard';
        this.currentRequestParams.image_model = imageModel;
    }

    async startFinalGeneration() {
        if (!this.currentStoryData) return;

        // 이미 생성 중이면 중복 생성 방지
        if (this.isGenerating) {
            this.showToast('이미 영상 생성이 진행 중입니다.', 'warning');
            return;
        }

        // 리뷰 섹션 옵션 동기화
        this._syncGenerationOptions();

        // 수정된 스토리 데이터 수집
        const titleInput = document.getElementById('review-title').value;
        this.currentStoryData.title = titleInput;

        const narrationInputs = document.querySelectorAll('.narration-input');
        const visualInputs = document.querySelectorAll('.visual-input');

        narrationInputs.forEach((input, idx) => {
            this.currentStoryData.scenes[idx].narration = input.value;
            this.currentStoryData.scenes[idx].sentence = input.value;
        });

        visualInputs.forEach((input, idx) => {
            this.currentStoryData.scenes[idx].visual_description = input.value;
            this.currentStoryData.scenes[idx].prompt = input.value;
            this.currentStoryData.scenes[idx].image_prompt = input.value;
        });

        // 생성 시작
        try {
            this.isGenerating = true;
            this._requestNotificationPermission();
            this.showSection('progress');
            const progressTitle = document.getElementById('progress-title');
            if (progressTitle) progressTitle.textContent = '⏳ 영상 생성 중...';

            let urlToUse = this.getApiBaseUrl();

            const headers = this.getAuthHeaders();

            // Collect character_voices from voice selection UI
            const characterVoices = Object.values(this._characterVoices || {});

            const payload = {
                request_params: this.currentRequestParams,
                story_data: this.currentStoryData,
                character_voices: characterVoices.length > 0 ? characterVoices : [],
            };

            this.addLog('INFO', '📤 영상 생성 요청 전송 중...');

            const response = await fetch(`${urlToUse}/api/generate/video`, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || error.error || '영상 생성 시작 실패');
            }

            const result = await response.json();
            this.projectId = result.project_id;

            this.addLog('INFO', `✅ 영상 생성 요청 접수됨 (Project ID: ${this.projectId})`);
            this.addLog('INFO', '⏳ 서버에서 영상 생성 중... 진행 상황을 아래에서 확인하세요.');

            // 진행률 초기화 (스토리는 이미 완료 → 장면 처리부터 시작)
            this.resetProgress('scenes');

            // WebSocket 연결 시도
            this.connectWebSocket(this.projectId);

            // Polling 시작 (WebSocket 실패 시 백업)
            setTimeout(() => {
                if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
                    this.addLog('INFO', 'Polling으로 상태 확인 중...');
                    this.startPolling(this.projectId);
                }
            }, 2000);

        } catch (error) {
            console.error('영상 생성 요청 실패:', error);
            this.addLog('ERROR', `❌ 오류: ${error.message}`);
            this.showToast('영상 생성에 실패했습니다. 다시 시도해주세요.', 'error');
            this.isGenerating = false;
            this.showSection('review');
        }
    }

    resetProgress(startFromStep = 'story') {
        this.updateProgress(5, '초기화 중...');

        // 단계 초기화
        document.querySelectorAll('.step').forEach(el => {
            el.classList.remove('active', 'completed');
            el.querySelector('.step-status').textContent = '대기 중';
        });

        // 시작 단계 설정
        const stepOrder = ['story', 'scenes', 'compose', 'optimize'];
        const startIndex = stepOrder.indexOf(startFromStep);

        // 시작 단계 이전 단계들은 완료 처리
        for (let i = 0; i < startIndex; i++) {
            const stepEl = document.querySelector(`[data-step="${stepOrder[i]}"]`);
            if (stepEl) {
                stepEl.classList.add('completed');
                stepEl.querySelector('.step-status').textContent = '완료';
            }
        }

        // 시작 단계 활성화
        const startStep = document.querySelector(`[data-step="${startFromStep}"]`);
        if (startStep) {
            startStep.classList.add('active');
            startStep.querySelector('.step-status').textContent = '진행 중';
        }
    }

    // ==================== Polling: 상태 주기적 확인 ====================
    startPolling(projectId) {
        this.addLog('INFO', '📊 Polling 시작 (2초마다 상태 확인)');

        // 이미 polling 중이면 중복 방지
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
        }

        // 2초마다 상태 확인
        this.pollingFailCount = 0;
        this.pollingStaleCount = 0;
        this._lastPollingProgress = -1;
        this.pollingInterval = setInterval(async () => {
            try {
                let urlToUse = this.getApiBaseUrl();
                const response = await fetch(`${urlToUse}/api/status/${projectId}`);

                if (!response.ok) {
                    console.error(`Status check failed: ${response.status}`);
                    return;
                }

                // 연결 성공 시 실패 카운터 리셋
                this.pollingFailCount = 0;

                const data = await response.json();

                // 상태에 따른 처리
                if (data.status === 'completed') {
                    this.addLog('SUCCESS', '영상 완성');
                    this.updateProgress(100, '완료');
                    this.updateStepStatus('complete', '완료');
                    this.stopPolling();
                    this.isGenerating = false;
                    this._sendCompletionNotification('StoryCut 영상 완성!', `"${data.title || '영상'}"이 완성되었습니다. 클릭해서 확인하세요.`);

                    // 1초 대기 후 결과 페이지로 이동
                    setTimeout(() => {
                        this.handleComplete({
                            project_id: projectId,
                            title: data.title
                        });
                    }, 1000);

                } else if (data.status === 'failed' || data.error_message) {
                    this.addLog('ERROR', `❌ 오류 발생: ${data.error_message}`);
                    this.updateProgress(0, '실패');
                    this.stopPolling();
                    this.isGenerating = false;
                    this.showToast('영상 생성에 실패했습니다. 다시 시도해주세요.', 'error');

                } else if (data.status === 'processing' || data.status === 'images_ready' || data.status === 'generating') {
                    // 진행 중 상태 업데이트
                    const progress = data.progress || 25;
                    const message = data.message || '영상 생성 중...';

                    this.updateProgress(progress, message);

                    // 진행률 변화 감지 (3분간 변화 없으면 타임아웃)
                    if (progress === this._lastPollingProgress) {
                        this.pollingStaleCount++;
                        if (this.pollingStaleCount >= 90) { // 90 * 2초 = 3분
                            this.addLog('ERROR', '❌ 영상 생성이 응답하지 않습니다. 다시 시도해주세요.');
                            this.updateProgress(0, '타임아웃');
                            this.stopPolling();
                            this.isGenerating = false;
                            this.showToast('영상 생성이 응답하지 않습니다. 페이지를 새로고침 후 다시 시도해주세요.', 'error');
                            return;
                        }
                    } else {
                        this.pollingStaleCount = 0;
                        this._lastPollingProgress = progress;
                    }

                    // 진행률 기반 단계 추정
                    if (progress < 20) {
                        this.updateStepStatus('story', '스토리 생성 중');
                    } else if (progress < 60) {
                        this.updateStepStatus('scenes', '장면 처리 중');
                    } else if (progress < 90) {
                        this.updateStepStatus('compose', '영상 합성 중');
                    } else if (progress < 100) {
                        this.updateStepStatus('optimize', '최적화 중');
                    }
                }

            } catch (error) {
                this.pollingFailCount++;
                console.warn(`Polling error (${this.pollingFailCount}/10):`, error.message);

                if (this.pollingFailCount >= 10) {
                    this.stopPolling();
                    this.isGenerating = false;
                    this.addLog('ERROR', '서버 연결이 끊어졌습니다. 페이지를 새로고침 해주세요.');
                    this.updateProgress(0, '서버 연결 끊김');
                    this.showToast('서버 연결이 끊어졌습니다. 잠시 후 새로고침 해주세요.', 'error');
                }
            }
        }, 2000);
    }

    stopPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    }

    // ==================== WebSocket: 실시간 업데이트 ====================
    connectWebSocket(projectId) {
        let wsUrl;

        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            wsUrl = `ws://${window.location.host}`;
        } else {
            wsUrl = 'https://web-production-bb6bf.up.railway.app';
        }

        const wsProtocol = wsUrl.startsWith('https') ? 'wss' : 'ws';
        const wsHost = wsUrl.replace(/https?:\/\//, '').replace(/wss?:\/\//, '');
        const wsPath = `${wsProtocol}://${wsHost}/ws/${projectId}`;

        this.addLog('INFO', `🔗 WebSocket 연결 시도: ${wsPath}`);

        try {
            this.websocket = new WebSocket(wsPath);

            this.websocket.onopen = () => {
                this.addLog('SUCCESS', '✅ WebSocket 연결 성공! (실시간 업데이트 활성화)');
                // WebSocket 연결되면 polling 중단
                this.stopPolling();
            };

            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    if (data.type === 'progress') {
                        const progress = data.progress || 0;
                        const message = data.message || '';

                        this.addLog('PROGRESS', `[${progress}%] ${data.step}: ${message}`);
                        this.updateProgress(progress, message);

                        // 단계별 상태 업데이트
                        if (data.step.startsWith('scene')) {
                            this.updateStepStatus('scenes', message);
                        } else if (data.step === 'story') {
                            this.updateStepStatus('story', message);
                        } else if (data.step === 'compose') {
                            this.updateStepStatus('compose', message);
                        } else if (data.step === 'optimize') {
                            this.updateStepStatus('optimize', message);
                        }

                        // 완료 감지
                        if (data.progress === 100 || data.step === 'complete') {
                            this.addLog('SUCCESS', '영상 완성');
                            this.updateStepStatus('complete', '완료');
                            this._sendCompletionNotification('StoryCut 영상 완성!', `"${data.data?.title || '영상'}"이 완성되었습니다. 클릭해서 확인하세요.`);
                            setTimeout(() => {
                                this.handleComplete({
                                    project_id: projectId,
                                    title: data.data?.title
                                });
                            }, 1000);
                        }
                    }
                } catch (e) {
                    console.error('WebSocket message parse error:', e);
                }
            };

            this.websocket.onerror = (error) => {
                this.addLog('WARNING', '⚠️ WebSocket 오류 - Polling으로 폴백');
                console.error('WebSocket error:', error);
            };

            this.websocket.onclose = () => {
                this.addLog('INFO', '📴 WebSocket 연결 종료');
                // WebSocket 종료 시 polling 재시작
                if (this.isGenerating && !this.pollingInterval) {
                    setTimeout(() => {
                        if (this.isGenerating) {
                            this.startPolling(projectId);
                        }
                    }, 2000);
                }
            };

        } catch (error) {
            this.addLog('ERROR', `❌ WebSocket 연결 실패: ${error.message}`);
        }
    }

    // ==================== 완료 처리 ====================
    handleComplete(data) {
        this.stopPolling();
        this.isGenerating = false;

        this.addLog('INFO', '📥 결과 정보 가져오는 중...');

        // 완료 섹션으로 전환 (결과 로드 후)
        setTimeout(() => {
            this.fetchAndShowResults(data.project_id);
        }, 500);
    }

    async fetchAndShowResults(projectId) {
        const maxRetries = 3;
        let lastError = null;

        for (let i = 0; i < maxRetries; i++) {
            try {
                let urlToUse = this.getApiBaseUrl();
                const targetUrl = `${urlToUse}/api/manifest/${projectId}`;

                this.addLog('INFO', `📥 결과 데이터 요청 중... (시도 ${i + 1}/${maxRetries})`);

                const fetchOpts = {};
                const token = localStorage.getItem('token');
                if (token) fetchOpts.headers = { 'Authorization': `Bearer ${token}` };
                const response = await fetch(targetUrl, fetchOpts);

                if (!response.ok) {
                    const errorText = await response.text();
                    console.error(`[Fetch Error] Status: ${response.status} ${response.statusText}, Body: ${errorText}`);
                    throw new Error(`Manifest 로드 실패 (${response.status}): ${errorText || response.statusText}`);
                }

                const manifest = await response.json();

                this.showResults({
                    project_id: projectId,
                    title: manifest.title,
                    status: manifest.status,
                    error_message: manifest.error_message,
                    title_candidates: manifest.outputs?.title_candidates,
                    thumbnail_texts: manifest.outputs?.thumbnail_texts,
                    hashtags: manifest.outputs?.hashtags,
                    video_path: manifest.outputs?.final_video_path,
                    server_url: urlToUse,
                    platform: manifest.input?.target_platform || 'youtube_long',
                    hook_text: manifest.hook_text || '',
                });

                return; // 성공 시 종료

            } catch (error) {
                console.error(`Attempt ${i + 1} failed:`, error);
                lastError = error;
                // 마지막 시도가 아니면 1초 대기
                if (i < maxRetries - 1) {
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }
            }
        }

        // 모든 시도 실패 시
        this.addLog('ERROR', `❌ 결과 가져오기 최종 실패: ${lastError.message}`);
        this.showResultError(projectId, `결과를 불러오지 못했습니다. (서버 응답 없음 또는 파일 누락)\n내용: ${lastError.message}`);
    }

    showResultError(projectId, message) {
        document.getElementById('result-section').classList.remove('hidden');
        document.getElementById('result-header-text').textContent = "⚠️ 프로젝트 로드 실패";
        document.getElementById('result-video-container').innerHTML = `<div class="error-box"><p>${escapeHtml(message)}</p></div>`;
    }

    async showResults(data) {
        // 결과 섹션 표시
        this.showSection('result');
        this.setNavActive('nav-create');

        // 헤더 텍스트 업데이트
        const headerText = document.getElementById('result-header-text');
        const videoContainer = document.getElementById('result-video-container');
        const downloadBtn = document.getElementById('download-btn');
        const isShorts = data.platform === 'youtube_shorts';

        // Shorts 모드 UI 토글
        const shortsHookArea = document.getElementById('shorts-hook-area');
        const youtubeUploadBtn = document.getElementById('youtube-upload-btn');
        const resultSection = document.getElementById('result-section');

        if (isShorts) {
            videoContainer.classList.add('shorts-player');
            resultSection.classList.add('shorts-mode');
        } else {
            videoContainer.classList.remove('shorts-player');
            resultSection.classList.remove('shorts-mode');
        }

        // Hook text 표시
        if (shortsHookArea) {
            if (isShorts && data.hook_text) {
                shortsHookArea.style.display = '';
                document.getElementById('shorts-hook-text').textContent = data.hook_text;
            } else {
                shortsHookArea.style.display = 'none';
            }
        }

        // 유튜브 업로드 버튼 표시
        if (youtubeUploadBtn) {
            youtubeUploadBtn.style.display = data.status === 'completed' ? 'inline-flex' : 'none';
        }

        // 기본 정보
        document.getElementById('result-project-id').textContent = data.project_id;
        document.getElementById('result-title').textContent = data.title_candidates?.[0] || data.title || '제목 없음';

        let backendUrl = data.server_url;
        if (!backendUrl) {
            backendUrl = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
                ? '' : 'https://web-production-bb6bf.up.railway.app';
        }

        // 상태별 UI 처리
        if (data.status === 'completed') {
            headerText.textContent = isShorts ? "쇼츠 완성" : "영상 완성";

            // 비디오 플레이어 복구/설정
            videoContainer.innerHTML = '<video id="result-video" controls style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);"></video>';
            const video = document.getElementById('result-video');
            video.src = `${backendUrl}/api/stream/${data.project_id}`;

            // 다운로드 버튼 활성화
            downloadBtn.style.display = 'inline-flex';
            downloadBtn.href = `${backendUrl}/api/download/${data.project_id}`;
            downloadBtn.download = `storycut_${data.project_id}.mp4`;

        } else if (data.status === 'processing') {
            headerText.textContent = "⏳ 영상 생성 중...";
            videoContainer.innerHTML = `
                <div style="text-align: center; padding: 40px; background: rgba(255,255,255,0.05); border-radius: 8px;">
                    <span style="font-size: 48px; display: block; margin-bottom: 20px;">🎬</span>
                    <h3>아직 영상이 만들어지고 있습니다.</h3>
                    <p>잠시 후 다시 확인해주세요.</p>
                </div>`;
            downloadBtn.style.display = 'none';

        } else {
            // failed or unknown
            headerText.textContent = "❌ 영상 생성 실패";
            videoContainer.innerHTML = `
                <div style="text-align: center; padding: 40px; background: rgba(255,50,50,0.1); border-radius: 8px;">
                    <span style="font-size: 48px; display: block; margin-bottom: 20px;">⚠️</span>
                    <h3>생성 도중 오류가 발생했습니다.</h3>
                    <p>${escapeHtml(data.error_message || '알 수 없는 오류')}</p>
                </div>`;
            downloadBtn.style.display = 'none';
        }

        // 최적화 패키지 (아래는 공통)
        if (data.title_candidates?.length > 0) this.displayTitleCandidates(data.title_candidates);
        if (data.thumbnail_texts?.length > 0) this.displayThumbnailTexts(data.thumbnail_texts);
        if (data.hashtags?.length > 0) this.displayHashtags(data.hashtags);

        // 씬 목록 로드
        if (data._fromArchive && data._scenes) {
            // 보관함에서 온 경우: 매니페스트의 scenes로 읽기 전용 패널 렌더링
            this.renderArchiveImagePanel(data._scenes, data.project_id, false);
        } else {
            await this.loadSceneList(data.project_id);
        }

        this.addLog('SUCCESS', '모든 정보 로드 완료!');
    }

    // ==================== UI 표시 함수 ====================
    updateProgress(progress, message) {
        const progressBar = document.getElementById('progress-bar');
        const progressPercentage = document.getElementById('progress-percentage');
        const statusMessage = document.getElementById('status-message');

        progressBar.style.width = `${Math.min(progress, 100)}%`;
        progressPercentage.textContent = `${Math.min(progress, 100)}%`;
        statusMessage.textContent = message;
    }

    updateStepStatus(step, message) {
        document.querySelectorAll('.step').forEach(el => {
            el.classList.remove('active', 'completed');
        });

        let currentStepElement = null;

        if (step === 'story') {
            currentStepElement = document.querySelector('[data-step="story"]');
        } else if (step === 'scenes') {
            currentStepElement = document.querySelector('[data-step="scenes"]');
        } else if (step === 'compose') {
            currentStepElement = document.querySelector('[data-step="compose"]');
        } else if (step === 'optimize') {
            currentStepElement = document.querySelector('[data-step="optimize"]');
        }

        if (currentStepElement) {
            currentStepElement.classList.add('active');
            const statusEl = currentStepElement.querySelector('.step-status');
            statusEl.textContent = message;

            // 이전 단계들은 완료로 표시
            let prev = currentStepElement.previousElementSibling;
            while (prev && prev.classList.contains('step')) {
                prev.classList.add('completed');
                prev.querySelector('.step-status').textContent = '완료';
                prev = prev.previousElementSibling;
            }
        }

        // 완료 시 모든 단계 완료로 표시
        if (step === 'complete') {
            document.querySelectorAll('.step').forEach(el => {
                el.classList.add('completed');
                el.querySelector('.step-status').textContent = '완료';
            });
        }
    }

    addLog(level, message) {
        const logContent = document.getElementById('log-content');
        const timestamp = new Date().toLocaleTimeString('ko-KR');

        const logEntry = document.createElement('div');
        const logLevel = level === 'ERROR' ? '❌' : level === 'SUCCESS' ? '✅' : level === 'WARNING' ? '⚠️' : level === 'INFO' ? 'ℹ️' : '▶️';
        logEntry.className = `log-entry log-${level.toLowerCase()}`;
        logEntry.innerHTML = `
            <span class="log-timestamp">[${timestamp}]</span>
            <span class="log-level">${logLevel}</span>
            <span class="log-message">${message}</span>
        `;
        logContent.appendChild(logEntry);

        // 스크롤을 최신 로그로
        logContent.scrollTop = logContent.scrollHeight;
    }

    displayTitleCandidates(titles) {
        const container = document.getElementById('title-candidates');
        container.innerHTML = '';

        titles.forEach((title, index) => {
            const item = document.createElement('div');
            item.className = 'candidate-item';
            item.textContent = `${index + 1}. ${title}`;
            item.style.cursor = 'pointer';
            item.title = '클릭하여 복사';
            item.onclick = () => {
                this.copyToClipboard(title);
                this.showToast('제목이 클립보드에 복사되었습니다!', 'success');
            };
            container.appendChild(item);
        });
    }

    displayThumbnailTexts(texts) {
        const container = document.getElementById('thumbnail-texts');
        container.innerHTML = '';

        texts.forEach((text, index) => {
            const item = document.createElement('div');
            item.className = 'candidate-item';
            item.textContent = `${index + 1}. ${text}`;
            item.style.cursor = 'pointer';
            item.title = '클릭하여 복사';
            item.onclick = () => {
                this.copyToClipboard(text);
                this.showToast('썸네일 문구가 클립보드에 복사되었습니다!', 'success');
            };
            container.appendChild(item);
        });
    }

    displayHashtags(hashtags) {
        const container = document.getElementById('hashtags');
        container.innerHTML = '';

        hashtags.forEach(tag => {
            const item = document.createElement('span');
            item.className = 'hashtag';
            item.textContent = tag;
            item.style.cursor = 'pointer';
            item.title = '클릭하여 복사';
            item.onclick = () => {
                this.copyToClipboard(tag);
                this.showToast('해시태그가 클립보드에 복사되었습니다!', 'success');
            };
            container.appendChild(item);
        });
    }

    copyToClipboard(text) {
        navigator.clipboard.writeText(text).catch(err => {
            console.error('클립보드 복사 실패:', err);
        });
    }

    showSection(sectionName) {
        // 폴링 인터벌 정리 (섹션 변경 시 불필요한 폴링 중단)
        if (sectionName !== 'image-preview' && this.imagePollingInterval) {
            clearInterval(this.imagePollingInterval);
            this.imagePollingInterval = null;
            console.log('[Cleanup] Image polling stopped');
        }
        if (sectionName !== 'mv-progress' && this.mvPollingInterval) {
            clearInterval(this.mvPollingInterval);
            this.mvPollingInterval = null;
            console.log('[Cleanup] MV polling stopped');
        }
        if (sectionName !== 'character-casting' && this.castingPollingInterval) {
            clearInterval(this.castingPollingInterval);
            this.castingPollingInterval = null;
            console.log('[Cleanup] Casting polling stopped');
        }

        // 모든 섹션 숨기기
        document.getElementById('input-section').classList.add('hidden');
        document.getElementById('progress-section').classList.add('hidden');
        document.getElementById('result-section').classList.add('hidden');
        document.getElementById('review-section').classList.add('hidden');
        document.getElementById('history-section').classList.add('hidden');
        document.getElementById('board-section')?.classList.add('hidden');
        document.getElementById('image-preview-section').classList.add('hidden');
        document.getElementById('character-casting-section')?.classList.add('hidden');
        // MV 섹션들
        document.getElementById('mv-section')?.classList.add('hidden');
        document.getElementById('mv-analysis-section')?.classList.add('hidden');
        document.getElementById('mv-progress-section')?.classList.add('hidden');
        document.getElementById('mv-image-review-section')?.classList.add('hidden');

        // 선택한 섹션 표시
        switch (sectionName) {
            case 'input':
                document.getElementById('input-section').classList.remove('hidden');
                break;
            case 'review':
                document.getElementById('review-section').classList.remove('hidden');
                break;
            case 'image-preview':
                document.getElementById('image-preview-section').classList.remove('hidden');
                break;
            case 'progress':
                document.getElementById('progress-section').classList.remove('hidden');
                break;
            case 'result':
                document.getElementById('result-section').classList.remove('hidden');
                break;
            case 'history':
                document.getElementById('history-section').classList.remove('hidden');
                break;
            // MV 섹션들
            case 'mv':
                document.getElementById('mv-section')?.classList.remove('hidden');
                break;
            case 'mv-analysis':
                document.getElementById('mv-analysis-section')?.classList.remove('hidden');
                break;
            case 'mv-progress':
                document.getElementById('mv-progress-section')?.classList.remove('hidden');
                break;
            case 'mv-image-review':
                document.getElementById('mv-image-review-section')?.classList.remove('hidden');
                break;
            case 'character-casting':
                document.getElementById('character-casting-section')?.classList.remove('hidden');
                break;
            case 'board':
                document.getElementById('board-section')?.classList.remove('hidden');
                break;
        }
    }

    resetUI() {
        this.projectId = null;
        this.currentStoryData = null;
        this.currentRequestParams = null;
        this.isGenerating = false;
        this.stopPolling();
        // 모든 폴링 인터벌 정리
        if (this.imagePollingInterval) {
            clearInterval(this.imagePollingInterval);
            this.imagePollingInterval = null;
        }
        if (this.mvPollingInterval) {
            clearInterval(this.mvPollingInterval);
            this.mvPollingInterval = null;
        }

        // 폼 초기화
        document.getElementById('generate-form').reset();
        document.getElementById('duration').value = 120;
        this.updateDurationDisplay();

        // 입력 섹션으로 이동
        this.showSection('input');
        this.setNavActive('nav-create');
    }

    // ==================== Scene Management 기능 ====================
    async loadSceneList(projectId) {
        try {
            const baseUrl = this.getApiBaseUrl();
            const response = await fetch(`${baseUrl}/api/projects/${projectId}/scenes`);

            if (!response.ok) {
                throw new Error('씬 목록을 가져올 수 없습니다');
            }

            const data = await response.json();
            this.renderSceneList(data.scenes, projectId);

        } catch (error) {
            console.error('씬 목록 로드 실패:', error);
            this.addLog('ERROR', `씬 목록 로드 실패: ${error.message}`);
        }
    }

    renderSceneList(scenes, projectId) {
        const grid = document.getElementById('result-scene-grid');
        if (!grid) return;

        grid.innerHTML = '';

        scenes.forEach(scene => {
            const card = document.createElement('div');
            card.className = 'result-scene-card';
            card.dataset.sceneId = scene.scene_id;

            // 상태에 따른 클래스 추가
            if (scene.status === 'regenerating') {
                card.classList.add('regenerating');
            } else if (scene.status === 'failed') {
                card.classList.add('failed');
            }

            // 상태 뱃지
            let statusBadge = '';
            if (scene.status === 'completed') {
                statusBadge = '<span class="scene-status-badge completed">✅ 완료</span>';
            } else if (scene.status === 'failed') {
                statusBadge = '<span class="scene-status-badge failed">❌ 실패</span>';
            } else if (scene.status === 'regenerating') {
                statusBadge = '<span class="scene-status-badge regenerating">🔄 재생성 중</span>';
            }

            // 에러 메시지
            let errorMsg = '';
            if (scene.error_message) {
                errorMsg = `<div class="scene-error-message">❌ ${escapeHtml(scene.error_message)}</div>`;
            }

            // 이미지 경로 추론: assets에 있으면 사용, 없으면 기본 경로 추정
            const imagePath = scene.assets?.image_path
                || `outputs/${projectId}/media/images/scene_${String(scene.scene_id).padStart(2, '0')}.png`;
            const imageUrl = this.resolveImageUrl(imagePath);

            card.innerHTML = `
                <div class="scene-card-header">
                    <span class="scene-card-title">Scene ${scene.scene_id}</span>
                    ${statusBadge}
                </div>

                <div class="scene-card-image">
                    <img src="${imageUrl}?t=${Date.now()}" alt="Scene ${scene.scene_id}"
                        onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';"
                        style="width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:6px;display:block;">
                    <div class="image-placeholder" style="display:none;">이미지 생성 대기</div>
                </div>

                <div class="scene-card-narration" data-scene-id="${scene.scene_id}" data-project-id="${projectId}">
                    <span class="narration-text">${scene.narration || '내레이션 없음'}</span>
                    <button class="btn-edit-narration" title="내레이션 수정">수정</button>
                </div>

                ${errorMsg}

                <div class="scene-card-actions">
                    ${scene.is_broll ? '<span class="broll-badge">B-Roll</span>' : `
                    <button class="btn-regenerate" data-scene-id="${scene.scene_id}" data-project-id="${projectId}"
                        ${scene.status === 'regenerating' ? 'disabled' : ''}>
                        재생성
                    </button>`}
                </div>
            `;

            grid.appendChild(card);
        });

        // 재생성 버튼 이벤트 리스너
        grid.querySelectorAll('.btn-regenerate').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const sceneId = parseInt(e.target.dataset.sceneId);
                const projectId = e.target.dataset.projectId;
                this.regenerateScene(projectId, sceneId);
            });
        });

        // 내레이션 편집 버튼 이벤트 리스너
        grid.querySelectorAll('.btn-edit-narration').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const narrationDiv = e.target.closest('.scene-card-narration');
                this.startNarrationEdit(narrationDiv);
            });
        });
    }

    startNarrationEdit(narrationDiv) {
        const sceneId = parseInt(narrationDiv.dataset.sceneId);
        const projectId = narrationDiv.dataset.projectId;
        const textSpan = narrationDiv.querySelector('.narration-text');
        const editBtn = narrationDiv.querySelector('.btn-edit-narration');
        const currentText = textSpan.textContent;

        // 이미 편집 중이면 무시
        if (narrationDiv.querySelector('.narration-edit-area')) return;

        // 텍스트 -> textarea로 교체
        textSpan.style.display = 'none';
        editBtn.style.display = 'none';

        const editHTML = `
            <textarea class="narration-edit-area">${currentText}</textarea>
            <div class="narration-edit-actions">
                <button class="btn-narration-save">저장</button>
                <button class="btn-narration-cancel">취소</button>
            </div>
        `;
        narrationDiv.insertAdjacentHTML('beforeend', editHTML);
        narrationDiv.querySelector('.narration-edit-area').focus();

        // 저장
        narrationDiv.querySelector('.btn-narration-save').onclick = async () => {
            const newText = narrationDiv.querySelector('.narration-edit-area').value.trim();
            if (!newText) return;
            await this.saveNarration(projectId, sceneId, newText, narrationDiv);
        };

        // 취소
        narrationDiv.querySelector('.btn-narration-cancel').onclick = () => {
            narrationDiv.querySelector('.narration-edit-area').remove();
            narrationDiv.querySelector('.narration-edit-actions').remove();
            textSpan.style.display = '';
            editBtn.style.display = '';
        };
    }

    async saveNarration(projectId, sceneId, newText, narrationDiv) {
        try {
            const baseUrl = this.getApiBaseUrl();
            const response = await fetch(
                `${baseUrl}/api/projects/${projectId}/scenes/${sceneId}/narration`,
                {
                    method: 'PUT',
                    headers: this.getAuthHeaders(),
                    body: JSON.stringify({ narration: newText })
                }
            );
            if (!response.ok) throw new Error('저장 실패');

            // UI 업데이트
            const textSpan = narrationDiv.querySelector('.narration-text');
            textSpan.textContent = newText;
            textSpan.style.display = '';
            narrationDiv.querySelector('.btn-edit-narration').style.display = '';
            narrationDiv.querySelector('.narration-edit-area').remove();
            narrationDiv.querySelector('.narration-edit-actions').remove();

            // 재합성 버튼 표시
            const recomposeBtn = document.getElementById('recompose-btn');
            if (recomposeBtn) recomposeBtn.style.display = 'inline-flex';

            this.showToast(`Scene ${sceneId} 내레이션 수정 완료! 영상 반영은 '영상 재합성'을 누르세요.`, 'success');
        } catch (err) {
            this.showToast(`내레이션 저장 실패: ${err.message}`, 'error');
        }
    }

    async regenerateScene(projectId, sceneId) {
        // 중복 클릭 방지
        const regenKey = `scene_${projectId}_${sceneId}`;
        if (this._regeneratingScenes.has(regenKey)) {
            this.showToast('이미 재생성 중입니다. 잠시 기다려주세요.', 'info');
            return;
        }

        const card = document.querySelector(`[data-scene-id="${sceneId}"]`);
        if (!card) {
            console.error(`[regenerateScene] Card not found for scene ${sceneId}`);
            this.showToast('씬 카드를 찾을 수 없습니다.', 'error');
            return;
        }

        const btn = card.querySelector('.btn-regenerate');
        const imageDiv = card.querySelector('.scene-card-image');

        this._regeneratingScenes.add(regenKey);

        try {
            // UI: 버튼 비활성화 + 로딩 오버레이 표시
            if (btn) {
                btn.disabled = true;
                btn.textContent = '재생성 중...';
            }
            card.classList.add('regenerating');

            if (imageDiv) {
                const overlay = document.createElement('div');
                overlay.className = 'regen-overlay';
                overlay.innerHTML = '<div class="regen-spinner"></div><span class="regen-text">이미지 재생성 중...</span>';
                imageDiv.appendChild(overlay);
            }

            this.addLog('INFO', `Scene ${sceneId} 재생성 시작...`);

            const baseUrl = this.getApiBaseUrl();
            const response = await fetch(`${baseUrl}/api/projects/${projectId}/scenes/${sceneId}/regenerate`, {
                method: 'POST',
                headers: this.getAuthHeaders(),
                body: JSON.stringify({
                    regenerate_image: true,
                    regenerate_tts: true,
                    regenerate_video: true
                })
            });

            if (!response.ok) {
                let errorMsg = '씬 재생성 실패';
                try {
                    const errorText = await response.text();
                    try {
                        const error = JSON.parse(errorText);
                        errorMsg = error.detail || error.message || errorMsg;
                    } catch (e) {
                        errorMsg = errorText || errorMsg;
                    }
                } catch (e) {
                    console.error("Error reading response error:", e);
                }
                throw new Error(errorMsg);
            }

            const result = await response.json();
            this.addLog('SUCCESS', `Scene ${sceneId} 재생성 완료!`);

            // 오버레이를 성공 표시로 변경
            const overlay = imageDiv ? imageDiv.querySelector('.regen-overlay') : null;
            if (overlay) {
                overlay.className = 'regen-overlay success';
                overlay.innerHTML = '<span class="regen-text">완료</span>';
                setTimeout(() => overlay.remove(), 1500);
            }

            // 이미지 즉시 갱신
            const img = imageDiv ? imageDiv.querySelector('img') : null;
            if (img && result.image_path) {
                const imageUrl = this.resolveImageUrl(result.image_path);
                img.src = `${imageUrl}?t=${Date.now()}`;
                img.style.display = 'block';
            }

            // 버튼 복구
            if (btn) {
                btn.disabled = false;
                btn.textContent = '재생성';
            }
            card.classList.remove('regenerating');

            // 재합성 버튼 표시
            const recomposeBtn = document.getElementById('recompose-btn');
            if (recomposeBtn) {
                recomposeBtn.style.display = 'block';
            }

            this.showToast(`Scene ${sceneId} 재생성 완료! 영상 반영은 "영상 재합성"을 누르세요.`, 'success');

        } catch (error) {
            console.error('씬 재생성 실패:', error);
            this.addLog('ERROR', `Scene ${sceneId} 재생성 실패: ${error.message}`);
            this.showToast(`Scene ${sceneId} 재생성 실패: ${error.message}`, 'error');

            // 오버레이 제거
            const overlay = imageDiv ? imageDiv.querySelector('.regen-overlay') : null;
            if (overlay) overlay.remove();

            // UI 복구
            if (btn) {
                btn.disabled = false;
                btn.textContent = '재생성';
            }
            card.classList.remove('regenerating');
        } finally {
            this._regeneratingScenes.delete(regenKey);
        }
    }

    async recomposeVideo(projectId) {
        const btn = document.getElementById('recompose-btn');

        try {
            btn.disabled = true;
            btn.innerHTML = '<span class="btn-icon">⏳</span> 재합성 중...';

            this.addLog('INFO', '영상 재합성 시작...');

            const baseUrl = this.getApiBaseUrl();
            const response = await fetch(`${baseUrl}/api/projects/${projectId}/recompose`, {
                method: 'POST',
                headers: this.getAuthHeaders()
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || '영상 재합성 실패');
            }

            const result = await response.json();
            this.addLog('SUCCESS', '✅ 영상 재합성 완료!');

            // 비디오 플레이어 새로고침
            const video = document.getElementById('result-video');
            const currentSrc = video.src;
            video.src = currentSrc + '?t=' + new Date().getTime(); // 캐시 우회
            video.load();

            // 재합성 버튼 숨기기
            btn.style.display = 'none';

            this.showToast('영상 재합성 완료! 새로운 영상이 플레이어에 반영되었습니다.', 'success');

        } catch (error) {
            console.error('영상 재합성 실패:', error);
            this.addLog('ERROR', `❌ 영상 재합성 실패: ${error.message}`);
            this.showToast('영상 재합성에 실패했습니다. 다시 시도해주세요.', 'error');

            btn.disabled = false;
            btn.innerHTML = '<span class="btn-icon">🔄</span> 영상 재합성 (수정된 씬 반영)';
        }
    }

    // ==================== History 기능 ====================
    async loadHistory() {
        try {
            const token = localStorage.getItem('token');
            if (!token) {
                document.getElementById('history-grid').innerHTML = '<p style="grid-column:1/-1;text-align:center;color:var(--text-muted);padding:40px;">로그인 후 이용할 수 있습니다.</p>';
                return;
            }

            let urlToUse = this.getApiBaseUrl();
            const response = await fetch(`${urlToUse}/api/history`, {
                headers: this.getAuthHeaders(),
            });

            if (!response.ok) {
                const errBody = await response.text().catch(() => '');
                console.error(`History API ${response.status}:`, errBody);
                throw new Error(`History ${response.status}`);
            }

            const data = await response.json();
            this._historyProjects = data.projects || [];

            // 필터 탭 이벤트 바인딩 (최초 1회)
            if (!this._historyFilterBound) {
                document.querySelectorAll('.history-filter-btn').forEach(btn => {
                    btn.addEventListener('click', () => {
                        document.querySelectorAll('.history-filter-btn').forEach(b => b.classList.remove('active'));
                        btn.classList.add('active');
                        this._renderHistoryGrid(btn.dataset.filter);
                    });
                });
                this._historyFilterBound = true;
            }

            // 현재 활성 필터 유지
            const activeFilter = document.querySelector('.history-filter-btn.active')?.dataset.filter || 'all';
            this._renderHistoryGrid(activeFilter);

        } catch (error) {
            console.error('History 로드 실패:', error);
            document.getElementById('history-grid').innerHTML = '<p style="color: #f66;">History 로드 실패</p>';
        }
    }

    _renderHistoryGrid(filter) {
        const historyGrid = document.getElementById('history-grid');
        historyGrid.innerHTML = '';

        const projects = (this._historyProjects || []).filter(p => {
            if (filter === 'all') return true;
            return (p.type || 'video') === filter;
        });

        if (projects.length === 0) {
            const label = filter === 'mv' ? '뮤직비디오가' : filter === 'video' ? '영상이' : '생성된 영상이';
            historyGrid.innerHTML = `<p style="grid-column: 1/-1; text-align: center; color: #888;">${label} 없습니다.</p>`;
            return;
        }

        projects.forEach(project => {
            const card = document.createElement('div');
            card.className = 'history-card';
            const isMV = project.type === 'mv';
            const typeBadge = isMV
                ? '<span class="history-type-badge mv">MV</span>'
                : '<span class="history-type-badge video">Video</span>';
            const fallbackIcon = isMV ? '🎵' : '📽️';

            // MV 추가 정보
            let mvInfo = '';
            if (isMV) {
                const parts = [];
                if (project.duration_sec) parts.push(`${Math.round(project.duration_sec)}s`);
                if (project.genre) parts.push(project.genre);
                if (project.style) parts.push(project.style);
                if (parts.length > 0) {
                    mvInfo = `<p class="history-mv-info">${parts.join(' · ')}</p>`;
                }
            }

            // 썸네일 결정: 이미지 > 비디오 프레임 > 폴백 아이콘
            let thumbContent;
            if (project.thumbnail_url) {
                thumbContent = `<img loading="lazy" data-src="${this.getMediaBaseUrl()}${escapeHtml(project.thumbnail_url)}" alt="${escapeHtml(project.title)}" onerror="this.style.display='none'; this.parentElement.querySelector('.thumb-fallback').style.display='flex'" style="opacity:0;transition:opacity 0.3s"><div class="thumb-fallback" style="display:none;width:100%;height:100%;align-items:center;justify-content:center;color:#555;">${fallbackIcon}</div>`;
            } else if (project.status === 'completed') {
                // R2에서 비디오 첫 프레임을 썸네일로 표시 (Range 요청으로 메타데이터만 로드)
                const videoAssetUrl = `${this.getMediaBaseUrl()}/api/asset/${project.project_id}/video/final_video.mp4`;
                thumbContent = `<video muted preload="metadata" data-video-thumb="${videoAssetUrl}#t=1" style="width:100%;height:100%;object-fit:cover;pointer-events:none;opacity:0;transition:opacity 0.3s" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex'"></video><div class="thumb-fallback" style="display:none;width:100%;height:100%;align-items:center;justify-content:center;color:#555;">${fallbackIcon}</div>`;
            } else {
                thumbContent = `<div class="thumb-fallback" style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;color:#555;">${fallbackIcon}</div>`;
            }

            card.innerHTML = `
                <div class="history-thumb" style="background: #1a1a2e;">
                    ${typeBadge}
                    ${thumbContent}
                </div>
                <div class="history-info">
                    <p class="history-title">${escapeHtml(project.title)}</p>
                    ${mvInfo}
                    <p class="history-date">${new Date(project.created_at).toLocaleDateString('ko-KR')}</p>
                    <span class="history-status ${project.status === 'completed' ? 'completed' : project.status === 'images_ready' ? 'images-ready' : project.status === 'anchors_ready' ? 'images-ready' : ''}">${project.status === 'completed' ? '완료' : project.status === 'anchors_ready' ? '앵커 리뷰' : project.status === 'images_ready' ? '이미지 완료' : project.status === 'failed' ? '실패' : '처리 중'}</span>
                </div>
            `;

            card.style.cursor = 'pointer';
            card.onclick = () => {
                // 클릭 피드백
                card.style.opacity = '0.6';
                card.style.pointerEvents = 'none';
                this.showArchiveDetail(project.project_id, project.type || 'video')
                    .finally(() => {
                        card.style.opacity = '1';
                        card.style.pointerEvents = 'auto';
                    });
            };

            historyGrid.appendChild(card);
        });

        // Lazy load: 이미지 + 비디오 썸네일 모두 IntersectionObserver로 지연 로드
        const lazyElements = historyGrid.querySelectorAll('img[data-src], video[data-video-thumb]');
        if (lazyElements.length > 0) {
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const el = entry.target;
                        if (el.tagName === 'IMG') {
                            el.src = el.dataset.src;
                            el.onload = () => { el.style.opacity = '1'; };
                        } else if (el.tagName === 'VIDEO') {
                            el.src = el.dataset.videoThumb;
                            el.onloadeddata = () => { el.style.opacity = '1'; };
                        }
                        observer.unobserve(el);
                    }
                });
            }, { rootMargin: '200px' });
            lazyElements.forEach(el => observer.observe(el));
        }
    }

    // ==================== 보관함 상세 보기 ====================
    async showArchiveDetail(projectId, type) {
        console.log(`[Archive] showArchiveDetail: projectId=${projectId}, type=${type}`);
        try {
            const baseUrl = this.getApiBaseUrl();
            const archFetchOpts = {};
            const archToken = localStorage.getItem('token');
            if (archToken) archFetchOpts.headers = { 'Authorization': `Bearer ${archToken}` };
            const response = await fetch(`${baseUrl}/api/manifest/${projectId}`, archFetchOpts);
            if (!response.ok) throw new Error(`Manifest 로드 실패 (${response.status})`);
            const manifest = await response.json();
            console.log(`[Archive] Manifest loaded: status=${manifest.status}, scenes=${(manifest.scenes || []).length}`);

            const isMV = type === 'mv';

            if (isMV) {
                // MV 프로젝트: 씬 이미지가 있으면 편집 가능한 결과 화면으로
                const scenes = manifest.scenes || [];
                const hasImages = scenes.some(s => s.image_path);

                // anchors_ready: 앵커 리뷰 화면으로 이동
                if (manifest.status === 'anchors_ready') {
                    const characters = manifest.visual_bible?.characters || manifest.characters || [];
                    this.showSection('mv-progress');
                    this.showMVCharacterReview(projectId, characters);
                    return;
                }

                if (manifest.status === 'completed' || (hasImages && (manifest.status === 'failed' || manifest.status === 'images_ready'))) {
                    const isCompleted = manifest.status === 'completed';
                    let headerText;
                    if (manifest.status === 'images_ready') {
                        headerText = 'Images Ready - Edit and recompose';
                    } else if (manifest.status === 'failed') {
                        headerText = '⚠️ 영상 합성 실패 - 음악 재업로드 후 재합성으로 복구';
                    }

                    this.showMVEditor(projectId, {
                        showVideo: isCompleted,
                        videoCompleted: isCompleted,
                        headerText: headerText,
                        scenes: scenes,
                        duration_sec: manifest.music_analysis?.duration_sec || 0,
                    });
                    this.setNavActive('nav-history');

                    // 실패/미합성: 재합성+음악 업로드 버튼 표시
                    if (!isCompleted) {
                        const recomposeBtn = document.getElementById('mv-editor-recompose-btn');
                        if (recomposeBtn) recomposeBtn.style.display = 'inline-flex';
                        const musicBtn = document.getElementById('mv-editor-music-upload-btn');
                        if (musicBtn) musicBtn.style.display = 'inline-flex';
                    }
                } else if (manifest.status === 'processing' || manifest.status === 'composing' || manifest.status === 'generating') {
                    // 진행 중인 MV → progress 화면 + 폴링 재개
                    this.showSection('mv-progress');
                    this.mvProjectId = projectId;
                    const progressVal = manifest.progress || (manifest.status === 'composing' ? 75 : 30);
                    this.updateMVProgress(progressVal, manifest.current_step || 'MV 생성 중...');
                    this.startMVPolling(projectId);
                } else {
                    this.showSection('result');
                    this.setNavActive('nav-history');
                    document.getElementById('result-header-text').textContent = "MV 생성 실패";
                    document.getElementById('result-video-container').innerHTML = `<div style="text-align:center;padding:40px;background:rgba(255,50,50,0.1);border-radius:8px;"><span style="font-size:48px;display:block;margin-bottom:20px;">⚠️</span><h3>생성 도중 오류가 발생했습니다.</h3><p>${escapeHtml(manifest.error_message || '알 수 없는 오류')}</p></div>`;
                    document.getElementById('download-btn').style.display = 'none';
                }

            } else {
                // 일반 영상: showResults 재사용 + _fromArchive 플래그
                await this.showResults({
                    project_id: projectId,
                    title: manifest.title,
                    status: manifest.status,
                    error_message: manifest.error_message,
                    title_candidates: manifest.outputs?.title_candidates,
                    thumbnail_texts: manifest.outputs?.thumbnail_texts,
                    hashtags: manifest.outputs?.hashtags,
                    video_path: manifest.outputs?.final_video_path,
                    server_url: baseUrl,
                    platform: manifest.input?.target_platform || 'youtube_long',
                    hook_text: manifest.hook_text || '',
                    _fromArchive: true,
                    _scenes: manifest.scenes || [],
                });
            }

        } catch (error) {
            console.error('보관함 상세 로드 실패:', error);
            this.showResultError(projectId, `보관함 데이터를 불러오지 못했습니다.\n${error.message}`);
        }
    }

    // ==================== 보관함 이미지 패널 ====================
    renderArchiveImagePanel(scenes, projectId, isMV) {
        console.log(`[Archive] renderArchiveImagePanel called: ${scenes.length} scenes, projectId=${projectId}, isMV=${isMV}`);
        try {
            const sceneManagement = document.getElementById('scene-management');
            if (!sceneManagement) {
                console.error('[Archive] scene-management element not found!');
                return;
            }
            // 확실히 보이게 설정
            sceneManagement.style.display = 'block';
            sceneManagement.style.visibility = 'visible';
            sceneManagement.style.opacity = '1';

            const header = sceneManagement.querySelector('h3');
            const desc = sceneManagement.querySelector('.section-description');
            const recomposeActions = sceneManagement.querySelector('.recompose-actions');
            if (header) header.textContent = isMV ? '🎵 MV 씬 이미지' : '🖼️ 씬 이미지';
            if (desc) desc.style.display = 'none';
            if (recomposeActions) recomposeActions.style.display = 'none';

            const grid = document.getElementById('result-scene-grid');
            if (!grid) {
                console.error('[Archive] result-scene-grid element not found!');
                return;
            }
            grid.innerHTML = '';

            if (!scenes || scenes.length === 0) {
                grid.innerHTML = '<p style="grid-column:1/-1;text-align:center;color:#888;padding:20px;">씬 이미지가 없습니다.</p>';
                sceneManagement.scrollIntoView({ behavior: 'smooth', block: 'start' });
                return;
            }

            scenes.forEach((scene, idx) => {
                const card = document.createElement('div');
                card.className = 'result-scene-card archive-image-card';

                let imagePath;
                if (isMV) {
                    imagePath = scene.image_path || `outputs/${projectId}/media/images/scene_${String(idx + 1).padStart(2, '0')}.png`;
                } else {
                    imagePath = scene.assets?.image_path || `outputs/${projectId}/media/images/scene_${String(scene.scene_id || idx + 1).padStart(2, '0')}.png`;
                }
                const imageUrl = this.resolveImageUrl(imagePath);

                // 텍스트: MV는 가사+타임스탬프, 일반은 내레이션
                let textContent = '';
                if (isMV) {
                    const startSec = scene.start_sec ?? scene.start_time;
                    const endSec = scene.end_sec ?? scene.end_time;
                    const timeBadge = startSec != null
                        ? `<span class="scene-time-badge">${this._formatTime(startSec)} - ${this._formatTime(endSec)}</span>`
                        : '';
                    const lyrics = scene.lyrics_text || scene.concept || '';
                    textContent = `${timeBadge}<div style="margin-top:4px;color:#ccc;font-size:0.85rem;">${lyrics}</div>`;
                } else {
                    textContent = `<div style="color:#ccc;font-size:0.85rem;">${scene.narration || ''}</div>`;
                }

                const sceneLabel = isMV ? `Scene ${idx + 1}` : `Scene ${scene.scene_id || idx + 1}`;

                // 이미지 또는 비디오 프레임 폴백
                let sceneMedia;
                if (imageUrl) {
                    sceneMedia = `
                        <img src="${imageUrl}?t=${Date.now()}" alt="${sceneLabel}"
                            onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';"
                            style="width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:6px;display:block;">
                        <div class="image-placeholder" style="display:none;">이미지 없음</div>`;
                } else {
                    sceneMedia = `<div class="image-placeholder" style="display:flex;">이미지 없음</div>`;
                }

                card.innerHTML = `
                    <div class="scene-card-header">
                        <span class="scene-card-title">${sceneLabel}</span>
                    </div>
                    <div class="scene-card-image">
                        ${sceneMedia}
                    </div>
                    <div class="scene-card-narration">${textContent}</div>
                `;

                grid.appendChild(card);
            });

            console.log(`[Archive] Rendered ${scenes.length} scene cards in grid`);
            // 이미지 패널이 보이도록 스크롤
            setTimeout(() => {
                sceneManagement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }, 100);

        } catch (err) {
            console.error('[Archive] renderArchiveImagePanel error:', err);
            const grid = document.getElementById('result-scene-grid');
            if (grid) {
                grid.innerHTML = `<p style="grid-column:1/-1;text-align:center;color:#f66;padding:20px;">이미지 패널 렌더링 오류: ${escapeHtml(err.message)}</p>`;
            }
        }
    }

    _formatTime(seconds) {
        if (seconds == null) return '';
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${String(s).padStart(2, '0')}`;
    }

    _extractMusicTitle(musicPath) {
        if (!musicPath) return null;
        const filename = musicPath.split('/').pop().split('\\').pop();
        return filename.replace(/\.[^.]+$/, '');
    }

    // ==================== 이미지 URL 경로 변환 ====================
    resolveImageUrl(imagePath) {
        if (!imagePath) return '';
        // 로컬 환경에서 원격 Railway URL → 로컬 asset 경로로 변환
        if (imagePath.startsWith('http') && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')) {
            try {
                const url = new URL(imagePath);
                // /api/asset/{pid}/images/{filename} 또는 /api/asset/{pid}/image/{filename} 패턴 매칭
                const assetMatch = url.pathname.match(/\/api\/asset\/([^/]+)\/images?\/(.+)/);
                if (assetMatch) {
                    return `/api/asset/${assetMatch[1]}/images/${assetMatch[2]}`;
                }
                // /media/{pid}/media/images/{filename} 패턴
                const mediaMatch = url.pathname.match(/\/media\/(.+)/);
                if (mediaMatch) {
                    return `/media/${mediaMatch[1]}`;
                }
            } catch (e) { }
        }
        if (imagePath.startsWith('http')) return imagePath;
        // outputs/xxx → /media/xxx 변환 (FastAPI StaticFiles 마운트: /media = outputs/)
        if (imagePath.startsWith('outputs/')) {
            return `${this.getMediaBaseUrl()}/media/${imagePath.slice('outputs/'.length)}`;
        }
        if (imagePath.startsWith('/')) {
            return `${this.getMediaBaseUrl()}${imagePath}`;
        }
        return `${this.getMediaBaseUrl()}/media/${imagePath}`;
    }

    // ==================== 캐릭터 캐스팅 워크플로우 ====================

    async startCharacterCasting() {
        if (!this.currentStoryData) {
            this.showToast('스토리 데이터가 없습니다. 먼저 스토리를 생성해주세요.', 'warning');
            return;
        }

        // 리뷰 섹션 옵션 동기화
        this._syncGenerationOptions();

        // 스토리 데이터 업데이트 (사용자 편집 반영)
        const title = document.getElementById('review-title').value;
        this.currentStoryData.title = title;
        document.querySelectorAll('.review-card').forEach((card) => {
            const sceneId = parseInt(card.dataset.sceneId);
            const scene = this.currentStoryData.scenes.find(s => s.scene_id === sceneId);
            if (scene) {
                scene.narration = card.querySelector('.narration-input').value;
                scene.sentence = card.querySelector('.narration-input').value;
                scene.visual_description = card.querySelector('.visual-input').value;
                scene.prompt = card.querySelector('.visual-input').value;
            }
        });

        // 캐릭터가 있는지 확인 (character_sheet는 {token: {...}} 형태의 객체)
        const characterSheet = this.currentStoryData.character_sheet || {};
        const characterTokens = Object.keys(characterSheet);
        if (characterTokens.length === 0) {
            console.log('[Casting] No characters found, skipping to image generation');
            this.showToast('캐릭터 없는 스토리 — 바로 이미지 생성으로 진행합니다.', 'info');
            this.startImageGeneration();
            return;
        }

        const apiUrl = this.getApiBaseUrl();
        const btn = document.getElementById('generate-images-btn');
        const originalBtnText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = '<span class="btn-icon">⏳</span> 캐릭터 캐스팅 시작...';

        try {

            const response = await fetch(`${apiUrl}/api/generate/characters`, {
                method: 'POST',
                headers: this.getAuthHeaders(),
                body: JSON.stringify({
                    project_id: this.projectId,
                    story_data: this.currentStoryData,
                    request_params: this.currentRequestParams
                })
            });

            if (!response.ok) {
                let errorDetail = response.statusText;
                try {
                    const errorBody = await response.json();
                    errorDetail = errorBody.detail || errorBody.error || errorBody.message || JSON.stringify(errorBody);
                } catch (e) { }
                throw new Error(`${response.status}: ${errorDetail}`);
            }

            const result = await response.json();
            this.projectId = result.project_id;

            // 캐스팅 화면으로 전환
            this.renderCastingPlaceholders(characterSheet);
            this.showSection('character-casting');

            // 로딩바 초기화 및 표시
            document.getElementById('casting-progress-container').classList.remove('hidden');
            document.getElementById('casting-progress-fill').style.width = '0%';
            document.getElementById('casting-progress-percent').textContent = '0%';
            document.getElementById('casting-progress-label').textContent = '캐릭터 캐스팅 중...';

            // 폴링 시작
            this.pollCastingStatus(this.projectId);

        } catch (error) {
            console.error('[Casting] Error:', error);
            this.showToast('캐릭터 캐스팅에 실패했습니다. 다시 시도해주세요.', 'error');
        } finally {
            btn.disabled = false;
            btn.innerHTML = originalBtnText;
        }
    }

    renderCastingPlaceholders(characterSheet) {
        const grid = document.getElementById('casting-grid');
        grid.innerHTML = '';

        Object.entries(characterSheet).forEach(([token, charData]) => {
            const card = document.createElement('div');
            card.className = 'image-card';
            card.dataset.characterToken = token;

            card.innerHTML = `
                <div class="image-card-header">
                    <span class="image-card-title">${charData.name || token}</span>
                </div>
                <div class="image-card-visual">
                    <div style="width:100%;aspect-ratio:3/2;background:rgba(255,255,255,0.05);display:flex;align-items:center;justify-content:center;border-radius:8px;">
                        <div class="spinner" style="width:40px;height:40px;border:3px solid rgba(255,255,255,0.1);border-top-color:#646cff;border-radius:50%;animation:spin 1s linear infinite;"></div>
                    </div>
                </div>
                <div class="image-card-body">
                    <div class="image-narration">${charData.description || charData.visual_description || ''}</div>
                </div>
            `;

            grid.appendChild(card);
        });

        // 액션 버튼 숨기기 (캐스팅 완료 후 표시)
        document.getElementById('casting-actions').style.display = 'none';
    }

    pollCastingStatus(projectId) {
        if (this.castingPollingInterval) {
            clearInterval(this.castingPollingInterval);
        }

        const apiUrl = this.getApiBaseUrl();
        let pollCount = 0;
        const MAX_POLL_COUNT = 150; // 5분 타임아웃 (2초 × 150)

        this.castingPollingInterval = setInterval(async () => {
            try {
                pollCount++;

                // 타임아웃 체크
                if (pollCount > MAX_POLL_COUNT) {
                    clearInterval(this.castingPollingInterval);
                    this.castingPollingInterval = null;
                    console.error('[Casting] Polling timeout after 5 minutes');
                    this.showToast('캐릭터 캐스팅 시간이 초과되었습니다. 다시 시도해주세요.', 'error');
                    document.getElementById('casting-progress-label').textContent = '시간 초과';
                    return;
                }

                const response = await fetch(`${apiUrl}/api/status/characters/${projectId}`, {
                    headers: this.getAuthHeaders()
                });

                if (!response.ok) {
                    console.warn(`[Casting Poll #${pollCount}] Status check failed:`, response.status);
                    return;
                }

                const data = await response.json();
                console.log(`[Casting Poll #${pollCount}]`, data.casting_status);

                // not_found 상태 처리 — 아직 manifest 미생성
                if (data.casting_status === 'not_found') {
                    if (pollCount > 30) { // 60초 이상 not_found면 에러
                        clearInterval(this.castingPollingInterval);
                        this.castingPollingInterval = null;
                        this.showToast('캐릭터 캐스팅 데이터를 찾을 수 없습니다.', 'error');
                        document.getElementById('casting-progress-label').textContent = '데이터 없음';
                    }
                    return;
                }

                // 프로그레스 업데이트
                this.updateCastingProgress(data);

                if (data.casting_status === 'casting_ready') {
                    clearInterval(this.castingPollingInterval);
                    this.castingPollingInterval = null;

                    // 프로그레스 100%
                    document.getElementById('casting-progress-fill').style.width = '100%';
                    document.getElementById('casting-progress-percent').textContent = '100%';
                    document.getElementById('casting-progress-label').textContent = '캐릭터 캐스팅 완료!';

                    // 프로그레스 바 숨기기
                    setTimeout(() => {
                        document.getElementById('casting-progress-container').classList.add('hidden');
                    }, 1000);

                    // 캐릭터 카드 업데이트
                    this.renderCastingResults(data.characters);

                    // 액션 버튼 표시
                    document.getElementById('casting-actions').style.display = '';
                } else if (data.casting_status === 'failed') {
                    clearInterval(this.castingPollingInterval);
                    this.castingPollingInterval = null;
                    const errorMsg = data.error ? `캐릭터 캐스팅 실패: ${data.error}` : '캐릭터 캐스팅에 실패했습니다.';
                    this.showToast(errorMsg, 'error');
                    document.getElementById('casting-progress-label').textContent = '캐스팅 실패';
                }

            } catch (error) {
                console.error('[Casting Poll] Error:', error);
            }
        }, 2000);
    }

    updateCastingProgress(data) {
        const characters = data.characters || [];
        const total = characters.length || 1;
        const done = characters.filter(c => c.image_path).length;
        const percent = Math.round((done / total) * 100);

        document.getElementById('casting-progress-fill').style.width = `${percent}%`;
        document.getElementById('casting-progress-percent').textContent = `${percent}%`;
        // 서버 메시지 우선 표시, 없으면 기본 메시지
        const label = data.message || `캐릭터 캐스팅 중... (${done}/${total})`;
        document.getElementById('casting-progress-label').textContent = label;
    }

    renderCastingResults(characters) {
        const grid = document.getElementById('casting-grid');
        grid.innerHTML = '';

        characters.forEach(char => {
            const token = char.token;
            const card = document.createElement('div');
            card.className = 'casting-character-card';
            card.dataset.characterToken = token;

            // 포즈 이미지들 (멀티포즈)
            const poseImages = char.pose_images || [];
            const mainImage = char.image_path ? this.resolveImageUrl(char.image_path) : '';

            const poseLabels = {
                'front': '정면',
                'three_quarter': '45도',
                'full_body': '전신',
                'side': '측면',
            };

            const cacheBuster = `v=${this.projectId}_${Date.now()}`;
            let posesHtml = '';
            // 상위 3개 포즈만 크게 표시 (front > three_quarter > full_body 우선)
            const POSE_PRIORITY = ['front', 'three_quarter', 'full_body', 'side', 'emotion_neutral', 'emotion_intense'];
            const sortedPoses = [...poseImages].sort((a, b) =>
                (POSE_PRIORITY.indexOf(a.pose) + 1 || 99) - (POSE_PRIORITY.indexOf(b.pose) + 1 || 99)
            ).slice(0, 3);

            if (sortedPoses.length > 0) {
                posesHtml = sortedPoses.map((p, idx) => {
                    const url = this.resolveImageUrl(p.image_path);
                    const label = poseLabels[p.pose] || p.pose;
                    const isSelected = p.image_path === char.image_path ? ' selected' : '';
                    const escapedUrl = url.replace(/'/g, "\\'");
                    const escapedLabel = label.replace(/'/g, "\\'");
                    const origIdx = poseImages.indexOf(p);
                    return `
                        <div class="casting-pose${isSelected}" data-token="${token}" data-pose-idx="${origIdx}" data-image-path="${p.image_path}" onclick="app.selectPose('${token}', ${origIdx})">
                            <img src="${url}?${cacheBuster}" alt="${label}" onerror="this.style.display='none'">
                            <button class="zoom-btn" onclick="event.stopPropagation(); app.openLightbox('${escapedUrl}?${cacheBuster}', '${escapedLabel}')" title="확대 보기">⤢</button>
                            <span class="casting-pose-label">${label}</span>
                        </div>
                    `;
                }).join('');
            } else if (mainImage) {
                posesHtml = `
                    <div class="casting-pose selected">
                        <img src="${mainImage}?${cacheBuster}" alt="${char.name || token}" onerror="this.style.display='none'">
                        <span class="casting-pose-label">정면</span>
                    </div>
                `;
            }

            card.innerHTML = `
                <div class="casting-char-header">
                    <span class="casting-char-name">${char.name || token}</span>
                    <span class="casting-char-meta">${char.gender || ''} ${char.age || ''}</span>
                </div>
                <div class="casting-poses-grid">
                    ${posesHtml || '<div style="padding:20px;color:var(--text-secondary);">이미지 없음</div>'}
                </div>
                <div class="casting-char-desc">${char.appearance || ''}</div>
                <div class="casting-char-actions">
                    <button class="btn-image-action btn-regenerate" onclick="app.regenerateCharacter('${token}')">
                        🔄 전체 재생성
                    </button>
                </div>
            `;

            grid.appendChild(card);
        });
    }

    selectPose(token, poseIdx) {
        // 해당 캐릭터의 모든 포즈에서 selected 제거
        document.querySelectorAll(`.casting-pose[data-token="${token}"]`).forEach(el => {
            el.classList.remove('selected');
        });
        // 선택한 포즈에 selected 추가
        const selected = document.querySelector(`.casting-pose[data-token="${token}"][data-pose-idx="${poseIdx}"]`);
        if (selected) {
            selected.classList.add('selected');
            // 선택된 포즈를 master로 기억
            if (!this._selectedPoses) this._selectedPoses = {};
            this._selectedPoses[token] = selected.dataset.imagePath;
        }
    }

    openLightbox(url, label) {
        document.querySelector('.casting-lightbox')?.remove();

        const lb = document.createElement('div');
        lb.className = 'casting-lightbox';
        lb.innerHTML = `
            <button class="casting-lightbox-close" title="닫기">✕</button>
            <img src="${url}" alt="${label}">
            <div class="casting-lightbox-label">${label}</div>
        `;
        lb.querySelector('.casting-lightbox-close').addEventListener('click', () => lb.remove());
        lb.addEventListener('click', (e) => { if (e.target === lb) lb.remove(); });

        const closeOnEsc = (e) => {
            if (e.key === 'Escape') { lb.remove(); document.removeEventListener('keydown', closeOnEsc); }
        };
        document.addEventListener('keydown', closeOnEsc);
        document.body.appendChild(lb);
    }

    async regenerateCharacter(token) {
        if (!this.projectId) {
            this.showToast('프로젝트 ID가 없습니다.', 'warning');
            return;
        }

        const apiUrl = this.getApiBaseUrl();
        const card = document.querySelector(`[data-character-token="${token}"]`);
        if (!card) return;

        const btn = card.querySelector('.btn-regenerate');
        const originalText = btn.textContent;
        btn.textContent = '⏳ 생성 중...';
        btn.disabled = true;

        try {
            console.log(`[Casting] Regenerating character: ${token}`);

            const response = await fetch(`${apiUrl}/api/regenerate/character/${this.projectId}/${token}`, {
                method: 'POST',
                headers: this.getAuthHeaders(),
                body: JSON.stringify({})
            });

            if (!response.ok) {
                throw new Error(`Regeneration failed: ${response.statusText}`);
            }

            const result = await response.json();

            // 포즈 그리드 전체 재렌더링
            if (result.pose_images && result.pose_images.length > 0) {
                const poseLabels = { front: '정면', three_quarter: '45도', full_body: '전신', side: '측면' };
                const reCacheBuster = `v=${this.projectId}_${Date.now()}`;
                const POSE_PRIORITY = ['front', 'three_quarter', 'full_body', 'side', 'emotion_neutral', 'emotion_intense'];
                const sortedPoses = [...result.pose_images].sort((a, b) =>
                    (POSE_PRIORITY.indexOf(a.pose) + 1 || 99) - (POSE_PRIORITY.indexOf(b.pose) + 1 || 99)
                ).slice(0, 3);
                const posesHtml = sortedPoses.map((p, idx) => {
                    const url = this.resolveImageUrl(p.web_path || p.image_path);
                    const label = poseLabels[p.pose] || p.pose;
                    const isSelected = p.pose === result.best_pose ? ' selected' : '';
                    const escapedUrl = url.replace(/'/g, "\\'");
                    const escapedLabel = label.replace(/'/g, "\\'");
                    const origIdx = result.pose_images.indexOf(p);
                    return `
                        <div class="casting-pose${isSelected}" data-token="${token}" data-pose-idx="${origIdx}" data-image-path="${p.image_path || p.web_path}" onclick="app.selectPose('${token}', ${origIdx})">
                            <img src="${url}?${reCacheBuster}" alt="${label}" onerror="this.style.display='none'">
                            <button class="zoom-btn" onclick="event.stopPropagation(); app.openLightbox('${escapedUrl}?${reCacheBuster}', '${escapedLabel}')" title="확대 보기">⤢</button>
                            <span class="casting-pose-label">${label}</span>
                        </div>`;
                }).join('');
                const grid = card.querySelector('.casting-poses-grid');
                if (grid) grid.innerHTML = posesHtml;
            } else if (result.image_path) {
                // 구버전 단일 이미지 응답 fallback
                const img = card.querySelector('img');
                if (img) img.src = `${this.resolveImageUrl(result.image_path)}?t=${Date.now()}`;
            }

            this.showToast(`${token} 캐릭터 재생성 완료!`, 'success');

        } catch (error) {
            console.error('[Casting] Regenerate error:', error);
            this.showToast(`캐릭터 재생성 실패: ${error.message}`, 'error');
        } finally {
            btn.textContent = originalText;
            btn.disabled = false;
        }
    }

    async startImageGenerationAfterCasting() {
        // 캐스팅 승인 후 이미지 생성으로 진행
        this.startImageGeneration();
    }

    // ==================== 이미지 생성 워크플로우 ====================

    async startImageGeneration() {
        if (!this.currentStoryData) {
            this.showToast('스토리 데이터가 없습니다. 먼저 스토리를 생성해주세요.', 'warning');
            return;
        }

        // 리뷰 섹션 옵션 동기화
        this._syncGenerationOptions();

        const apiUrl = this.getApiBaseUrl();
        const btn = document.getElementById('generate-images-btn');
        const originalBtnText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = '<span class="btn-icon">⏳</span> 이미지 생성 시작...';

        try {
            const title = document.getElementById('review-title').value;
            this.currentStoryData.title = title;

            document.querySelectorAll('.review-card').forEach((card, index) => {
                const sceneId = parseInt(card.dataset.sceneId);
                const scene = this.currentStoryData.scenes.find(s => s.scene_id === sceneId);
                if (scene) {
                    scene.narration = card.querySelector('.narration-input').value;
                    scene.sentence = card.querySelector('.narration-input').value;
                    scene.visual_description = card.querySelector('.visual-input').value;
                    scene.prompt = card.querySelector('.visual-input').value;
                }
            });

            console.log('[Image Generation] Starting (async)...');
            console.log('[Image Generation] Scenes count:', this.currentStoryData.scenes?.length);

            const response = await fetch(`${apiUrl}/api/generate/images`, {
                method: 'POST',
                headers: this.getAuthHeaders(),
                body: JSON.stringify({
                    project_id: this.projectId,
                    story_data: this.currentStoryData,
                    request_params: this.currentRequestParams
                })
            });

            if (!response.ok) {
                let errorDetail = response.statusText;
                try {
                    const errorBody = await response.json();
                    errorDetail = errorBody.detail || errorBody.error || errorBody.message || JSON.stringify(errorBody);
                } catch (e) {
                    try { errorDetail = await response.text(); } catch (e2) { }
                }
                throw new Error(`${response.status}: ${errorDetail}`);
            }

            const result = await response.json();
            this.projectId = result.project_id;

            // 즉시 프리뷰 화면으로 전환 (플레이스홀더 표시)
            this.renderImagePreviewPlaceholders(this.currentStoryData.scenes, result.total_scenes);
            this.showSection('image-preview');

            // 진행 바 표시
            const progressContainer = document.getElementById('image-progress-container');
            if (progressContainer) progressContainer.classList.remove('hidden');

            // approve 버튼 비활성화 (생성 완료까지)
            const approveBtn = document.getElementById('approve-images-btn');
            if (approveBtn) approveBtn.disabled = true;

            // 폴링 시작
            this.pollImageStatus(this.projectId);

        } catch (error) {
            console.error('[Image Generation] Error:', error);
            this.showToast('이미지 생성에 실패했습니다. 다시 시도해주세요.', 'error');
        } finally {
            btn.disabled = false;
            btn.innerHTML = originalBtnText;
        }
    }

    renderImagePreviewPlaceholders(scenes, totalScenes) {
        const grid = document.getElementById('image-preview-grid');
        grid.innerHTML = '';

        scenes.forEach((scene, index) => {
            const card = document.createElement('div');
            card.className = 'image-card';
            card.dataset.sceneId = scene.scene_id;

            card.innerHTML = `
                <div class="image-card-header">
                    <span class="image-card-title">Scene ${scene.scene_id}</span>
                </div>
                <div class="image-placeholder" style="width:100%;aspect-ratio:16/9;background:rgba(255,255,255,0.05);display:flex;align-items:center;justify-content:center;border-radius:8px;font-size:24px;">
                    <div class="spinner" style="width:40px;height:40px;border:3px solid rgba(255,255,255,0.1);border-top-color:#646cff;border-radius:50%;animation:spin 1s linear infinite;"></div>
                </div>
                <div class="image-card-body">
                    <div class="image-narration">${scene.narration || scene.sentence || ''}</div>
                    <div class="image-actions">
                        <button class="btn-image-action btn-regenerate" disabled>🔄 재생성</button>
                        <button class="btn-image-action btn-i2v" disabled>🎬 I2V</button>
                    </div>
                </div>
            `;

            grid.appendChild(card);
        });

        // 진행 바 초기화
        this.updateImageProgress(0, totalScenes, '이미지 생성 준비 중...');
    }

    updateImageProgress(completed, total, message) {
        const pct = total > 0 ? Math.round((completed / total) * 100) : 0;
        const bar = document.getElementById('image-progress-bar');
        const pctEl = document.getElementById('image-progress-percentage');
        const msgEl = document.getElementById('image-status-message');
        const labelEl = document.getElementById('image-progress-label');

        if (bar) bar.style.width = `${pct}%`;
        if (pctEl) pctEl.textContent = `${pct}%`;
        if (msgEl) msgEl.textContent = message || '';
        if (labelEl) labelEl.textContent = `이미지 생성 중 (${completed}/${total})`;
    }

    async pollImageStatus(projectId) {
        if (this.imagePollingInterval) {
            clearInterval(this.imagePollingInterval);
        }

        const apiUrl = this.getApiBaseUrl();
        const pollStartTime = Date.now();
        const POLL_TIMEOUT_MS = 5 * 60 * 1000; // 5분 타임아웃

        this.imagePollingInterval = setInterval(async () => {
            try {
                // 타임아웃 체크 — not_found 무한루프 방지
                if (Date.now() - pollStartTime > POLL_TIMEOUT_MS) {
                    clearInterval(this.imagePollingInterval);
                    this.imagePollingInterval = null;
                    this.updateImageProgress(0, 0, '서버 응답 시간 초과 — 페이지를 새로고침해주세요.');
                    this.addLog('ERROR', '이미지 생성 상태 확인 시간 초과 (5분). 서버 로그를 확인하세요.');
                    const approveBtn = document.getElementById('approve-images-btn');
                    if (approveBtn) approveBtn.disabled = false;
                    return;
                }

                const response = await fetch(`${apiUrl}/api/status/images/${projectId}`);
                if (!response.ok) {
                    console.warn(`[Image Polling] HTTP ${response.status}`);
                    return;
                }

                const data = await response.json();
                const { completed, total, scenes, status, error_message } = data;
                console.log(`[Image Polling] status=${status}, completed=${completed}/${total}, scenes=${scenes?.length || 0}`);

                // 상태별 메시지 표시 (서버 message 우선 사용)
                let statusMsg = data.message || `Scene ${completed}/${total} 완료`;
                if (status === 'not_found') {
                    statusMsg = '프로젝트 초기화 중...';
                } else if (status === 'preparing' && !data.message) {
                    statusMsg = '스타일/캐릭터 앵커 준비 중...';
                } else if (status === 'generating_images' && completed === 0 && !data.message) {
                    statusMsg = '이미지 생성 시작 중...';
                } else if (status === 'generating_images' && completed > 0) {
                    statusMsg = `Scene ${completed}/${total} 완료`;
                }

                // 진행 바 업데이트
                this.updateImageProgress(completed, total, statusMsg);

                // 완료된 씬 카드 업데이트
                scenes.forEach(scene => {
                    if (scene.status === 'completed' && scene.image_path) {
                        const card = document.querySelector(`.image-card[data-scene-id="${scene.scene_id}"]`);
                        if (!card) return;

                        // 플레이스홀더 → 이미지 교체 (아직 이미지 없는 경우만)
                        if (!card.querySelector('img')) {
                            const placeholder = card.querySelector('.image-placeholder');
                            if (placeholder) {
                                const imageUrl = this.resolveImageUrl(scene.image_path);
                                console.log(`[Image Preview] Scene ${scene.scene_id} URL: ${imageUrl}`);
                                const img = document.createElement('img');
                                img.alt = `Scene ${scene.scene_id}`;
                                img.onload = () => {
                                    console.log(`[Image Preview] Scene ${scene.scene_id} loaded OK`);
                                };
                                img.onerror = () => {
                                    console.error(`[Image Preview] Failed to load: ${img.src}`);
                                    img.alt = 'Image load failed';
                                };
                                img.src = `${imageUrl}?t=${Date.now()}`;
                                placeholder.replaceWith(img);
                            }
                        }

                        // 버튼 활성화 (이미지 교체와 무관하게 항상 실행)
                        const regenBtn = card.querySelector('.btn-regenerate');
                        const i2vBtn = card.querySelector('.btn-i2v');
                        if (regenBtn && regenBtn.disabled) {
                            regenBtn.disabled = false;
                            regenBtn.setAttribute('onclick', `app.regenerateImage('${projectId}', ${scene.scene_id})`);
                        }
                        if (i2vBtn && i2vBtn.disabled) {
                            i2vBtn.disabled = false;
                            i2vBtn.setAttribute('onclick', `app.convertToVideo('${projectId}', ${scene.scene_id})`);
                        }
                    }
                });

                // 전체 완료 체크 (preparing/generating 단계에서는 완료 아님)
                if (status === 'images_ready' || (status !== 'preparing' && status !== 'generating_images' && status !== 'not_found' && completed === total && total > 0)) {
                    clearInterval(this.imagePollingInterval);
                    this.imagePollingInterval = null;

                    // 이미지 경로를 currentStoryData에 저장 (영상 생성 시 재사용)
                    if (this.currentStoryData && scenes) {
                        scenes.forEach(scene => {
                            if (scene.status === 'completed' && scene.image_path) {
                                const idx = this.currentStoryData.scenes.findIndex(
                                    s => s.scene_id === scene.scene_id || s.scene_id === String(scene.scene_id)
                                );
                                if (idx !== -1) {
                                    if (!this.currentStoryData.scenes[idx].assets) {
                                        this.currentStoryData.scenes[idx].assets = {};
                                    }
                                    this.currentStoryData.scenes[idx].assets.image_path = scene.image_path;
                                    console.log(`[Image Sync] Scene ${scene.scene_id} image_path saved to storyData`);
                                }
                            }
                        });
                        this.currentStoryData._images_pregenerated = true;
                    }

                    this.updateImageProgress(total, total, '모든 이미지 생성 완료!');

                    // 진행 바 숨기기 (1초 후)
                    setTimeout(() => {
                        const progressContainer = document.getElementById('image-progress-container');
                        if (progressContainer) progressContainer.classList.add('hidden');
                    }, 1500);

                    // approve 버튼 활성화
                    const approveBtn = document.getElementById('approve-images-btn');
                    if (approveBtn) approveBtn.disabled = false;
                }

                // 실패 체크
                if (status === 'failed') {
                    clearInterval(this.imagePollingInterval);
                    this.imagePollingInterval = null;
                    this.updateImageProgress(completed, total, `오류: ${error_message || '이미지 생성 실패'}`);
                    const approveBtn = document.getElementById('approve-images-btn');
                    if (approveBtn) approveBtn.disabled = false;
                }

            } catch (error) {
                console.error('[Image Polling] Error:', error);
            }
        }, 2500);
    }

    renderImagePreview(data) {
        const grid = document.getElementById('image-preview-grid');
        grid.innerHTML = '';

        // 진행 바 숨기기 (완성 데이터를 직접 렌더링하는 경우)
        const progressContainer = document.getElementById('image-progress-container');
        if (progressContainer) progressContainer.classList.add('hidden');

        // approve 버튼 활성화
        const approveBtn = document.getElementById('approve-images-btn');
        if (approveBtn) approveBtn.disabled = false;

        const scenes = data.scenes || data.story_data?.scenes || [];
        const projectId = data.project_id || this.projectId;

        scenes.forEach(scene => {
            const card = document.createElement('div');
            card.className = 'image-card';
            card.dataset.sceneId = scene.scene_id;

            const imagePath = scene.assets?.image_path || scene.image_path || '';
            const imageUrl = this.resolveImageUrl(imagePath);

            card.innerHTML = `
                <div class="image-card-header">
                    <span class="image-card-title">Scene ${scene.scene_id}</span>
                    ${scene.i2v_converted ? '<span class="i2v-done-badge">I2V</span>' : ''}
                </div>
                <div class="image-card-visual">
                    <img src="${imageUrl}?t=${Date.now()}" alt="Scene ${scene.scene_id}">
                    <div class="i2v-overlay" style="display:none">
                        <div class="i2v-overlay-content">
                            <div class="i2v-spinner"></div>
                            <span class="i2v-overlay-text">I2V 변환 중...</span>
                            <span class="i2v-overlay-sub">약 1~2분 소요</span>
                        </div>
                    </div>
                </div>
                <div class="image-card-body">
                    <div class="image-narration">${scene.narration || scene.sentence || ''}</div>
                    <div class="image-actions">
                        <button class="btn-image-action btn-regenerate" onclick="app.regenerateImage('${projectId}', ${scene.scene_id})">🔄 재생성</button>
                        <button class="btn-image-action btn-i2v" onclick="app.convertToVideo('${projectId}', ${scene.scene_id})" ${scene.i2v_converted ? 'disabled' : ''}>${scene.i2v_converted ? '✅ I2V' : '🎬 I2V'}</button>
                    </div>
                </div>
            `;

            grid.appendChild(card);
        });

        this.projectId = projectId;
    }

    async regenerateImage(projectId, sceneId) {
        // 크레딧 사전 확인
        if (typeof checkCreditsBeforeAction === 'function') {
            const ok = await checkCreditsBeforeAction('image_regen');
            if (!ok) return;
        }

        // 중복 클릭 방지
        const regenKey = `image_${projectId}_${sceneId}`;
        if (this._regeneratingScenes.has(regenKey)) {
            this.showToast('이미 재생성 중입니다. 잠시 기다려주세요.', 'info');
            return;
        }

        const card = document.querySelector(`.image-card[data-scene-id="${sceneId}"]`);
        if (!card) {
            console.error(`[regenerateImage] Card not found for scene ${sceneId}`);
            this.showToast('씬 카드를 찾을 수 없습니다.', 'error');
            return;
        }

        const btn = card.querySelector('.btn-regenerate');
        const imgEl = card.querySelector('img');

        this._regeneratingScenes.add(regenKey);

        try {
            // UI: 버튼 비활성화 + 이미지 영역에 로딩 오버레이
            if (btn) {
                btn.textContent = '재생성 중...';
                btn.disabled = true;
            }

            // 이미지 위에 오버레이 추가
            if (imgEl && imgEl.parentElement) {
                const wrapper = imgEl.parentElement;
                wrapper.style.position = 'relative';
                const overlay = document.createElement('div');
                overlay.className = 'regen-overlay';
                overlay.innerHTML = '<div class="regen-spinner"></div><span class="regen-text">이미지 재생성 중...</span>';
                wrapper.appendChild(overlay);
            }

            const response = await fetch(`${this.getApiBaseUrl()}/api/regenerate/image/${projectId}/${sceneId}`, {
                method: 'POST',
                headers: this.getAuthHeaders(),
                body: JSON.stringify({})
            });

            if (!response.ok) {
                if (typeof handleApiError === 'function' && await handleApiError(response.clone(), 'image_regen')) {
                    // Remove overlay on billing error
                    const overlay = imgEl?.parentElement?.querySelector('.regen-overlay');
                    if (overlay) overlay.remove();
                    return;
                }
                let errorDetail = response.statusText;
                try {
                    const errorBody = await response.json();
                    errorDetail = errorBody.detail || errorBody.error || errorBody.message || JSON.stringify(errorBody);
                } catch (e) {
                    try { errorDetail = await response.text(); } catch (e2) { }
                }
                throw new Error(`${response.status}: ${errorDetail}`);
            }

            const result = await response.json();

            // 이미지 갱신
            if (imgEl && result.image_path) {
                const imageUrl = this.resolveImageUrl(result.image_path);
                imgEl.src = `${imageUrl}?t=${Date.now()}`;
                imgEl.style.display = 'block';
            }

            // 오버레이를 성공 표시로 변경
            const overlay = imgEl ? imgEl.parentElement.querySelector('.regen-overlay') : null;
            if (overlay) {
                overlay.className = 'regen-overlay success';
                overlay.innerHTML = '<span class="regen-text">완료</span>';
                setTimeout(() => overlay.remove(), 1500);
            }

            this.showToast(`Scene ${sceneId} 이미지 재생성 완료!`, 'success');

        } catch (error) {
            console.error('[regenerateImage] Error:', error);
            this.showToast(`재생성 실패: ${error.message}`, 'error');

            // 오버레이 제거
            if (imgEl && imgEl.parentElement) {
                const overlay = imgEl.parentElement.querySelector('.regen-overlay');
                if (overlay) overlay.remove();
            }
        } finally {
            if (btn) {
                btn.textContent = '\uD83D\uDD04 재생성';
                btn.disabled = false;
            }
            this._regeneratingScenes.delete(regenKey);
        }
    }

    async testImageGeneration() {
        const btn = document.getElementById('test-image-btn');
        const resultDiv = document.getElementById('image-test-result');
        btn.disabled = true;
        btn.textContent = '🧪 테스트 중...';
        resultDiv.style.display = 'block';
        resultDiv.textContent = 'Gemini 이미지 모델 테스트 중...';

        try {
            const response = await fetch(`${this.getApiBaseUrl()}/api/test/image`);
            const data = await response.json();

            let output = `상태: ${data.status}\n`;
            output += `작동 모델: ${data.working_models?.join(', ') || '없음'}\n\n`;

            for (const d of (data.details || [])) {
                output += `--- ${d.model} ---\n`;
                output += `  HTTP: ${d.status_code || 'N/A'}\n`;
                output += `  이미지: ${d.has_image ? 'YES' : 'NO'}\n`;
                if (d.text) output += `  텍스트: ${d.text}\n`;
                if (d.finish_reason) output += `  finishReason: ${d.finish_reason}\n`;
                if (d.error) output += `  에러: ${d.error}\n`;
                if (d.test_image_url) {
                    output += `  테스트 이미지: ${d.test_image_url}\n`;
                    // 테스트 이미지 표시
                    const testImg = document.createElement('img');
                    testImg.src = `${this.getMediaBaseUrl()}${d.test_image_url}?t=${Date.now()}`;
                    testImg.style.cssText = 'max-width:200px; margin-top:8px; border-radius:8px;';
                    resultDiv.appendChild(document.createElement('br'));
                    resultDiv.appendChild(testImg);
                }
                output += '\n';
            }

            resultDiv.textContent = output;
            // Re-append image if working
            const working = data.details?.find(d => d.test_image_url);
            if (working) {
                const testImg = document.createElement('img');
                testImg.src = `${this.getMediaBaseUrl()}${working.test_image_url}?t=${Date.now()}`;
                testImg.style.cssText = 'max-width:200px; margin-top:8px; border-radius:8px;';
                resultDiv.appendChild(testImg);
            }
        } catch (error) {
            resultDiv.textContent = `테스트 실패: ${error.message}`;
        }

        btn.textContent = '🧪 이미지 생성 테스트';
        btn.disabled = false;
    }

    // ─── I2V Progress Banner Helpers ───

    _getI2VStageMessage(elapsed) {
        if (elapsed < 10) return 'Veo API 연결 중...';
        if (elapsed < 30) return '영상 생성 요청 전송 중...';
        if (elapsed < 90) return `AI 영상 생성 중... (${elapsed}초 경과)`;
        if (elapsed < 150) return `영상 생성 거의 완료... (${elapsed}초 경과)`;
        return `생성 대기 중... (${elapsed}초 경과)`;
    }

    _createI2VBanner(sceneId) {
        // 기존 배너 제거
        this._removeI2VBanner(sceneId);

        const banner = document.createElement('div');
        banner.className = 'i2v-progress-banner';
        banner.setAttribute('data-scene-id', sceneId);
        banner.innerHTML = `
            <div class="i2v-banner-icon">🎬</div>
            <div class="i2v-banner-content">
                <div class="i2v-banner-title">Scene ${sceneId} I2V 변환</div>
                <div class="i2v-banner-stage">Veo API 연결 중...</div>
                <div class="i2v-banner-bar">
                    <div class="i2v-banner-bar-fill" style="width: 0%"></div>
                </div>
            </div>
        `;
        document.body.appendChild(banner);

        // 타이머 시작
        const startTime = Date.now();
        const estimatedDuration = 120; // 120초 예상
        this._i2vTimers[sceneId] = setInterval(() => {
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            const stageEl = banner.querySelector('.i2v-banner-stage');
            const fillEl = banner.querySelector('.i2v-banner-bar-fill');
            if (stageEl) stageEl.textContent = this._getI2VStageMessage(elapsed);
            if (fillEl) {
                const pct = Math.min((elapsed / estimatedDuration) * 100, 90);
                fillEl.style.width = `${pct}%`;
            }
            // 기존 오버레이 텍스트도 업데이트
            this._updateI2VOverlayText(sceneId, elapsed);
        }, 1000);
    }

    _updateI2VOverlayText(sceneId, elapsed) {
        // 표준 I2V 오버레이
        const stdCard = document.querySelector(`.image-card[data-scene-id="${sceneId}"]`);
        if (stdCard) {
            const txt = stdCard.querySelector('.i2v-overlay-text');
            if (txt) txt.textContent = this._getI2VStageMessage(elapsed);
        }
        // MV I2V 오버레이
        const mvCard = document.querySelector(`.mv-review-card[data-scene-id="${sceneId}"]`);
        if (mvCard) {
            const txt = mvCard.querySelector('.regen-text');
            if (txt && txt.textContent.includes('I2V')) {
                txt.textContent = this._getI2VStageMessage(elapsed);
            }
        }
    }

    _removeI2VBanner(sceneId, success) {
        if (this._i2vTimers[sceneId]) {
            clearInterval(this._i2vTimers[sceneId]);
            delete this._i2vTimers[sceneId];
        }
        const banner = document.querySelector(`.i2v-progress-banner[data-scene-id="${sceneId}"]`);
        if (!banner) return;

        if (success === true) {
            banner.querySelector('.i2v-banner-icon').textContent = '✅';
            banner.querySelector('.i2v-banner-stage').textContent = '변환 완료!';
            banner.querySelector('.i2v-banner-bar-fill').style.width = '100%';
            banner.classList.add('i2v-banner-success');
            setTimeout(() => banner.remove(), 1500);
        } else if (success === false) {
            banner.querySelector('.i2v-banner-icon').textContent = '❌';
            banner.querySelector('.i2v-banner-stage').textContent = '변환 실패';
            banner.classList.add('i2v-banner-error');
            setTimeout(() => banner.remove(), 2000);
        } else {
            banner.remove();
        }
    }

    async convertToVideo(projectId, sceneId) {
        // 크레딧 사전 확인 (I2V)
        if (typeof checkCreditsBeforeAction === 'function') {
            const ok = await checkCreditsBeforeAction('i2v');
            if (!ok) return;
        }

        const card = document.querySelector(`.image-card[data-scene-id="${sceneId}"]`);
        if (!card) {
            this.showToast('I2V 실패: 씬 카드를 찾을 수 없습니다.', 'error');
            return;
        }
        const btn = card.querySelector('.btn-i2v');
        if (!btn) {
            this.showToast('I2V 실패: 버튼을 찾을 수 없습니다.', 'error');
            return;
        }
        const overlay = card.querySelector('.i2v-overlay');
        const regenBtn = card.querySelector('.btn-regenerate');

        btn.textContent = '⏳ 변환 중...';
        btn.disabled = true;
        if (regenBtn) regenBtn.disabled = true;

        // 오버레이 표시
        if (overlay) overlay.style.display = 'flex';

        // 진행 배너 표시
        this._createI2VBanner(sceneId);

        try {
            const response = await fetch(`${this.getApiBaseUrl()}/api/convert/i2v/${projectId}/${sceneId}`, {
                method: 'POST',
                headers: this.getAuthHeaders(),
                body: JSON.stringify({ motion_prompt: "camera slowly pans and zooms" })
            });

            if (!response.ok) {
                if (typeof handleApiError === 'function' && await handleApiError(response.clone(), 'i2v')) {
                    btn.textContent = '🎬 I2V';
                    btn.disabled = false;
                    if (regenBtn) regenBtn.disabled = false;
                    if (overlay) overlay.style.display = 'none';
                    this._removeI2VBanner(sceneId);
                    return;
                }
                let errorDetail = response.statusText;
                try {
                    const errorBody = await response.json();
                    errorDetail = errorBody.detail || errorBody.error || errorBody.message || JSON.stringify(errorBody);
                } catch (e) {
                    try { errorDetail = await response.text(); } catch (e2) { }
                }
                throw new Error(`${response.status}: ${errorDetail}`);
            }

            btn.textContent = '✅ I2V';
            btn.disabled = true;

            // 오버레이 → 완료 표시 후 fade out
            if (overlay) {
                overlay.innerHTML = '<div class="i2v-overlay-content"><span class="i2v-overlay-text">✅ 변환 완료!</span></div>';
                setTimeout(() => { overlay.style.display = 'none'; }, 1500);
            }

            // 헤더에 I2V 뱃지 추가
            const header = card.querySelector('.image-card-header');
            if (header && !header.querySelector('.i2v-done-badge')) {
                const badge = document.createElement('span');
                badge.className = 'i2v-done-badge';
                badge.textContent = 'I2V';
                header.appendChild(badge);
            }

            this._removeI2VBanner(sceneId, true);
            this.showToast(`Scene ${sceneId} I2V 변환 완료!`, 'success');

        } catch (error) {
            this._removeI2VBanner(sceneId, false);
            this.showToast(`I2V 변환 실패: ${error.message}`, 'error');
            btn.textContent = '🎬 I2V';
            btn.disabled = false;
            if (overlay) overlay.style.display = 'none';
        }

        if (regenBtn) regenBtn.disabled = false;
    }

    async startFinalGenerationAfterImageReview() {
        if (!this.projectId) {
            this.showToast('프로젝트 ID가 없습니다.', 'warning');
            return;
        }

        // 중복 생성 방지
        if (this.isGenerating) {
            this.showToast('이미 영상 생성이 진행 중입니다.', 'warning');
            return;
        }

        this.isGenerating = true;
        this._requestNotificationPermission();

        // 이미지 프리뷰 → 영상 생성: "장면 처리" 단계부터 시작
        this.showSection('progress');
        const progressTitle = document.getElementById('progress-title');
        if (progressTitle) progressTitle.textContent = '⏳ 영상 생성 중...';

        // 스토리/이미지는 이미 완료 → scenes 단계부터 시작
        this.resetProgress('scenes');
        this.updateProgress(25, '영상 생성 시작 중...');
        this.updateStepStatus('scenes', '장면 처리 준비 중');

        this.addLog('INFO', '📤 영상 생성 요청 전송 중...');

        try {
            const characterVoices = Object.values(this._characterVoices || {});
            const response = await fetch(`${this.getApiBaseUrl()}/api/generate/video`, {
                method: 'POST',
                headers: this.getAuthHeaders(),
                body: JSON.stringify({
                    project_id: this.projectId,
                    story_data: this.currentStoryData,
                    request_params: this.currentRequestParams,
                    character_voices: characterVoices.length > 0 ? characterVoices : [],
                })
            });

            if (!response.ok) {
                let errorDetail = response.statusText;
                try {
                    const errorBody = await response.json();
                    errorDetail = errorBody.detail || errorBody.error || errorBody.message || JSON.stringify(errorBody);
                } catch (e) {
                    try { errorDetail = await response.text(); } catch (e2) { }
                }
                throw new Error(`${response.status}: ${errorDetail}`);
            }

            const result = await response.json();
            this.projectId = result.project_id;

            this.addLog('INFO', `✅ 영상 생성 요청 접수됨 (Project ID: ${this.projectId})`);
            this.addLog('INFO', '⏳ 서버에서 영상 생성 중...');

            this.connectWebSocket(this.projectId);
            this.startPolling(this.projectId);

        } catch (error) {
            console.error('영상 생성 요청 실패:', error);
            this.addLog('ERROR', `❌ 오류: ${error.message}`);
            this.showToast('영상 생성에 실패했습니다. 다시 시도해주세요.', 'error');
            this.isGenerating = false;
            // 에러 시 이미지 프리뷰 화면으로 복귀
            this.showSection('image-preview');
        }
    }

    // ==================== Music Video Mode ====================

    initMVEventListeners() {
        // MV 네비게이션 - 클릭 시 이전 MV 상태 완전 초기화
        document.getElementById('nav-mv')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.resetMVUI();
        });

        // MV 파일 선택 시 파일명 표시
        const mvFileInput = document.getElementById('mv-music-file');
        const mvDropzone = document.getElementById('mv-dropzone');
        if (mvFileInput && mvDropzone) {
            mvFileInput.addEventListener('change', () => {
                const file = mvFileInput.files[0];
                const textEl = mvDropzone.querySelector('.file-dropzone__text');
                const hintEl = mvDropzone.querySelector('.file-dropzone__hint');
                if (file) {
                    const sizeMB = (file.size / 1024 / 1024).toFixed(1);
                    textEl.textContent = file.name;
                    hintEl.textContent = `${sizeMB} MB`;
                    mvDropzone.style.borderColor = 'var(--brand-primary)';
                    mvDropzone.style.background = 'rgba(99, 102, 241, 0.06)';
                } else {
                    textEl.textContent = '음악 파일을 드래그하거나 클릭해서 선택';
                    hintEl.textContent = 'MP3, WAV, M4A, OGG, FLAC (최대 10분)';
                    mvDropzone.style.borderColor = '';
                    mvDropzone.style.background = '';
                }
            });
        }

        // MV 폼 제출 (음악 업로드)
        document.getElementById('mv-form')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.uploadAndAnalyzeMusic();
        });

        // MV 분석 결과에서 뒤로
        document.getElementById('mv-back-btn')?.addEventListener('click', () => {
            this.showSection('mv');
        });

        // MV 생성 시작
        document.getElementById('mv-generate-btn')?.addEventListener('click', () => {
            this.startMVGeneration();
        });

        // MV 로그 클리어
        document.getElementById('mv-clear-log-btn')?.addEventListener('click', () => {
            document.getElementById('mv-log-content').innerHTML = '';
        });

        // MV 생성 중단
        document.getElementById('mv-cancel-btn')?.addEventListener('click', () => {
            this.cancelMVGeneration();
        });

        // MV 자막 테스트
        document.getElementById('mv-subtitle-test-btn')?.addEventListener('click', () => {
            this.mvSubtitleTest();
        });

        // 자막 전용 테스트 토글 → Demucs 옵션 표시/숨김
        document.getElementById('mv-subtitle-only')?.addEventListener('change', (e) => {
            const demucsWrap = document.getElementById('mv-subtitle-demucs-wrap');
            if (demucsWrap) demucsWrap.style.display = e.target.checked ? '' : 'none';
        });
    }

    async uploadAndAnalyzeMusic() {
        const fileInput = document.getElementById('mv-music-file');
        const file = fileInput.files[0];

        if (!file) {
            this.showToast('음악 파일을 선택해주세요.', 'warning');
            return;
        }

        // 이전 MV 상태 초기화
        this.stopMVPolling();
        const prevGrid = document.getElementById('mv-image-review-grid');
        if (prevGrid) prevGrid.innerHTML = '';
        const prevSceneGrid = document.getElementById('mv-scene-grid');
        if (prevSceneGrid) prevSceneGrid.innerHTML = '';
        const composeBtn = document.getElementById('mv-compose-btn');
        if (composeBtn) {
            composeBtn.disabled = false;
            composeBtn.innerHTML = '<span class="btn-icon">🎬</span> 최종 뮤직비디오 생성';
        }

        const btn = document.getElementById('mv-upload-btn');
        const originalText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = '<span class="btn-icon">⏳</span> 업로드 중...';

        try {
            const formData = new FormData();
            formData.append('music_file', file);
            formData.append('lyrics', document.getElementById('mv-lyrics').value || '');
            formData.append('concept', document.getElementById('mv-concept').value || '');
            formData.append('character_setup', document.getElementById('mv-character-setup').value);
            formData.append('character_ethnicity', document.getElementById('mv-character-ethnicity').value);
            formData.append('genre', document.getElementById('mv-genre').value);
            formData.append('mood', document.getElementById('mv-mood').value);
            formData.append('style', document.getElementById('mv-style').value);

            // Worker 경유 업로드 (인증 + 프록시)
            const baseUrl = this.getApiBaseUrl();
            const token = localStorage.getItem('token');
            const headers = {};
            if (token) headers['Authorization'] = `Bearer ${token}`;
            const response = await fetch(`${baseUrl}/api/mv/upload`, {
                method: 'POST',
                headers,
                body: formData
            });

            if (!response.ok) {
                let errorMsg = '음악 업로드 실패';
                try {
                    const error = await response.json();
                    errorMsg = error.detail || error.message || errorMsg;
                } catch (e) { }
                throw new Error(errorMsg);
            }

            const result = await response.json();
            this.mvProjectId = result.project_id;

            // 비동기 분석: status가 analyzing이면 폴링으로 완료 대기
            if (result.status === 'analyzing') {
                btn.innerHTML = '<span class="btn-icon">🎵</span> 음악 분석 중...';
                const analysisResult = await this.pollMVAnalysis(result.project_id);
                if (!analysisResult) {
                    throw new Error('음악 분석에 실패했습니다.');
                }
                this.mvAnalysis = analysisResult.music_analysis;
                // 폴링 결과를 result 형태로 합성
                result.music_analysis = analysisResult.music_analysis;
                result.status = 'ready';
            } else {
                // 동기 응답 (로컬 테스트 등)
                this.mvAnalysis = result.music_analysis;
            }

            // Gemini로 추출된 가사가 있고, 사용자가 직접 입력하지 않았으면 자동 채우기
            const lyricsInput = document.getElementById('mv-lyrics');
            if (result.extracted_lyrics && !lyricsInput.value.trim()) {
                lyricsInput.value = result.extracted_lyrics;
                console.log(`[MV] Auto-filled lyrics: ${result.extracted_lyrics.length} chars`);
            }

            this.mvRequestParams = {
                lyrics: lyricsInput.value || '',
                concept: document.getElementById('mv-concept').value || '',
                character_setup: document.getElementById('mv-character-setup').value,
                character_ethnicity: document.getElementById('mv-character-ethnicity').value,
                genre: document.getElementById('mv-genre').value,
                mood: document.getElementById('mv-mood').value,
                style: document.getElementById('mv-style').value,
                subtitle_enabled: document.getElementById('mv-subtitle-enabled')?.checked !== false
            };

            // 분석 결과 저장
            this.renderMVAnalysisResult(result);

            // 자막만 테스트 모드: 이미지 생성 없이 자막 테스트 실행
            if (document.getElementById('mv-subtitle-only')?.checked) {
                this.showSection('mv-analysis');
                this.mvSubtitleTest();
                return;
            }

            // 일반 모드: 바로 생성 시작
            this.startMVGeneration();

        } catch (error) {
            console.error('MV 업로드 실패:', error);
            this.showToast(error.message || '오류가 발생했습니다. 다시 시도해주세요.', 'error');
        } finally {
            btn.disabled = false;
            btn.innerHTML = originalText;
        }
    }

    /**
     * 음악 분석 완료까지 폴링 (2초 간격, 최대 120회 = 4분)
     * @returns {Object|null} 분석 완료 시 status 응답, 실패 시 null
     */
    async pollMVAnalysis(projectId) {
        const baseUrl = this.getApiBaseUrl();
        const maxAttempts = 180;  // 6분 (Demucs ~90s + Gemini align ~120s + STT)
        const intervalMs = 2000;

        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            await new Promise(resolve => setTimeout(resolve, intervalMs));

            try {
                const response = await fetch(`${baseUrl}/api/mv/status/${projectId}`, {
                    headers: this.getAuthHeaders()
                });

                if (!response.ok) {
                    console.warn(`[MV Analysis Poll] Status check failed: ${response.status}`);
                    continue;
                }

                const data = await response.json();
                console.log(`[MV Analysis Poll] attempt=${attempt + 1}, status=${data.status}`);

                if (data.status === 'ready') {
                    return data;
                } else if (data.status === 'failed') {
                    console.error('[MV Analysis Poll] Analysis failed:', data.error_message);
                    this.showToast(`음악 분석 실패: ${data.error_message || '알 수 없는 오류'}`, 'error');
                    return null;
                }
                // analyzing → 계속 폴링
            } catch (err) {
                console.warn('[MV Analysis Poll] Error:', err);
            }
        }

        console.error('[MV Analysis Poll] Timeout after max attempts');
        this.showToast('음악 분석 시간이 초과되었습니다. 다시 시도해주세요.', 'error');
        return null;
    }

    renderMVAnalysisResult(result) {
        const analysis = result.music_analysis;

        // 기본 정보 표시
        const durationMin = Math.floor(analysis.duration_sec / 60);
        const durationSec = Math.floor(analysis.duration_sec % 60);
        document.getElementById('mv-duration').textContent = `${durationMin}:${durationSec.toString().padStart(2, '0')}`;
        document.getElementById('mv-bpm').textContent = analysis.bpm ? Math.round(analysis.bpm) : '-';
        document.getElementById('mv-suggested-scenes').textContent = analysis.segments?.length || '-';
        document.getElementById('mv-detected-mood').textContent = analysis.mood || '-';

        // 씬 편집기 렌더링
        const editor = document.getElementById('mv-scene-editor');
        editor.innerHTML = '';

        const segments = analysis.segments || [];
        segments.forEach((seg, index) => {
            const card = document.createElement('div');
            card.className = 'mv-scene-card';
            card.style.cssText = 'background: #1a1a2e; border: 1px solid #393e46; border-radius: 8px; padding: 15px;';
            card.dataset.segmentIndex = index;

            const startMin = Math.floor(seg.start_sec / 60);
            const startSec = Math.floor(seg.start_sec % 60);
            const endMin = Math.floor(seg.end_sec / 60);
            const endSec = Math.floor(seg.end_sec % 60);

            card.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="color: #00adb5; font-weight: bold;">Scene ${index + 1}</span>
                    <span style="color: #888; font-size: 0.85rem;">${startMin}:${startSec.toString().padStart(2, '0')} - ${endMin}:${endSec.toString().padStart(2, '0')}</span>
                </div>
                <div style="display: flex; gap: 10px; margin-bottom: 10px;">
                    <span style="background: #393e46; padding: 3px 8px; border-radius: 12px; font-size: 0.8rem;">${seg.segment_type || 'verse'}</span>
                    <span style="color: #888; font-size: 0.85rem;">${seg.duration_sec?.toFixed(1) || '-'}초</span>
                </div>
                <label style="font-size: 0.85rem; color: #ccc; margin-bottom: 5px; display: block;">씬 설명 (선택)</label>
                <textarea class="mv-scene-description input" rows="2" placeholder="이 구간에 원하는 비주얼을 설명하세요..." style="width: 100%; font-size: 0.9rem;"></textarea>
            `;

            editor.appendChild(card);
        });
    }

    async startMVGeneration() {
        // 자막만 테스트 모드가 켜져있으면 자막 테스트로 리다이렉트
        const subtitleOnlyBox = document.getElementById('mv-subtitle-only');
        if (subtitleOnlyBox && subtitleOnlyBox.checked) {
            subtitleOnlyBox.checked = false;  // 다음 클릭 시 전체 생성 가능
            this.mvSubtitleTest();
            return;
        }

        // 크레딧 사전 확인
        if (typeof checkCreditsBeforeAction === 'function') {
            const ok = await checkCreditsBeforeAction('mv');
            if (!ok) return;
        }

        if (!this.mvProjectId) {
            this.showToast('프로젝트 ID가 없습니다. 다시 업로드해주세요.', 'warning');
            return;
        }

        this._requestNotificationPermission();

        // 씬 설명 수집
        const sceneDescriptions = [];
        document.querySelectorAll('.mv-scene-description').forEach((textarea, index) => {
            if (textarea.value.trim()) {
                sceneDescriptions.push({
                    segment_index: index,
                    description: textarea.value.trim()
                });
            }
        });

        const btn = document.getElementById('mv-generate-btn');
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<span class="btn-icon">⏳</span> 생성 요청 중...';
        }

        // auto_compose 플래그 저장 (폴링에서 사용)
        this.mvAutoCompose = document.getElementById('mv-auto-compose')?.checked || false;

        try {
            const baseUrl = this.getApiBaseUrl();
            const response = await fetch(`${baseUrl}/api/mv/generate`, {
                method: 'POST',
                headers: this.getAuthHeaders(),
                body: JSON.stringify({
                    project_id: this.mvProjectId,
                    lyrics: this.mvRequestParams?.lyrics || '',
                    concept: this.mvRequestParams?.concept || '',
                    character_setup: this.mvRequestParams?.character_setup || 'auto',
                    character_ethnicity: this.mvRequestParams?.character_ethnicity || 'auto',
                    genre: this.mvRequestParams?.genre || 'fantasy',
                    mood: this.mvRequestParams?.mood || 'epic',
                    style: this.mvRequestParams?.style || 'cinematic',
                    subtitle_enabled: this.mvRequestParams?.subtitle_enabled !== false,
                    watermark_enabled: this._shouldShowWatermark(),
                    max_scenes: document.getElementById('mv-quick-test')?.checked ? 5 : null,
                    preview_duration_sec: document.getElementById('mv-quick-test')?.checked ? 45 : null,
                    auto_compose: document.getElementById('mv-auto-compose')?.checked || false,
                    scene_descriptions: sceneDescriptions
                })
            });

            if (!response.ok) {
                if (typeof handleApiError === 'function' && await handleApiError(response.clone(), 'mv')) {
                    return;
                }
                let errorMsg = 'MV generation failed';
                try {
                    const error = await response.json();
                    errorMsg = error.detail || error.message || errorMsg;
                } catch (e) { }
                throw new Error(errorMsg);
            }

            const result = await response.json();

            // 클립 차감 반영
            if (typeof deductLocalClips === 'function') deductLocalClips('mv');

            // 진행 화면으로 전환 - 이전 데이터 클리어
            const sceneGrid = document.getElementById('mv-scene-grid');
            if (sceneGrid) sceneGrid.innerHTML = '';
            const logContent = document.getElementById('mv-log-content');
            if (logContent) logContent.innerHTML = '';

            this.showSection('mv-progress');
            this.mvAddLog('INFO', `✅ MV 생성 시작 (Project: ${this.mvProjectId})`);
            this.mvAddLog('INFO', `📊 총 ${result.total_scenes}개 씬, 예상 소요: ${Math.ceil(result.estimated_time_sec / 60)}분`);

            // 진행률 초기화
            this.updateMVProgress(5, '씬 프롬프트 생성 중...');
            this.updateMVStepStatus('scenes', '진행 중');

            // 폴링 시작 (새 생성이므로 단계 리셋)
            this._mvMinStageIdx = 0;
            this.startMVPolling(this.mvProjectId);

        } catch (error) {
            console.error('MV 생성 요청 실패:', error);
            this.showToast('오류가 발생했습니다. 다시 시도해주세요.', 'error');
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '<span class="btn-icon">🎬</span> 뮤직비디오 생성 시작';
            }
        }
    }

    startMVPolling(projectId) {
        if (this.mvPollingInterval) {
            clearInterval(this.mvPollingInterval);
        }

        const baseUrl = this.getApiBaseUrl();

        this.mvPollingFailCount = 0;
        // 단계 역행 방지: 한 번 지나간 단계로 되돌아가지 않음
        const _MV_STAGE_ORDER = ['generating', 'anchors_ready', 'images_ready', 'composing', 'completed', 'failed', 'cancelled'];
        if (this._mvMinStageIdx === undefined) this._mvMinStageIdx = 0;

        this.mvPollingInterval = setInterval(async () => {
            try {
                const response = await fetch(`${baseUrl}/api/mv/status/${projectId}`, {
                    headers: this.getAuthHeaders()
                });

                if (!response.ok) {
                    console.warn(`MV status check failed: ${response.status}`);
                    return;
                }

                // 연결 성공 시 실패 카운터 리셋
                this.mvPollingFailCount = 0;

                const data = await response.json();

                // 단계 역행 방지: 서버가 이전 단계 상태를 반환하면 무시
                const _stageIdx = _MV_STAGE_ORDER.indexOf(data.status);
                if (_stageIdx >= 0 && _stageIdx < this._mvMinStageIdx) {
                    console.warn(`[MV] Ignoring stale status '${data.status}' (min stage: ${_MV_STAGE_ORDER[this._mvMinStageIdx]})`);
                    return;
                }
                if (_stageIdx >= 0) this._mvMinStageIdx = _stageIdx;

                // 상태별 처리
                if (data.status === 'anchors_ready') {
                    if (this.mvAutoCompose) {
                        this.mvAddLog('SUCCESS', 'Character anchors ready. Auto mode: continuing...');
                        this.updateMVProgress(40, '캐릭터 생성 완료 → 이미지 생성 중...');
                        // 폴링 계속 (멈추지 않음)
                    } else {
                        this.mvAddLog('SUCCESS', 'Character anchors ready. Moving to review.');
                        this.updateMVProgress(40, '캐릭터 앵커 리뷰 대기');
                        this.stopMVPolling();
                        this.showMVCharacterReview(projectId, data.characters || []);
                    }

                } else if (data.status === 'images_ready') {
                    if (this.mvAutoCompose) {
                        this.mvAddLog('SUCCESS', 'Images ready. Auto mode: composing...');
                        this.updateMVProgress(70, '이미지 생성 완료 → 영상 합성 중...');
                        // 폴링 계속 (멈추지 않음)
                    } else {
                        this.mvAddLog('SUCCESS', 'Image generation complete. Moving to review.');
                        this.updateMVProgress(70, '이미지 리뷰 대기');
                        this.stopMVPolling();
                        this.showMVImageReview(projectId);
                    }

                } else if (data.status === 'completed') {
                    this.mvAddLog('SUCCESS', 'Music video generation complete!');
                    this.updateMVProgress(100, '완료');
                    this.stopMVPolling();
                    this._sendCompletionNotification('StoryCut 뮤직비디오 완성!', '뮤직비디오가 완성되었습니다. 클릭해서 확인하세요.');
                    this.fetchMVResult(projectId);

                } else if (data.status === 'failed') {
                    this.mvAddLog('ERROR', `❌ 오류: ${data.error_message || '알 수 없는 오류'}`);
                    this.updateMVProgress(0, '실패');
                    this.stopMVPolling();
                    this.showToast('뮤직비디오 생성에 실패했습니다. 다시 시도해주세요.', 'error');

                } else if (data.status === 'cancelled') {
                    this.mvAddLog('WARNING', '생성이 중단되었습니다.');
                    this.updateMVProgress(0, '중단됨');
                    this.stopMVPolling();
                    const cancelBtn = document.getElementById('mv-cancel-btn');
                    if (cancelBtn) {
                        cancelBtn.disabled = false;
                        cancelBtn.textContent = '생성 중단';
                    }

                } else {
                    // 진행 중
                    const progress = data.progress || 10;
                    const step = data.current_step || '';

                    this.updateMVProgress(progress, step);

                    // 단계 상태 업데이트
                    if (step.includes('씬') || step.includes('scene') || step.includes('프롬프트')) {
                        this.updateMVStepStatus('scenes', step);
                    } else if (step.includes('이미지') || step.includes('image')) {
                        this.updateMVStepStatus('images', step);
                    } else if (step.includes('합성') || step.includes('compose') || step.includes('비디오')) {
                        this.updateMVStepStatus('compose', step);
                    }

                    // 씬 그리드 업데이트 (이미지가 있으면)
                    if (data.scenes && data.scenes.length > 0) {
                        this.renderMVSceneGrid(data.scenes);
                    }
                }

            } catch (error) {
                this.mvPollingFailCount++;
                console.warn(`MV polling error (${this.mvPollingFailCount}/10):`, error.message);

                if (this.mvPollingFailCount >= 10) {
                    this.stopMVPolling();
                    this.mvAddLog('ERROR', '서버 연결이 끊어졌습니다. 페이지를 새로고침 해주세요.');
                    this.updateMVProgress(0, '서버 연결 끊김');
                    this.showToast('서버 연결이 끊어졌습니다. 잠시 후 새로고침 해주세요.', 'error');
                }
            }
        }, 3000);
    }

    stopMVPolling() {
        if (this.mvPollingInterval) {
            clearInterval(this.mvPollingInterval);
            this.mvPollingInterval = null;
        }
    }

    _restoreMVProgressUI() {
        // showMVCharacterReview가 덮어쓴 진행률 UI를 원본으로 복원
        if (this._mvProgressOriginalHTML) {
            const mvSection = document.getElementById('mv-progress-section');
            const container = mvSection ? mvSection.querySelector('.app-wide-container') : null;
            if (container) {
                container.innerHTML = this._mvProgressOriginalHTML;
            }
        }
    }

    async cancelMVGeneration() {
        if (!this.mvProjectId) return;

        const cancelBtn = document.getElementById('mv-cancel-btn');
        if (cancelBtn) {
            cancelBtn.disabled = true;
            cancelBtn.textContent = '중단 요청 중...';
        }

        try {
            const baseUrl = this.getApiBaseUrl();
            const resp = await fetch(`${baseUrl}/api/mv/cancel/${this.mvProjectId}`, { method: 'POST', headers: this.getAuthHeaders() });
            if (resp.ok) {
                this.mvAddLog('WARNING', '생성 중단을 요청했습니다. 현재 씬 완료 후 중단됩니다.');
                this.updateMVProgress(0, '중단 요청됨...');
            }
        } catch (e) {
            console.error('Cancel failed:', e);
        }
    }

    async fetchMVResult(projectId) {
        try {
            const baseUrl = this.getApiBaseUrl();
            const response = await fetch(`${baseUrl}/api/mv/result/${projectId}`, {
                headers: this.getAuthHeaders()
            });

            if (!response.ok) {
                throw new Error('결과 조회 실패');
            }

            const result = await response.json();
            this.showMVEditor(result.project_id, {
                showVideo: true,
                videoCompleted: true,
                scenes: result.scenes || [],
                duration_sec: result.duration_sec,
            });

        } catch (error) {
            console.error('MV 결과 조회 실패:', error);
            this.mvAddLog('ERROR', `결과 조회 실패: ${error.message}`);
        }
    }

    /**
     * showMVEditor - 통합 MV 씬 에디터
     * @param {string} projectId
     * @param {object} options - { showVideo, videoCompleted, headerText, scenes, duration_sec }
     */
    showMVEditor(projectId, options = {}) {
        const { showVideo = false, videoCompleted = false, headerText, scenes, duration_sec } = options;
        this.showSection('mv-image-review');
        this._currentMVResultProjectId = projectId;
        this.mvProjectId = projectId;

        const baseUrl = this.getApiBaseUrl();

        // 헤더 텍스트
        const header = document.getElementById('mv-editor-header');
        if (headerText) {
            header.textContent = headerText;
        } else if (showVideo && videoCompleted) {
            header.textContent = 'MV 완성';
        } else {
            header.textContent = '씬 검토';
        }

        // 비디오 영역
        const videoArea = document.getElementById('mv-editor-video-area');
        const video = document.getElementById('mv-editor-video');
        const downloadBtn = document.getElementById('mv-editor-download-btn');
        const recomposeBtn = document.getElementById('mv-editor-recompose-btn');
        const musicUploadBtn = document.getElementById('mv-editor-music-upload-btn');

        if (showVideo) {
            videoArea.style.display = '';
            video.src = `${baseUrl}/api/mv/stream/${projectId}?t=${Date.now()}`;
            downloadBtn.href = `${baseUrl}/api/mv/download/${projectId}`;
            downloadBtn.download = `musicvideo_${projectId}.mp4`;

            if (videoCompleted) {
                downloadBtn.style.display = '';
                recomposeBtn.style.display = 'none';
                musicUploadBtn.style.display = 'none';
            } else {
                // 미완성/실패: 재합성 표시, 다운로드 숨김
                downloadBtn.style.display = 'none';
                recomposeBtn.style.display = 'inline-flex';
                musicUploadBtn.style.display = 'inline-flex';
                video.src = ''; // 비디오 없으면 비움
                videoArea.style.display = 'none';
            }
        } else {
            videoArea.style.display = 'none';
        }

        // 하단 합성 버튼: 비디오 완성 상태면 숨김
        const composeBtn = document.getElementById('mv-compose-btn');
        if (composeBtn) {
            if (showVideo && videoCompleted) {
                composeBtn.style.display = 'none';
            } else {
                composeBtn.style.display = '';
                composeBtn.disabled = false;
                composeBtn.innerHTML = '<span class="btn-icon">🎬</span> 최종 뮤직비디오 생성';
                composeBtn.onclick = () => this.mvStartCompose(projectId);
            }
        }

        // 씬 그리드 렌더링 (scenes가 전달된 경우)
        if (scenes) {
            this.renderMVReviewGrid(scenes, projectId);
        }
    }

    /** showMVResult - showMVEditor로 위임 (하위 호환) */
    showMVResult(data) {
        this.showMVEditor(data.project_id, {
            showVideo: true,
            videoCompleted: true,
            scenes: data.scenes || [],
            duration_sec: data.duration_sec,
        });
    }

    renderMVSceneGrid(scenes) {
        const grid = document.getElementById('mv-scene-grid');

        // 스마트 업데이트: 이미 있는 씬은 건너뛰고, 새 씬만 추가
        scenes.forEach(scene => {
            if (scene.status === 'completed' && scene.image_path) {
                // 이미 해당 씬 카드가 있는지 확인
                const existingCard = grid.querySelector(`.scene-card[data-scene-id="${scene.scene_id}"]`);
                if (existingCard) {
                    return; // 이미 있으면 스킵 (깜빡임 방지)
                }

                const card = document.createElement('div');
                card.className = 'scene-card';
                card.setAttribute('data-scene-id', scene.scene_id);

                const imageUrl = this.resolveImageUrl(scene.image_path);

                card.innerHTML = `
                    <img src="${imageUrl}?t=${Date.now()}" alt="Scene ${scene.scene_id}"
                        loading="eager"
                        onerror="if(!this.dataset.retried){this.dataset.retried='1';this.src=this.src.split('?')[0]+'?retry='+Date.now();}else{this.style.display='none';this.nextElementSibling.style.display='flex';}">
                    <div class="scene-placeholder" style="display: none;">📷</div>
                    <div class="scene-info">
                        <span class="scene-narration">Scene ${scene.scene_id}</span>
                        <span class="scene-visual">${scene.lyrics_text || ''}</span>
                    </div>
                `;

                grid.appendChild(card);
            }
        });
    }


    updateMVProgress(progress, message) {
        const bar = document.getElementById('mv-progress-bar');
        const pct = document.getElementById('mv-progress-percentage');
        const msg = document.getElementById('mv-status-message');

        if (bar) bar.style.width = `${Math.min(progress, 100)}%`;
        if (pct) pct.textContent = `${Math.min(progress, 100)}%`;
        if (msg) msg.textContent = message;
    }

    updateMVStepStatus(step, message) {
        const container = document.getElementById('mv-steps-container');
        if (!container) return;

        // 모든 단계 비활성화
        container.querySelectorAll('.step').forEach(el => {
            el.classList.remove('active');
        });

        // 현재 단계 활성화
        const currentStep = container.querySelector(`[data-step="${step}"]`);
        if (currentStep) {
            currentStep.classList.add('active');
            currentStep.querySelector('.step-status').textContent = message;

            // 이전 단계들은 완료로
            let prev = currentStep.previousElementSibling;
            while (prev && prev.classList.contains('step')) {
                prev.classList.add('completed');
                prev.querySelector('.step-status').textContent = '완료';
                prev = prev.previousElementSibling;
            }
        }
    }

    mvAddLog(level, message) {
        const logContent = document.getElementById('mv-log-content');
        if (!logContent) return;

        const timestamp = new Date().toLocaleTimeString('ko-KR');
        const logLevel = level === 'ERROR' ? '❌' : level === 'SUCCESS' ? '✅' : level === 'WARNING' ? '⚠️' : 'ℹ️';

        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.innerHTML = `
            <span class="log-timestamp">[${timestamp}]</span>
            <span class="log-level">${logLevel}</span>
            <span class="log-message">${message}</span>
        `;
        logContent.appendChild(entry);
        logContent.scrollTop = logContent.scrollHeight;
    }

    // ==================== MV Character Anchor Review ====================

    showMVCharacterReview(projectId, characters) {
        // mv-progress-section 내부 컨테이너를 사용 (main-content는 index.html에 없음)
        const mvSection = document.getElementById('mv-progress-section');
        const mainContent = mvSection ? mvSection.querySelector('.app-wide-container') : document.getElementById('main-content');
        if (!mainContent) return;

        // 원본 진행률 UI 백업 (앵커 승인/compose 시 복원용)
        if (!this._mvProgressOriginalHTML) {
            this._mvProgressOriginalHTML = mainContent.innerHTML;
        }

        // MV 진행 섹션이 보이도록 보장
        if (mvSection) mvSection.classList.remove('hidden');

        const baseUrl = this.getApiBaseUrl();

        // 앵커 이미지 경로를 웹 URL로 변환
        const toWebUrl = (path) => {
            if (!path) return '';
            if (path.startsWith('http')) return path;
            // outputs/{project_id}/media/characters/xxx.png → /api/asset/{project_id}/image/xxx.png
            if (path.startsWith('outputs/')) {
                const parts = path.replace(/\\/g, '/').split('/');
                // parts: ["outputs", "{project_id}", "media", "characters", "xxx.png"]
                const pid = parts[1];
                const filename = parts[parts.length - 1];
                return `/api/asset/${pid}/image/${filename}`;
            }
            return path;
        };

        if (!characters || characters.length === 0) {
            mainContent.innerHTML = `
                <div style="max-width:900px; margin:0 auto; padding:40px 20px; text-align:center;">
                    <h2 style="color:#fff; margin:0 0 12px;">캐릭터 앵커 리뷰</h2>
                    <p style="color:rgba(255,255,255,0.6); margin:0 0 24px;">캐릭터 데이터를 불러올 수 없습니다. 잠시 후 다시 시도해주세요.</p>
                    <button onclick="location.reload()" style="
                        background: linear-gradient(135deg, #6c5ce7, #a29bfe);
                        color:#fff; border:none; padding:12px 24px;
                        border-radius:8px; font-size:14px; cursor:pointer;
                    ">새로고침</button>
                </div>`;
            return;
        }

        const characterCards = characters.map((char, idx) => {
            const frontImg = toWebUrl(char.anchor_image_path);
            const poses = char.anchor_poses || {};
            const fullBodyImg = toWebUrl(poses.full_body || '');
            const threeQuarterImg = toWebUrl(poses.three_quarter || '');

            return `
                <div class="mv-char-card" data-role="${char.role}" style="
                    background: var(--card-bg, #1a1a2e);
                    border-radius: 12px;
                    padding: 16px;
                    border: 1px solid rgba(255,255,255,0.1);
                ">
                    <div style="display:flex; gap:8px; margin-bottom:12px; flex-wrap:wrap; justify-content:center;">
                        ${frontImg ? `<img src="${frontImg.startsWith('http') ? frontImg : baseUrl + frontImg}" alt="front" style="width:120px; height:160px; object-fit:cover; border-radius:8px; border:2px solid #6c5ce7;">` : ''}
                        ${threeQuarterImg ? `<img src="${threeQuarterImg.startsWith('http') ? threeQuarterImg : baseUrl + threeQuarterImg}" alt="3/4" style="width:120px; height:160px; object-fit:cover; border-radius:8px; border:1px solid rgba(255,255,255,0.2);">` : ''}
                        ${fullBodyImg ? `<img src="${fullBodyImg.startsWith('http') ? fullBodyImg : baseUrl + fullBodyImg}" alt="full" style="width:120px; height:160px; object-fit:cover; border-radius:8px; border:1px solid rgba(255,255,255,0.2);">` : ''}
                    </div>
                    <h4 style="margin:0 0 4px; color:#fff; font-size:14px;">${char.role}</h4>
                    <p style="margin:0 0 8px; color:rgba(255,255,255,0.6); font-size:12px; line-height:1.4;">
                        ${char.description || ''}
                    </p>
                    ${char.outfit ? `<p style="margin:0; color:rgba(255,255,255,0.5); font-size:11px;">👗 ${char.outfit}</p>` : ''}
                </div>
            `;
        }).join('');

        const reviewHtml = `
            <div style="max-width:900px; margin:0 auto; padding:20px;">
                <div style="text-align:center; margin-bottom:24px;">
                    <h2 style="color:#fff; margin:0 0 8px;">캐릭터 앵커 리뷰</h2>
                    <p style="color:rgba(255,255,255,0.6); margin:0;">
                        생성된 캐릭터를 확인하세요. 승인하면 씬 이미지 생성이 시작됩니다.
                    </p>
                </div>

                <div style="
                    display:grid;
                    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                    gap:16px;
                    margin-bottom:24px;
                ">
                    ${characterCards}
                </div>

                <div style="display:flex; gap:12px; justify-content:center; flex-wrap:wrap;">
                    <button id="mv-regenerate-anchors-btn" style="
                        background: rgba(255,255,255,0.1);
                        color:#fff; border:1px solid rgba(255,255,255,0.3); padding:14px 24px;
                        border-radius:8px; font-size:14px; font-weight:500;
                        cursor:pointer; transition: opacity 0.2s;
                    ">캐릭터 재생성</button>
                    <button id="mv-approve-anchors-btn" style="
                        background: linear-gradient(135deg, #6c5ce7, #a29bfe);
                        color:#fff; border:none; padding:14px 32px;
                        border-radius:8px; font-size:16px; font-weight:600;
                        cursor:pointer; transition: opacity 0.2s;
                    ">승인 및 이미지 생성</button>
                </div>
            </div>
        `;

        mainContent.innerHTML = reviewHtml;

        // 재생성 버튼 이벤트
        const regenBtn = document.getElementById('mv-regenerate-anchors-btn');
        if (regenBtn) {
            regenBtn.addEventListener('click', async () => {
                regenBtn.disabled = true;
                regenBtn.textContent = '재생성 중...';
                regenBtn.style.opacity = '0.6';

                try {
                    const resp = await fetch(`${baseUrl}/api/mv/regenerate/anchors/${projectId}`, {
                        method: 'POST',
                        headers: this.getAuthHeaders()
                    });

                    if (!resp.ok) {
                        const err = await resp.json();
                        throw new Error(err.detail || 'Failed to regenerate anchors');
                    }

                    this.showToast('캐릭터를 재생성합니다', 'success');
                    this.showSection('mv-progress');
                    this.updateMVProgress(30, '캐릭터 앵커 재생성 중...');
                    // 재생성이므로 단계 역행 방지를 리셋 (anchors_ready 복귀 허용)
                    this._mvMinStageIdx = 0;
                    this.startMVPolling(projectId);

                } catch (error) {
                    console.error('Failed to regenerate anchors:', error);
                    this.showToast(`재생성 실패: ${error.message}`, 'error');
                    regenBtn.disabled = false;
                    regenBtn.textContent = '캐릭터 재생성';
                    regenBtn.style.opacity = '1';
                }
            });
        }

        // 승인 버튼 이벤트
        const approveBtn = document.getElementById('mv-approve-anchors-btn');
        if (approveBtn) {
            approveBtn.addEventListener('click', async () => {
                approveBtn.disabled = true;
                approveBtn.textContent = '이미지 생성 시작 중...';
                approveBtn.style.opacity = '0.6';

                try {
                    const resp = await fetch(`${baseUrl}/api/mv/generate/images/${projectId}`, {
                        method: 'POST',
                        headers: this.getAuthHeaders()
                    });

                    if (!resp.ok) {
                        const err = await resp.json();
                        throw new Error(err.detail || 'Failed to start image generation');
                    }

                    this.showToast('이미지 생성이 시작되었습니다', 'success');

                    // 앵커 리뷰로 덮어쓴 진행률 UI 복원
                    this._restoreMVProgressUI();

                    // MV 생성 UI로 돌아가서 폴링 재개 — anchors_ready 이후이므로 역행 방지
                    this._mvMinStageIdx = 2; // 'images_ready' 이전 단계(generating)로 역행 불가
                    this.showSection('mv-progress');
                    this.updateMVProgress(45, '이미지 생성 중...');
                    this.startMVPolling(projectId);

                } catch (error) {
                    console.error('Failed to start image generation:', error);
                    this.showToast(`이미지 생성 시작 실패: ${error.message}`, 'error');
                    approveBtn.disabled = false;
                    approveBtn.textContent = '승인 및 이미지 생성';
                    approveBtn.style.opacity = '1';
                }
            });
        }
    }

    // ==================== MV Image Review ====================

    async showMVImageReview(projectId) {
        try {
            const baseUrl = this.getApiBaseUrl();
            const response = await fetch(`${baseUrl}/api/mv/status/${projectId}`, {
                headers: this.getAuthHeaders()
            });
            if (!response.ok) throw new Error('Failed to load project');

            const data = await response.json();
            this.showMVEditor(projectId, {
                showVideo: false,
                scenes: data.scenes,
            });
        } catch (error) {
            console.error('MV Image Review load failed:', error);
            this.showToast('이미지 리뷰 로드 실패', 'error');
        }
    }

    renderMVReviewGrid(scenes, projectId) {
        const grid = document.getElementById('mv-image-review-grid');
        if (!grid) return;
        grid.innerHTML = '';

        scenes.forEach(scene => {
            const card = document.createElement('div');
            card.className = 'mv-review-card';
            card.setAttribute('data-scene-id', scene.scene_id);

            const imageUrl = scene.image_path ? this.resolveImageUrl(scene.image_path) : '';
            const startTime = scene.start_sec != null ? this._formatTime(scene.start_sec) : '';
            const endTime = scene.end_sec != null ? this._formatTime(scene.end_sec) : '';
            const timeBadge = startTime ? `${startTime} - ${endTime}` : '';
            const lyrics = scene.lyrics_text || '';
            const hasI2V = !!scene.video_path;

            card.innerHTML = `
                <div class="mv-review-img-wrap">
                    ${imageUrl
                    ? `<img src="${imageUrl}?t=${Date.now()}" alt="Scene ${scene.scene_id}"
                            onerror="if(!this.dataset.retried){this.dataset.retried='1';this.src=this.src.split('?')[0]+'?retry='+Date.now();}">`
                    : '<div style="width:100%;aspect-ratio:16/9;background:#2a2d35;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#666;font-size:2rem;">📷</div>'}
                    ${hasI2V ? '<span class="i2v-badge">I2V</span>' : ''}
                </div>
                <div class="mv-review-info">
                    <span style="font-weight:bold;">Scene ${scene.scene_id}</span>
                    <span class="mv-review-time">${timeBadge}</span>
                </div>
                ${lyrics ? `<div class="mv-review-lyrics" title="${lyrics}">${lyrics}</div>` : ''}
                <div class="mv-prompt-area" style="margin:4px 0;">
                    <div class="mv-prompt-text" style="font-size:0.7rem;color:#888;max-height:2.4em;overflow:hidden;cursor:pointer;word-break:break-all;line-height:1.2em;"
                        title="클릭하여 프롬프트 편집"
                        onclick="app.togglePromptEdit(this, '${projectId}', ${scene.scene_id})"
                    >${(scene.image_prompt || '').replace(/'/g, '&#39;').replace(/"/g, '&quot;')}</div>
                </div>
                <div class="mv-review-actions">
                    <button class="mv-regen-btn" onclick="app.mvRegenerateScene('${projectId}', ${scene.scene_id})">
                        🔄 재생성
                    </button>
                    <button class="mv-i2v-btn" onclick="app.mvGenerateI2V('${projectId}', ${scene.scene_id})"
                        ${hasI2V ? 'disabled' : ''}>
                        ${hasI2V ? '✅ I2V' : '🎬 I2V'}
                    </button>
                </div>
            `;

            grid.appendChild(card);
        });
    }

    togglePromptEdit(el, projectId, sceneId) {
        const area = el.closest('.mv-prompt-area');
        // 이미 편집 중이면 무시
        if (area.querySelector('textarea')) return;

        const currentText = el.textContent.trim();
        el.style.display = 'none';

        const textarea = document.createElement('textarea');
        textarea.value = currentText;
        textarea.style.cssText = 'width:100%;min-height:60px;font-size:0.75rem;background:#1e1e2e;color:#ccc;border:1px solid #555;border-radius:4px;padding:4px;resize:vertical;font-family:inherit;';

        const btnWrap = document.createElement('div');
        btnWrap.style.cssText = 'display:flex;gap:4px;margin-top:3px;';
        btnWrap.innerHTML = `
            <button style="flex:1;padding:3px 6px;font-size:0.7rem;background:#f59e0b;border:none;border-radius:4px;color:#000;cursor:pointer;font-weight:600;">저장 후 재생성</button>
            <button style="padding:3px 6px;font-size:0.7rem;background:#333;border:1px solid #555;border-radius:4px;color:#ccc;cursor:pointer;">취소</button>
        `;

        area.appendChild(textarea);
        area.appendChild(btnWrap);
        textarea.focus();

        // 저장 후 재생성
        btnWrap.children[0].onclick = () => {
            const newPrompt = textarea.value.trim();
            textarea.remove();
            btnWrap.remove();
            el.textContent = newPrompt || currentText;
            el.style.display = '';
            if (newPrompt && newPrompt !== currentText) {
                // custom_prompt와 함께 재생성
                this.mvRegenerateScene(projectId, sceneId, newPrompt);
            }
        };

        // 취소
        btnWrap.children[1].onclick = () => {
            textarea.remove();
            btnWrap.remove();
            el.style.display = '';
        };
    }

    async mvRegenerateScene(projectId, sceneId, customPrompt = null) {
        // 크레딧 사전 확인 (이미지 재생성)
        if (typeof checkCreditsBeforeAction === 'function') {
            const ok = await checkCreditsBeforeAction('image_regen');
            if (!ok) return;
        }

        const regenKey = `mv_${projectId}_${sceneId}`;
        if (this._regeneratingScenes.has(regenKey)) {
            this.showToast('이미 재생성 중입니다', 'warning');
            return;
        }

        const card = document.querySelector(`.mv-review-card[data-scene-id="${sceneId}"]`);
        if (!card) return;

        this._regeneratingScenes.add(regenKey);
        card.classList.add('regenerating');

        // 로딩 오버레이
        const imgWrap = card.querySelector('.mv-review-img-wrap');
        const overlay = document.createElement('div');
        overlay.className = 'regen-overlay';
        overlay.innerHTML = '<div class="regen-spinner"></div><span class="regen-text">이미지 재생성 중...</span>';
        imgWrap.appendChild(overlay);

        // 버튼 비활성화
        const btn = card.querySelector('.mv-regen-btn');
        if (btn) btn.disabled = true;

        try {
            const baseUrl = this.getApiBaseUrl();
            const fetchBody = customPrompt ? JSON.stringify({ custom_prompt: customPrompt }) : '{}';
            const response = await fetch(`${baseUrl}/api/mv/scenes/${projectId}/${sceneId}/regenerate`, {
                method: 'POST',
                headers: this.getAuthHeaders(),
                body: fetchBody,
            });

            if (!response.ok) {
                if (typeof handleApiError === 'function' && await handleApiError(response.clone(), 'image_regen')) {
                    if (btn) btn.disabled = false;
                    return;
                }
                const err = await response.json();
                throw new Error(err.detail || 'Regeneration failed');
            }

            const result = await response.json();

            // 이미지 업데이트
            const img = imgWrap.querySelector('img');
            if (img && result.image_url) {
                const imageUrl = `${this.getMediaBaseUrl()}${result.image_url}`;
                img.src = `${imageUrl}?t=${Date.now()}`;
            }

            // I2V 뱃지 제거 (이미지 변경으로 I2V 무효화)
            const badge = imgWrap.querySelector('.i2v-badge');
            if (badge) badge.remove();

            // I2V 버튼 리셋
            const i2vBtn = card.querySelector('.mv-i2v-btn');
            if (i2vBtn) { i2vBtn.disabled = false; i2vBtn.textContent = '🎬 I2V'; }

            // 오버레이 성공 표시
            overlay.className = 'regen-overlay success';
            overlay.innerHTML = '<span class="regen-text">완료</span>';
            setTimeout(() => overlay.remove(), 1500);

            // 재합성 버튼 표시
            const recomposeBtn = document.getElementById('mv-editor-recompose-btn');
            if (recomposeBtn) recomposeBtn.style.display = 'inline-flex';

            this.showToast(`Scene ${sceneId} 이미지 재생성 완료`, 'success');
        } catch (error) {
            console.error('MV regeneration failed:', error);
            overlay.remove();
            this.showToast(`재생성 실패: ${error.message}`, 'error');
        } finally {
            this._regeneratingScenes.delete(regenKey);
            card.classList.remove('regenerating');
            if (btn) btn.disabled = false;
        }
    }

    async mvGenerateI2V(projectId, sceneId) {
        const i2vKey = `i2v_${projectId}_${sceneId}`;
        if (this._regeneratingScenes.has(i2vKey)) {
            this.showToast('이미 I2V 변환 중입니다', 'warning');
            return;
        }

        const card = document.querySelector(`.mv-review-card[data-scene-id="${sceneId}"]`);
        if (!card) return;

        this._regeneratingScenes.add(i2vKey);

        // 로딩 오버레이
        const imgWrap = card.querySelector('.mv-review-img-wrap');
        const overlay = document.createElement('div');
        overlay.className = 'regen-overlay';
        overlay.innerHTML = '<div class="regen-spinner"></div><span class="regen-text">I2V 변환 중...</span>';
        imgWrap.appendChild(overlay);

        // 버튼 비활성화
        const btn = card.querySelector('.mv-i2v-btn');
        if (btn) btn.disabled = true;

        // 진행 배너 표시
        this._createI2VBanner(sceneId);

        try {
            const baseUrl = this.getApiBaseUrl();
            const response = await fetch(`${baseUrl}/api/mv/scenes/${projectId}/${sceneId}/i2v`, {
                method: 'POST',
                headers: this.getAuthHeaders()
            });

            if (!response.ok) {
                if (typeof handleApiError === 'function' && await handleApiError(response.clone(), 'i2v')) {
                    if (btn) btn.disabled = false;
                    overlay.remove();
                    this._removeI2VBanner(sceneId);
                    return;
                }
                const err = await response.json();
                throw new Error(err.detail || 'I2V failed');
            }

            // I2V 뱃지 추가
            let badge = imgWrap.querySelector('.i2v-badge');
            if (!badge) {
                badge = document.createElement('span');
                badge.className = 'i2v-badge';
                badge.textContent = 'I2V';
                imgWrap.appendChild(badge);
            }

            overlay.className = 'regen-overlay success';
            overlay.innerHTML = '<span class="regen-text">I2V 완료</span>';
            setTimeout(() => overlay.remove(), 1500);

            // 버튼 상태 업데이트
            if (btn) { btn.textContent = '✅ I2V'; btn.disabled = true; }

            // 재합성 버튼 표시
            const recomposeBtn = document.getElementById('mv-editor-recompose-btn');
            if (recomposeBtn) recomposeBtn.style.display = 'inline-flex';

            this._removeI2VBanner(sceneId, true);
            this.showToast(`Scene ${sceneId} I2V 변환 완료`, 'success');
        } catch (error) {
            console.error('MV I2V failed:', error);
            this._removeI2VBanner(sceneId, false);
            overlay.remove();
            this.showToast(`I2V 실패: ${error.message}`, 'error');
        } finally {
            this._regeneratingScenes.delete(i2vKey);
            if (btn) btn.disabled = false;
        }
    }

    async mvStartCompose(projectId) {
        const btn = document.getElementById('mv-compose-btn');
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<span class="btn-icon">⏳</span> 합성 시작 중...';
        }

        try {
            // Worker 경유 compose (인증 + 프록시)
            const baseUrl = this.getApiBaseUrl();
            const token = localStorage.getItem('token');
            const headers = { 'Content-Type': 'application/json' };
            if (token) headers['Authorization'] = `Bearer ${token}`;
            const response = await fetch(`${baseUrl}/api/mv/compose/${projectId}`, {
                method: 'POST',
                headers
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Compose failed');
            }

            // 앵커 리뷰로 덮어쓴 진행률 UI 복원 + 화면 전환
            this._restoreMVProgressUI();
            this.showSection('mv-progress');
            this.updateMVProgress(75, '영상 합성 중...');
            this.updateMVStepStatus('compose', '영상 합성 중...');
            this.mvAddLog('INFO', '🎬 최종 뮤직비디오 합성을 시작합니다.');

            // 폴링 재시작 (completed 대기) — composing 이전 단계로 역행 방지
            this._mvMinStageIdx = 3; // 'composing' index in _MV_STAGE_ORDER
            this.startMVPolling(projectId);
        } catch (error) {
            console.error('MV compose failed:', error);
            this.showToast(`합성 시작 실패: ${error.message}`, 'error');
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '<span class="btn-icon">🎬</span> 최종 뮤직비디오 생성';
            }
        }
    }

    async mvSubtitleTest() {
        const projectId = this.mvProjectId;
        if (!projectId) {
            this.showToast('프로젝트 ID를 찾을 수 없습니다', 'error');
            return;
        }

        const btn = document.getElementById('mv-subtitle-test-btn');
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<span class="btn-icon">⏳</span> 자막 생성 중...';
        }

        try {
            const baseUrl = this.getApiBaseUrl();
            const useDemucs = document.getElementById('mv-subtitle-demucs')?.checked !== false;
            const response = await fetch(`${baseUrl}/api/mv/subtitle-test/${projectId}`, {
                method: 'POST',
                headers: { ...this.getAuthHeaders(), 'Content-Type': 'application/json' },
                body: JSON.stringify({ use_demucs: useDemucs })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Subtitle test failed');
            }

            const demucsLabel = useDemucs ? 'Demucs 적용' : 'Demucs 미적용 (원본)';
            this.showToast(`자막 테스트 생성 중 (${demucsLabel})... 완료되면 자동으로 미리보기가 열립니다.`, 'success');

            // 폴링으로 완료 대기 (5초 간격, 최대 5분)
            const maxAttempts = 60;
            let attempt = 0;
            const pollInterval = setInterval(async () => {
                attempt++;
                try {
                    const statusResp = await fetch(`${baseUrl}/api/mv/status/${projectId}`, {
                        headers: this.getAuthHeaders()
                    });
                    if (!statusResp.ok) return;
                    const data = await statusResp.json();

                    const isDone = data.current_step === '자막 테스트 완료';
                    const isFailed = data.error_message && data.error_message.includes('Subtitle test');

                    if (isDone || isFailed || attempt >= maxAttempts) {
                        clearInterval(pollInterval);
                        if (btn) {
                            btn.disabled = false;
                            btn.innerHTML = '<span class="btn-icon">📝</span> 자막 테스트';
                        }

                        if (isFailed) {
                            this.showToast(`자막 테스트 실패: ${data.error_message}`, 'error');
                            return;
                        }
                        if (attempt >= maxAttempts && !isDone) {
                            this.showToast('자막 테스트 시간 초과 (5분)', 'error');
                            return;
                        }

                        // 성공: 영상 팝업
                        const testVideoUrl = `${baseUrl}/api/asset/${projectId}/video/final_mv_subtitle_test.mp4`;
                        const overlay = document.createElement('div');
                        overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.85);z-index:9999;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:16px;';
                        overlay.innerHTML = `
                            <h3 style="color:#fff;margin:0;">자막 테스트 미리보기</h3>
                            <video controls autoplay style="max-width:90vw;max-height:70vh;border-radius:8px;" src="${testVideoUrl}"></video>
                            <div style="display:flex;gap:12px;">
                                <button class="close-btn" style="padding:10px 24px;border-radius:8px;border:none;background:#666;color:#fff;cursor:pointer;font-size:1rem;">닫기</button>
                            </div>
                        `;
                        overlay.querySelector('.close-btn').addEventListener('click', () => overlay.remove());
                        overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
                        document.body.appendChild(overlay);
                    } else if (btn) {
                        btn.innerHTML = `<span class="btn-icon">⏳</span> 자막 생성 중... (${attempt * 5}초)`;
                    }
                } catch (pollErr) {
                    console.error('Subtitle test poll error:', pollErr);
                }
            }, 5000);

        } catch (error) {
            console.error('Subtitle test failed:', error);
            this.showToast(`자막 테스트 실패: ${error.message}`, 'error');
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '<span class="btn-icon">📝</span> 자막 테스트';
            }
        }
    }



    // ── MV 리컴포즈 ──

    async mvUploadMusicForRecompose(file) {
        if (!file) return;
        const projectId = this._currentMVResultProjectId;
        if (!projectId) {
            this.showToast('프로젝트 ID를 찾을 수 없습니다', 'error');
            return;
        }

        const musicBtn = document.getElementById('mv-editor-music-upload-btn');
        if (musicBtn) musicBtn.style.opacity = '0.5';

        try {
            const baseUrl = this.getApiBaseUrl();
            const formData = new FormData();
            formData.append('music_file', file);

            const token = localStorage.getItem('token');
            const uploadHeaders = {};
            if (token) uploadHeaders['Authorization'] = `Bearer ${token}`;
            const response = await fetch(`${baseUrl}/api/mv/${projectId}/upload-music`, {
                method: 'POST',
                headers: uploadHeaders,
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Upload failed');
            }

            this.showToast(`음악 파일 업로드 완료: ${file.name}`, 'success');
            if (musicBtn) musicBtn.style.display = 'none';
        } catch (error) {
            console.error('Music upload failed:', error);
            this.showToast(`음악 업로드 실패: ${error.message}`, 'error');
        } finally {
            if (musicBtn) musicBtn.style.opacity = '1';
        }
    }

    async mvRecompose() {
        // 크레딧 사전 확인 (MV 리컴포즈)
        if (typeof checkCreditsBeforeAction === 'function') {
            const ok = await checkCreditsBeforeAction('mv_recompose');
            if (!ok) return;
        }

        const projectId = this._currentMVResultProjectId;
        if (!projectId) {
            this.showToast('프로젝트 ID를 찾을 수 없습니다', 'error');
            return;
        }

        const recomposeBtn = document.getElementById('mv-editor-recompose-btn');
        if (recomposeBtn) {
            recomposeBtn.disabled = true;
            recomposeBtn.innerHTML = '<span class="btn-icon">⏳</span> 재합성 중...';
        }

        try {
            const baseUrl = this.getApiBaseUrl();
            const response = await fetch(`${baseUrl}/api/mv/${projectId}/recompose`, {
                method: 'POST',
                headers: this.getAuthHeaders()
            });

            if (!response.ok) {
                if (typeof handleApiError === 'function' && await handleApiError(response.clone(), 'mv_recompose')) {
                    if (recomposeBtn) {
                        recomposeBtn.disabled = false;
                        recomposeBtn.innerHTML = '<span class="btn-icon">🔄</span> 재합성';
                    }
                    return;
                }
                const err = await response.json();
                throw new Error(err.detail || '재합성 실패');
            }

            this.showToast('영상 재합성 중...', 'info');

            // 폴링으로 완료 대기
            this._pollMVRecompose(projectId);

        } catch (error) {
            console.error('MV recompose failed:', error);
            this.showToast(`재합성 실패: ${error.message}`, 'error');
            if (recomposeBtn) {
                recomposeBtn.disabled = false;
                recomposeBtn.innerHTML = '<span class="btn-icon">🔄</span> 영상 재합성 (수정 반영)';
            }
        }
    }

    async _pollMVRecompose(projectId) {
        const baseUrl = this.getApiBaseUrl();
        const recomposeBtn = document.getElementById('mv-editor-recompose-btn');
        const maxAttempts = 120; // 최대 4분 (2초 간격)
        let attempts = 0;

        const poll = async () => {
            attempts++;
            try {
                const response = await fetch(`${baseUrl}/api/mv/status/${projectId}`, {
                    headers: this.getAuthHeaders()
                });
                if (!response.ok) throw new Error('Status check failed');

                const data = await response.json();

                if (data.status === 'completed' || data.status === 'COMPLETED') {
                    // 비디오 영역 표시 및 갱신
                    const videoArea = document.getElementById('mv-editor-video-area');
                    if (videoArea) videoArea.style.display = '';
                    const video = document.getElementById('mv-editor-video');
                    if (video) {
                        video.src = `${baseUrl}/api/mv/stream/${projectId}?t=${Date.now()}`;
                    }
                    const downloadBtn = document.getElementById('mv-editor-download-btn');
                    if (downloadBtn) {
                        downloadBtn.href = `${baseUrl}/api/mv/download/${projectId}?t=${Date.now()}`;
                        downloadBtn.style.display = '';
                    }
                    // 합성 버튼 숨김
                    const composeBtn = document.getElementById('mv-compose-btn');
                    if (composeBtn) composeBtn.style.display = 'none';
                    // 헤더 업데이트
                    const header = document.getElementById('mv-editor-header');
                    if (header) header.textContent = 'MV 완성';
                    // 리컴포즈 버튼 숨기기
                    if (recomposeBtn) {
                        recomposeBtn.style.display = 'none';
                        recomposeBtn.disabled = false;
                        recomposeBtn.innerHTML = '<span class="btn-icon">🔄</span> 영상 재합성 (수정 반영)';
                    }
                    this.showToast('영상 재합성 완료!', 'success');
                    return;
                }

                if (data.status === 'failed' || data.status === 'FAILED') {
                    throw new Error(data.error_message || '재합성 실패');
                }

                // 아직 진행 중 - 계속 폴링
                if (attempts < maxAttempts) {
                    setTimeout(poll, 2000);
                } else {
                    throw new Error('재합성 시간 초과');
                }

            } catch (error) {
                console.error('MV recompose poll error:', error);
                this.showToast(`재합성 실패: ${error.message}`, 'error');
                if (recomposeBtn) {
                    recomposeBtn.disabled = false;
                    recomposeBtn.innerHTML = '<span class="btn-icon">🔄</span> 영상 재합성 (수정 반영)';
                }
            }
        };

        setTimeout(poll, 2000); // 2초 후 첫 폴링
    }

    // ================================================================
    // 가사 타이밍 에디터
    // ================================================================

    async openLyricsTimeline() {
        const projectId = this.mvProjectId || this._currentMVResultProjectId;
        if (!projectId) {
            this.showToast('프로젝트 ID를 찾을 수 없습니다', 'error');
            return;
        }

        try {
            const baseUrl = this.getApiBaseUrl();
            const resp = await fetch(`${baseUrl}/api/mv/${projectId}/lyrics-timeline`, {
                headers: this.getAuthHeaders()
            });
            if (!resp.ok) throw new Error('Failed to load lyrics timeline');

            const data = await resp.json();
            this._timelineOriginalData = data;
            this._renderTimelineTable(data);

            document.getElementById('lyrics-timeline-overlay').classList.add('active');
        } catch (err) {
            console.error('openLyricsTimeline error:', err);
            this.showToast(`가사 타이밍 로드 실패: ${err.message}`, 'error');
        }
    }

    closeLyricsTimeline() {
        document.getElementById('lyrics-timeline-overlay').classList.remove('active');
    }

    _renderTimelineTable(data) {
        const tbody = document.getElementById('lyrics-timeline-tbody');
        tbody.innerHTML = '';

        const sttMap = new Map();
        (data.stt_sentences || []).forEach(s => {
            sttMap.set(Math.round(s.t * 10), s.text);
        });

        const lyrics = data.timed_lyrics || [];
        if (lyrics.length === 0 && (data.stt_sentences || []).length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:#888;padding:20px;">가사 타이밍 데이터가 없습니다</td></tr>';
            return;
        }

        // Merge: timed_lyrics를 기준으로 STT 매칭
        const rows = [];
        lyrics.forEach(entry => {
            const tKey = Math.round(entry.t * 10);
            // 가장 가까운 STT 매칭 (+-0.5초 이내)
            let sttText = '';
            for (let offset = 0; offset <= 5; offset++) {
                if (sttMap.has(tKey + offset)) { sttText = sttMap.get(tKey + offset); sttMap.delete(tKey + offset); break; }
                if (offset > 0 && sttMap.has(tKey - offset)) { sttText = sttMap.get(tKey - offset); sttMap.delete(tKey - offset); break; }
            }
            const cleanText = (entry.text || '').replace(/^[""\u201c]+|[""\u201d]+$/g, '');
            rows.push({ t: entry.t, stt: sttText, text: cleanText });
        });

        // 매칭 안 된 STT 엔트리도 추가 (회색으로)
        sttMap.forEach((text, tKey) => {
            rows.push({ t: tKey / 10, stt: text, text: '' });
        });

        // 시간순 정렬
        rows.sort((a, b) => a.t - b.t);

        rows.forEach((row, idx) => {
            this._appendTimelineRow(tbody, row.t, row.stt, row.text, idx);
        });
    }

    _appendTimelineRow(tbody, t, sttText, lyricsText, idx) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td class="col-time">
                <input type="text" class="time-input" value="${this._formatTimeMS(t)}" data-orig-sec="${t}">
            </td>
            <td class="col-stt">
                <span class="stt-text">${sttText || '(없음)'}</span>
            </td>
            <td class="col-lyrics">
                <input type="text" class="lyrics-input" value="${(lyricsText || '').replace(/"/g, '&quot;')}" placeholder="자막 텍스트 입력...">
            </td>
            <td class="col-del">
                <button class="btn-row-del" onclick="this.closest('tr').remove()" title="삭제">&times;</button>
            </td>
        `;
        tbody.appendChild(tr);
    }

    timelineAddRow() {
        const tbody = document.getElementById('lyrics-timeline-tbody');
        const rows = tbody.querySelectorAll('tr');
        // 마지막 행의 시간 + 3초를 기본값으로
        let lastT = 0;
        if (rows.length > 0) {
            const lastInput = rows[rows.length - 1].querySelector('.time-input');
            if (lastInput) {
                lastT = this._parseTimeInput(lastInput.value) + 3;
            }
        }
        this._appendTimelineRow(tbody, lastT, '', '', rows.length);
        // 새 행으로 스크롤
        const body = document.querySelector('.lyrics-timeline-body');
        if (body) body.scrollTop = body.scrollHeight;
    }

    timelineReset() {
        if (!this._timelineOriginalData) return;
        if (!confirm('편집 내용을 초기화하시겠습니까?')) return;
        this._renderTimelineTable(this._timelineOriginalData);
    }

    async timelineSave() {
        const projectId = this.mvProjectId || this._currentMVResultProjectId;
        if (!projectId) return;

        const tbody = document.getElementById('lyrics-timeline-tbody');
        const rows = tbody.querySelectorAll('tr');
        const entries = [];

        rows.forEach(tr => {
            const timeInput = tr.querySelector('.time-input');
            const lyricsInput = tr.querySelector('.lyrics-input');
            if (!timeInput || !lyricsInput) return;

            const text = lyricsInput.value.trim();
            if (!text) return; // 빈 텍스트 행 스킵

            const t = this._parseTimeInput(timeInput.value);
            entries.push({ t, text });
        });

        if (entries.length === 0) {
            this.showToast('저장할 가사가 없습니다', 'error');
            return;
        }

        // 시간순 정렬
        entries.sort((a, b) => a.t - b.t);

        const saveBtn = document.getElementById('btn-tl-save');
        if (saveBtn) { saveBtn.disabled = true; saveBtn.textContent = '저장 중...'; }

        try {
            const baseUrl = this.getApiBaseUrl();
            const resp = await fetch(`${baseUrl}/api/mv/${projectId}/lyrics-timeline`, {
                method: 'PUT',
                headers: this.getAuthHeaders(),
                body: JSON.stringify({ timed_lyrics: entries })
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Save failed');
            }

            this.showToast(`가사 타이밍 저장 완료 (${entries.length}줄). 재합성하면 반영됩니다.`, 'success');
            this.closeLyricsTimeline();

            // 재합성 버튼 표시
            const recomposeBtn = document.getElementById('mv-editor-recompose-btn');
            if (recomposeBtn) {
                recomposeBtn.style.display = 'inline-flex';
            }
        } catch (err) {
            console.error('timelineSave error:', err);
            this.showToast(`저장 실패: ${err.message}`, 'error');
        } finally {
            if (saveBtn) { saveBtn.disabled = false; saveBtn.textContent = '저장'; }
        }
    }

    _formatTimeMS(sec) {
        const s = Math.max(0, sec);
        const m = Math.floor(s / 60);
        const sRem = (s % 60).toFixed(1);
        return `${m}:${sRem.padStart(4, '0')}`;
    }

    _parseTimeInput(str) {
        // "M:SS.s" 또는 "SS.s" 형식 파싱
        str = (str || '0').trim();
        const parts = str.split(':');
        if (parts.length === 2) {
            return (parseFloat(parts[0]) || 0) * 60 + (parseFloat(parts[1]) || 0);
        }
        return parseFloat(str) || 0;
    }

    resetMVUI() {
        this.mvProjectId = null;
        this.mvAnalysis = null;
        this.mvRequestParams = null;
        this._currentMVResultProjectId = null;
        this.stopMVPolling();

        // 폼 리셋 (파일, 가사, 컨셉, 셀렉트 등 전부 초기화)
        const mvForm = document.getElementById('mv-form');
        if (mvForm) mvForm.reset();

        // 파일명 표시 초기화
        const fileLabel = document.getElementById('mv-file-name');
        if (fileLabel) fileLabel.textContent = '';

        // 분석 결과 섹션 초기화
        const sceneEditor = document.getElementById('mv-scene-editor');
        if (sceneEditor) sceneEditor.innerHTML = '';
        ['mv-duration', 'mv-bpm', 'mv-suggested-scenes', 'mv-detected-mood'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.textContent = '-';
        });

        // 이미지 그리드 초기화 (진행 중 미리보기 + 리뷰 에디터)
        const grid = document.getElementById('mv-image-review-grid');
        if (grid) grid.innerHTML = '';
        const sceneGrid = document.getElementById('mv-scene-grid');
        if (sceneGrid) sceneGrid.innerHTML = '';

        // 합성 버튼 초기화
        const composeBtn = document.getElementById('mv-compose-btn');
        if (composeBtn) {
            composeBtn.disabled = false;
            composeBtn.innerHTML = '<span class="btn-icon">🎬</span> 최종 뮤직비디오 생성';
        }

        // 진행 로그 초기화
        const logContent = document.getElementById('mv-log-content');
        if (logContent) logContent.innerHTML = '';

        // 진행률 바 초기화
        const progressBar = document.getElementById('mv-progress-bar');
        if (progressBar) progressBar.style.width = '0%';
        const progressText = document.getElementById('mv-progress-text');
        if (progressText) progressText.textContent = '';

        this.showSection('mv');
        this.setNavActive('nav-mv');
    }

    // ==================== 게시판 (커뮤니티) ====================

    async loadBoard(category, page) {
        try {
            const cat = category !== undefined ? category : (this._boardCategory || '');
            const pg = page || 1;
            this._boardCategory = cat;
            this._boardPage = pg;

            let urlToUse = this.getApiBaseUrl();
            let apiUrl = `${urlToUse}/api/board/posts?page=${pg}&limit=20`;
            if (cat) apiUrl += `&category=${cat}`;

            const response = await fetch(apiUrl, { headers: this.getAuthHeaders() });
            if (!response.ok) throw new Error('게시판 로드 실패');
            const data = await response.json();

            this._boardPosts = data.posts || [];
            this._boardPagination = data.pagination || {};

            // 필터 탭 이벤트 (1회만)
            if (!this._boardFilterBound) {
                document.querySelectorAll('.board-filter-btn').forEach(btn => {
                    btn.addEventListener('click', () => {
                        document.querySelectorAll('.board-filter-btn').forEach(b => b.classList.remove('active'));
                        btn.classList.add('active');
                        this.loadBoard(btn.dataset.category, 1);
                    });
                });
                this._boardFilterBound = true;
            }

            this._renderBoardGrid();
            this._renderBoardPagination();
        } catch (error) {
            console.error('게시판 로드 실패:', error);
            document.getElementById('board-grid').innerHTML = '<p style="text-align:center;color:#f66;padding:20px;">게시판 로드 실패</p>';
        }
    }

    _renderBoardGrid() {
        const grid = document.getElementById('board-grid');
        grid.innerHTML = '';

        const posts = this._boardPosts || [];
        if (posts.length === 0) {
            grid.innerHTML = '<p style="text-align:center;color:var(--text-muted);padding:40px;">아직 게시글이 없습니다.</p>';
            return;
        }

        const categoryLabels = {
            feedback: '피드백', bug: '버그', feature: '기능 요청',
            question: '질문', tip: '팁', general: '일반'
        };

        posts.forEach(post => {
            const row = document.createElement('div');
            row.className = 'board-row';

            const catLabel = categoryLabels[post.category] || '일반';
            const badgeClass = `badge-${post.category || 'general'}`;
            const pinIcon = post.is_pinned ? '<span class="pinned-icon">📌</span>' : '';
            const timeAgo = this._timeAgo(post.created_at);

            row.innerHTML = `
                <span class="category-badge ${badgeClass}">${catLabel}</span>
                <span class="post-title">${pinIcon}${escapeHtml(post.title)}</span>
                <span class="post-meta">${escapeHtml(post.author_name || '익명')}</span>
                <span class="post-stats">
                    <span>👁 ${post.view_count || 0}</span>
                    <span>❤ ${post.like_count || 0}</span>
                    <span>💬 ${post.comment_count || 0}</span>
                </span>
                <span class="post-meta">${timeAgo}</span>
            `;

            row.addEventListener('click', () => this.openPost(post.id));
            grid.appendChild(row);
        });
    }

    _renderBoardPagination() {
        const container = document.getElementById('board-pagination');
        container.innerHTML = '';
        const pg = this._boardPagination;
        if (!pg || pg.pages <= 1) return;

        for (let i = 1; i <= pg.pages; i++) {
            const btn = document.createElement('button');
            btn.textContent = i;
            if (i === pg.page) btn.classList.add('active');
            btn.addEventListener('click', () => this.loadBoard(this._boardCategory, i));
            container.appendChild(btn);
        }
    }

    async openPost(postId) {
        try {
            const modal = document.getElementById('post-modal');
            const content = document.getElementById('post-modal-content');
            content.innerHTML = '<p style="text-align:center;padding:40px;color:var(--text-muted);">로딩 중...</p>';
            modal.style.display = 'flex';

            let urlToUse = this.getApiBaseUrl();
            const authHeaders = this.getAuthHeaders();

            const [postRes, commentsRes] = await Promise.all([
                fetch(`${urlToUse}/api/board/posts/${postId}`, { headers: authHeaders }),
                fetch(`${urlToUse}/api/board/posts/${postId}/comments`, { headers: authHeaders }),
            ]);

            if (!postRes.ok) throw new Error('게시글 로드 실패');
            const post = await postRes.json();
            const commentsData = commentsRes.ok ? await commentsRes.json() : { comments: [] };

            const categoryLabels = {
                feedback: '피드백', bug: '버그', feature: '기능 요청',
                question: '질문', tip: '팁', general: '일반'
            };
            const badgeClass = `badge-${post.category || 'general'}`;
            const catLabel = categoryLabels[post.category] || '일반';
            const likedClass = post.liked ? 'liked' : '';
            const currentUser = JSON.parse(localStorage.getItem('user') || 'null');
            const isOwner = currentUser && post.user_id === currentUser.id;

            let ownerActions = '';
            if (isOwner) {
                ownerActions = `
                    <button onclick="app.editPost(${post.id})">수정</button>
                    <button onclick="app.deletePost(${post.id})">삭제</button>
                `;
            }

            const commentsHtml = (commentsData.comments || []).map(c => `
                <div class="comment-item">
                    <span class="comment-author">${escapeHtml(c.author_name || '익명')}</span>
                    <span class="comment-date">${this._timeAgo(c.created_at)}</span>
                    <div class="comment-text">${escapeHtml(c.content)}</div>
                </div>
            `).join('');

            const hasToken = !!localStorage.getItem('token');
            const commentFormHtml = hasToken ? `
                <div class="comment-form">
                    <textarea id="comment-input" placeholder="댓글을 입력하세요..." maxlength="1000"></textarea>
                    <button onclick="app.addComment(${post.id})">등록</button>
                </div>
            ` : '<p style="font-size:0.82rem;color:var(--text-muted);">댓글을 작성하려면 로그인하세요.</p>';

            content.innerHTML = `
                <div class="post-detail-header">
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                        <span class="category-badge ${badgeClass}">${catLabel}</span>
                    </div>
                    <h2>${escapeHtml(post.title)}</h2>
                    <div class="post-detail-meta">
                        <span>${escapeHtml(post.author_name || '익명')}</span>
                        <span>${new Date(post.created_at).toLocaleDateString('ko-KR')} ${new Date(post.created_at).toLocaleTimeString('ko-KR', {hour:'2-digit',minute:'2-digit'})}</span>
                        <span>조회 ${post.view_count || 0}</span>
                    </div>
                </div>
                <div class="post-detail-body">${escapeHtml(post.content)}</div>
                <div class="post-actions">
                    <button class="${likedClass}" onclick="app.toggleLike(${post.id}, this)">❤ 좋아요 ${post.like_count || 0}</button>
                    ${ownerActions}
                </div>
                <div class="comments-section">
                    <h4>💬 댓글 ${commentsData.comments?.length || 0}</h4>
                    ${commentsHtml}
                    ${commentFormHtml}
                </div>
            `;
        } catch (error) {
            console.error('게시글 열기 실패:', error);
            document.getElementById('post-modal-content').innerHTML =
                '<p style="text-align:center;color:#f66;padding:20px;">게시글을 불러올 수 없습니다.</p>';
        }
    }

    openWritePostModal(editData) {
        const token = localStorage.getItem('token');
        if (!token) {
            this.showToast('로그인이 필요합니다.', 'error');
            return;
        }

        const modal = document.getElementById('post-write-modal');
        const titleInput = document.getElementById('post-title-input');
        const contentInput = document.getElementById('post-content-input');
        const categorySelect = document.getElementById('post-category-select');
        const submitBtn = document.getElementById('post-submit-btn');

        if (editData) {
            titleInput.value = editData.title || '';
            contentInput.value = editData.content || '';
            categorySelect.value = editData.category || 'general';
            submitBtn.textContent = '수정';
            submitBtn.dataset.editId = editData.id;
        } else {
            titleInput.value = '';
            contentInput.value = '';
            categorySelect.value = 'general';
            submitBtn.textContent = '등록';
            delete submitBtn.dataset.editId;
        }

        modal.style.display = 'flex';
    }

    async submitPost() {
        const titleInput = document.getElementById('post-title-input');
        const contentInput = document.getElementById('post-content-input');
        const categorySelect = document.getElementById('post-category-select');
        const submitBtn = document.getElementById('post-submit-btn');

        const title = titleInput.value.trim();
        const content = contentInput.value.trim();
        const category = categorySelect.value;

        if (!title) { this.showToast('제목을 입력하세요.', 'error'); return; }
        if (!content) { this.showToast('내용을 입력하세요.', 'error'); return; }

        submitBtn.disabled = true;
        submitBtn.textContent = '처리 중...';

        try {
            let urlToUse = this.getApiBaseUrl();
            const editId = submitBtn.dataset.editId;
            const method = editId ? 'PUT' : 'POST';
            const apiUrl = editId
                ? `${urlToUse}/api/board/posts/${editId}`
                : `${urlToUse}/api/board/posts`;

            const response = await fetch(apiUrl, {
                method,
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`,
                },
                body: JSON.stringify({ title, content, category }),
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || '요청 실패');
            }

            document.getElementById('post-write-modal').style.display = 'none';
            this.showToast(editId ? '수정되었습니다.' : '등록되었습니다.', 'success');
            this.loadBoard(this._boardCategory, this._boardPage);

            if (editId) {
                // 수정 후 상세 모달도 새로고침
                this.openPost(parseInt(editId));
            }
        } catch (error) {
            this.showToast(error.message, 'error');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = submitBtn.dataset.editId ? '수정' : '등록';
        }
    }

    async editPost(postId) {
        try {
            let urlToUse = this.getApiBaseUrl();
            const response = await fetch(`${urlToUse}/api/board/posts/${postId}`, {
                headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` },
            });
            if (!response.ok) throw new Error();
            const post = await response.json();

            document.getElementById('post-modal').style.display = 'none';
            this.openWritePostModal({ id: post.id, title: post.title, content: post.content, category: post.category });
        } catch {
            this.showToast('게시글을 불러올 수 없습니다.', 'error');
        }
    }

    async deletePost(postId) {
        if (!confirm('정말 삭제하시겠습니까?')) return;

        try {
            let urlToUse = this.getApiBaseUrl();
            const response = await fetch(`${urlToUse}/api/board/posts/${postId}`, {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` },
            });
            if (!response.ok) throw new Error();

            document.getElementById('post-modal').style.display = 'none';
            this.showToast('삭제되었습니다.', 'success');
            this.loadBoard(this._boardCategory, this._boardPage);
        } catch {
            this.showToast('삭제에 실패했습니다.', 'error');
        }
    }

    async toggleLike(postId, btn) {
        const token = localStorage.getItem('token');
        if (!token) { this.showToast('로그인이 필요합니다.', 'error'); return; }

        try {
            let urlToUse = this.getApiBaseUrl();
            const response = await fetch(`${urlToUse}/api/board/posts/${postId}/like`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` },
            });
            if (!response.ok) throw new Error();
            const data = await response.json();

            // 버튼 UI 업데이트
            if (data.liked) {
                btn.classList.add('liked');
            } else {
                btn.classList.remove('liked');
            }

            // 좋아요 수 갱신 — 상세 모달 다시 로드
            this.openPost(postId);
        } catch {
            this.showToast('좋아요 처리에 실패했습니다.', 'error');
        }
    }

    async addComment(postId) {
        const input = document.getElementById('comment-input');
        const content = input?.value.trim();
        if (!content) { this.showToast('댓글을 입력하세요.', 'error'); return; }

        try {
            let urlToUse = this.getApiBaseUrl();
            const response = await fetch(`${urlToUse}/api/board/posts/${postId}/comments`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`,
                },
                body: JSON.stringify({ content }),
            });
            if (!response.ok) throw new Error();

            this.openPost(postId); // 댓글 목록 새로고침
        } catch {
            this.showToast('댓글 작성에 실패했습니다.', 'error');
        }
    }

    _timeAgo(dateStr) {
        if (!dateStr) return '';
        const now = new Date();
        const date = new Date(dateStr);
        const diffMs = now - date;
        const diffMin = Math.floor(diffMs / 60000);
        if (diffMin < 1) return '방금';
        if (diffMin < 60) return `${diffMin}분 전`;
        const diffHour = Math.floor(diffMin / 60);
        if (diffHour < 24) return `${diffHour}시간 전`;
        const diffDay = Math.floor(diffHour / 24);
        if (diffDay < 30) return `${diffDay}일 전`;
        return date.toLocaleDateString('ko-KR');
    }
}

// 앱 초기화
const app = new StorycutApp();

// 글로벌 showToast 헬퍼 (auth.js 등 클래스 외부에서 사용)
window.showToast = (msg, type) => app.showToast(msg, type);

// MV 이벤트 리스너 초기화
app.initMVEventListeners();
