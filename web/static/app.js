// STORYCUT v2.0 - 프론트엔드 로직

class StorycutApp {
    constructor() {
        this.projectId = null;
        this.websocket = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.updateDurationDisplay();
    }

    setupEventListeners() {
        // 폼 제출
        const form = document.getElementById('generate-form');
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.startGeneration();
        });

        // 슬라이더
        const durationSlider = document.getElementById('duration');
        durationSlider.addEventListener('input', () => {
            this.updateDurationDisplay();
        });

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
    }

    updateDurationDisplay() {
        const duration = document.getElementById('duration').value;
        document.getElementById('duration-display').textContent = duration;
    }

    // [Config] Cloudflare Worker URL (Middleware)
    // 이 주소가 '카운터' 역할을 하는 곳입니다.
    getApiBaseUrl() {
        // 로컬 개발 환경용 (localhost)
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return ''; // 상대 경로 사용
        }
        // 배포 환경용 (Cloudflare Worker 주소)
        return 'https://storycut-worker.twinspa0713.workers.dev';
    }

    async startGeneration() {
        // 폼 데이터 수집
        const formData = new FormData(document.getElementById('generate-form'));

        // ... (중략) ... 

        const requestData = {
            topic: formData.get('topic') || null,
            genre: formData.get('genre'),
            mood: formData.get('mood'),
            style: formData.get('style'),
            duration: parseInt(formData.get('duration')),
            platform: formData.get('platform'),

            // Feature Flags
            hook_scene1_video: document.getElementById('hook_scene1_video').checked,
            ffmpeg_kenburns: document.getElementById('ffmpeg_kenburns').checked,
            ffmpeg_audio_ducking: document.getElementById('ffmpeg_audio_ducking').checked,
            subtitle_burn_in: document.getElementById('subtitle_burn_in').checked,
            context_carry_over: document.getElementById('context_carry_over').checked,
            optimization_pack: document.getElementById('optimization_pack').checked,
        };

        try {
            // UI 전환
            this.showSection('progress');

            // API 호출
            const baseUrl = this.getApiBaseUrl();
            const token = localStorage.getItem('token');

            const headers = {
                'Content-Type': 'application/json',
            };

            if (token) {
                headers['Authorization'] = `Bearer ${token}`;
            }

            const response = await fetch(`${baseUrl}/api/generate`, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error(`API 오류: ${response.status}`);
            }

            const result = await response.json();
            this.projectId = result.project_id;

            this.addLog('INFO', `프로젝트 시작: ${this.projectId}`);

            // WebSocket 연결
            this.connectWebSocket(this.projectId);

        } catch (error) {
            console.error('생성 시작 실패:', error);
            alert(`영상 생성을 시작할 수 없습니다: ${error.message}`);
            this.showSection('input');
        }
    }

    connectWebSocket(projectId) {
        const baseUrl = this.getApiBaseUrl();
        let wsUrl;

        if (baseUrl === '') {
            // 로컬 환경
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            wsUrl = `${protocol}//${window.location.host}/ws/${projectId}`;
        } else {
            // 배포 환경 (https -> wss 로 변경)
            wsUrl = baseUrl.replace('https://', 'wss://').replace('http://', 'ws://') + `/ws/${projectId}`;
        }

        this.websocket = new WebSocket(wsUrl);

        this.websocket.onopen = () => {
            console.log('WebSocket 연결됨');
            this.addLog('INFO', 'WebSocket 연결 성공');

            // 핑 전송 (연결 유지)
            setInterval(() => {
                if (this.websocket.readyState === WebSocket.OPEN) {
                    this.websocket.send('ping');
                }
            }, 30000);
        };

        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleProgress(data);
        };

        this.websocket.onerror = (error) => {
            console.error('WebSocket 오류:', error);
            this.addLog('ERROR', 'WebSocket 연결 오류');
        };

        this.websocket.onclose = () => {
            console.log('WebSocket 연결 종료');
            this.addLog('INFO', 'WebSocket 연결 종료');
        };
    }

    handleProgress(data) {
        if (data.type === 'pong') return;

        if (data.type === 'progress') {
            // 진행률 업데이트
            this.updateProgress(data.progress, data.message);

            // 단계별 상태 업데이트
            this.updateStepStatus(data.step, data.message);

            // 로그 추가
            this.addLog('PROGRESS', data.message);

            // 완료 처리
            if (data.step === 'complete') {
                this.handleComplete(data.data);
            }

            // [NEW] 스토리 프리뷰
            if (data.step === 'story' && data.data && data.data.story_data) {
                this.renderStoryScript(data.data.story_data);
            }

            // [NEW] 씬 이미지 프리뷰
            if (data.step.startsWith('scene_') && data.message.includes('완료') && data.data && data.data.image_url) {
                // data.step format: "scene_1" -> extract ID
                const sceneId = data.data.scene_id;
                this.renderSceneImage(sceneId, data.data.image_url, `Scene ${sceneId}`);
            }
        }
    }

    updateProgress(progress, message) {
        const progressBar = document.getElementById('progress-bar');
        const progressPercentage = document.getElementById('progress-percentage');
        const statusMessage = document.getElementById('status-message');

        progressBar.style.width = `${progress}%`;
        progressPercentage.textContent = `${progress}%`;
        statusMessage.textContent = message;
    }

    updateStepStatus(step, message) {
        // 모든 단계 초기화
        document.querySelectorAll('.step').forEach(el => {
            el.classList.remove('active', 'completed');
        });

        // 현재 단계 매핑
        let currentStepElement = null;

        if (step === 'story') {
            currentStepElement = document.querySelector('[data-step="story"]');
        } else if (step.startsWith('scene_')) {
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
        logEntry.className = 'log-entry';

        const logLevel = level === 'ERROR' ? '❌' : level === 'INFO' ? 'ℹ️' : '▶️';

        logEntry.innerHTML = `
            <span class="log-timestamp">[${timestamp}]</span>
            <span class="log-level">${logLevel}</span>
            <span class="log-message">${message}</span>
        `;

        logContent.appendChild(logEntry);

        // 스크롤을 최신 로그로
        logContent.scrollTop = logContent.scrollHeight;
    }

    async handleComplete(data) {
        console.log('생성 완료:', data);

        // WebSocket 종료
        if (this.websocket) {
            this.websocket.close();
        }

        // 결과 섹션으로 전환
        setTimeout(() => {
            this.showResults(data);
        }, 1000);
    }

    async showResults(data) {
        // 결과 정보 표시
        document.getElementById('result-project-id').textContent = data.project_id;
        document.getElementById('result-title').textContent = data.title_candidates ? data.title_candidates[0] : '제목 없음';

        // 비디오 플레이어
        const video = document.getElementById('result-video');
        video.src = `/api/download/${data.project_id}`;

        // 다운로드 버튼
        const downloadBtn = document.getElementById('download-btn');
        downloadBtn.href = `/api/download/${data.project_id}`;

        // 최적화 패키지
        if (data.title_candidates && data.title_candidates.length > 0) {
            this.displayTitleCandidates(data.title_candidates);
        }

        if (data.thumbnail_texts && data.thumbnail_texts.length > 0) {
            this.displayThumbnailTexts(data.thumbnail_texts);
        }

        if (data.hashtags && data.hashtags.length > 0) {
            this.displayHashtags(data.hashtags);
        }

        // UI 전환
        this.showSection('result');
    }

    displayTitleCandidates(titles) {
        const container = document.getElementById('title-candidates');
        container.innerHTML = '';

        titles.forEach((title, index) => {
            const item = document.createElement('div');
            item.className = 'candidate-item';
            item.textContent = `${index + 1}. ${title}`;
            item.onclick = () => {
                this.copyToClipboard(title);
                alert('제목이 클립보드에 복사되었습니다!');
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
            item.onclick = () => {
                this.copyToClipboard(text);
                alert('썸네일 문구가 클립보드에 복사되었습니다!');
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
            item.onclick = () => {
                this.copyToClipboard(tag);
                alert('해시태그가 클립보드에 복사되었습니다!');
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
        // 모든 섹션 숨기기
        document.getElementById('input-section').classList.add('hidden');
        document.getElementById('progress-section').classList.add('hidden');
        document.getElementById('result-section').classList.add('hidden');

        // 선택한 섹션 표시
        if (sectionName === 'input') {
            document.getElementById('input-section').classList.remove('hidden');
        } else if (sectionName === 'progress') {
            document.getElementById('progress-section').classList.remove('hidden');
        } else if (sectionName === 'result') {
            document.getElementById('result-section').classList.remove('hidden');
        }
    }

    renderStoryScript(storyData) {
        const previewContainer = document.getElementById('preview-container');
        const storyPreview = document.getElementById('story-preview');
        const scriptContent = document.getElementById('script-content');

        previewContainer.classList.remove('hidden');
        storyPreview.classList.remove('hidden');

        let html = `<strong>[Title] ${storyData.title}</strong>\n\n`;
        storyData.scenes.forEach(scene => {
            html += `<strong>[Scene ${scene.scene_id}]</strong> ${scene.narration || scene.sentence}\n`;
            html += `<em>(Visual: ${scene.visual_description})</em>\n\n`;
        });

        scriptContent.innerHTML = html;
        scriptContent.scrollTop = 0;
    }

    renderSceneImage(sceneId, imageUrl, title) {
        const previewContainer = document.getElementById('preview-container');
        const visualPreview = document.getElementById('visual-preview');
        const sceneGrid = document.getElementById('scene-grid');

        previewContainer.classList.remove('hidden');
        visualPreview.classList.remove('hidden');

        // 중복 방지 (카드 이미 있으면 업데이트)
        let card = document.getElementById(`scene-card-${sceneId}`);

        if (!card) {
            card = document.createElement('div');
            card.id = `scene-card-${sceneId}`;
            card.className = 'scene-card';
            sceneGrid.appendChild(card);
        }

        card.innerHTML = `
            <img src="${imageUrl}" alt="${title}" loading="lazy">
            <div class="scene-info">${title}</div>
        `;
    }

    resetUI() {
        // 폼 초기화
        document.getElementById('generate-form').reset();
        this.updateDurationDisplay();

        // 진행률 초기화
        document.getElementById('progress-bar').style.width = '0%';
        document.getElementById('progress-percentage').textContent = '0%';
        document.getElementById('status-message').textContent = '초기화 중...';
        document.getElementById('log-content').innerHTML = '';

        // [NEW] 프리뷰 초기화
        document.getElementById('preview-container').classList.add('hidden');
        document.getElementById('story-preview').classList.add('hidden');
        document.getElementById('visual-preview').classList.add('hidden');
        document.getElementById('script-content').innerHTML = '';
        document.getElementById('scene-grid').innerHTML = '';

        // 단계 초기화
        document.querySelectorAll('.step').forEach(el => {
            el.classList.remove('active', 'completed');
            el.querySelector('.step-status').textContent = '대기 중';
        });

        // 입력 섹션으로 돌아가기
        this.showSection('input');

        // WebSocket 종료
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }

        this.projectId = null;
    }
}

// 앱 초기화
document.addEventListener('DOMContentLoaded', () => {
    new StorycutApp();
});
