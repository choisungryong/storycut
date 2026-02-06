// STORYCUT v2.0 - 프론트엔드 로직 (완전 재작성)

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

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.updateDurationDisplay();
    }

    setupEventListeners() {
        // 1단계: 스토리 생성 (폼 제출)
        const form = document.getElementById('generate-form');
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.startStoryGeneration();
        });

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

        // 이미지만 먼저 생성 버튼
        const generateImagesBtn = document.getElementById('generate-images-btn');
        if (generateImagesBtn) {
            generateImagesBtn.addEventListener('click', () => {
                this.startImageGeneration();
            });
        }
    }

    setNavActive(navId) {
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        document.getElementById(navId).classList.add('active');
    }

    updateDurationDisplay() {
        const duration = document.getElementById('duration').value;
        document.getElementById('duration-display').textContent = duration;
    }

    getApiBaseUrl() {
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return '';
        }
        // Railway 백엔드 (영상 생성, 상태 조회 등)
        return 'https://web-production-bb6bf.up.railway.app';
    }

    getWorkerUrl() {
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return '';
        }
        // Cloudflare Worker (스토리 생성, 인증)
        return 'https://storycut-worker.twinspa0713.workers.dev';
    }

    // ==================== Step 1: 스토리 생성 ====================
    async startStoryGeneration() {
        const formData = new FormData(document.getElementById('generate-form'));

        const btn = document.getElementById('generate-story-btn');
        const originalBtnText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = '<span class="btn-icon">⏳</span> 스토리 생성 중...';

        const requestData = {
            topic: formData.get('topic') || null,
            genre: formData.get('genre'),
            mood: formData.get('mood'),
            style: formData.get('style'),
            voice: formData.get('voice'),
            duration: parseInt(formData.get('duration')),
            platform: formData.get('platform'),

            // Feature Flags (with null checks)
            hook_scene1_video: document.getElementById('hook_scene1_video')?.checked || false,
            ffmpeg_kenburns: document.getElementById('ffmpeg_kenburns')?.checked || true,
            ffmpeg_audio_ducking: document.getElementById('ffmpeg_audio_ducking')?.checked || false,
            subtitle_burn_in: document.getElementById('subtitle_burn_in')?.checked || true,
            context_carry_over: document.getElementById('context_carry_over')?.checked || true,
            optimization_pack: document.getElementById('optimization_pack')?.checked || true,
        };

        this.currentRequestParams = requestData;

        try {
            // 즉시 progress 화면으로 전환
            btn.disabled = false;
            btn.innerHTML = originalBtnText;
            this.showSection('progress');
            this.updateStepStatus('story', '스토리 생성 중...');
            document.getElementById('status-message').textContent = 'AI가 스토리를 구상하고 있습니다...';
            document.getElementById('progress-percentage').textContent = '5%';
            document.getElementById('progress-bar').style.width = '5%';

            // 가짜 진행률 애니메이션 (체감 속도 개선)
            const storyMessages = [
                { pct: 8,  msg: '장르와 분위기를 분석하고 있습니다...' },
                { pct: 15, msg: '캐릭터와 세계관을 설계하고 있습니다...' },
                { pct: 22, msg: '스토리 구조를 잡고 있습니다...' },
                { pct: 30, msg: '기승전결 아크를 설계하고 있습니다...' },
                { pct: 38, msg: '장면별 내러티브를 작성하고 있습니다...' },
                { pct: 45, msg: '대사와 나레이션을 다듬고 있습니다...' },
                { pct: 52, msg: '비주얼 프롬프트를 생성하고 있습니다...' },
                { pct: 58, msg: '카메라 워크를 설정하고 있습니다...' },
                { pct: 64, msg: '스토리 일관성을 검증하고 있습니다...' },
                { pct: 70, msg: '유튜브 최적화 데이터를 생성하고 있습니다...' },
                { pct: 75, msg: '최종 스토리를 정리하고 있습니다...' },
                { pct: 78, msg: '거의 완료되었습니다...' },
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
            let response;

            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 120000); // 2분 타임아웃
                response = await fetch(`${workerUrl}/api/generate/story`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestData),
                    signal: controller.signal
                });
                clearTimeout(timeoutId);
            } catch (workerError) {
                console.warn('[Story] Worker 실패, Railway 폴백:', workerError.message);
                this.updateProgress(40, 'Worker 타임아웃 — 백엔드로 재시도 중...');
                response = await fetch(`${railwayUrl}/api/generate/story`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestData)
                });
            }

            clearInterval(progressInterval);

            if (!response.ok) {
                let errorMsg = '스토리 생성 실패';
                try {
                    const error = await response.json();
                    errorMsg = error.detail || error.error || errorMsg;
                } catch (e) {}
                throw new Error(errorMsg);
            }

            // 완료 애니메이션
            this.updateProgress(90, '스토리 생성 완료! 결과를 불러오는 중...');

            const result = await response.json();

            if (result.story_data) {
                this.updateProgress(100, '스토리가 완성되었습니다!');
                this.currentStoryData = result.story_data;
                this.currentRequestParams = requestData;

                // 짧은 딜레이로 100% 표시 후 전환
                await new Promise(r => setTimeout(r, 500));
                this.renderStoryReview(this.currentStoryData);
                this.showSection('review');
                this.setNavActive('nav-create');
            } else {
                throw new Error('잘못된 응답 형식');
            }

        } catch (error) {
            console.error('스토리 생성 실패:', error);
            alert(`오류 발생: ${error.message}`);
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

        document.getElementById('review-title').value = storyData.title;

        storyData.scenes.forEach((scene, index) => {
            const card = document.createElement('div');
            card.className = 'review-card';
            card.dataset.sceneId = scene.scene_id; // IMPORTANT: Add scene_id to dataset
            card.innerHTML = `
                <div class="review-card-header">
                    <span>Scene ${scene.scene_id}</span>
                    <span>${scene.duration_sec}초</span>
                </div>

                <label>내레이션 / 대사</label>
                <textarea class="review-textarea narration-input" data-idx="${index}">${scene.narration || scene.sentence}</textarea>

                <label>화면 묘사 (Prompt)</label>
                <textarea class="review-textarea visual-textarea visual-input" data-idx="${index}">${scene.visual_description || scene.prompt}</textarea>
            `;
            grid.appendChild(card);
        });
    }

    // ==================== Step 2: 영상 생성 시작 ====================
    async startFinalGeneration() {
        if (!this.currentStoryData) return;

        // 이미 생성 중이면 중복 생성 방지
        if (this.isGenerating) {
            alert('이미 영상 생성이 진행 중입니다.');
            return;
        }

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
        });

        // 생성 시작
        try {
            this.isGenerating = true;
            this.showSection('progress');
            const progressTitle = document.getElementById('progress-title');
            if (progressTitle) progressTitle.textContent = '⏳ 영상 생성 중...';

            let urlToUse = this.getApiBaseUrl();

            // 인증 토큰 (선택적 - 백엔드가 인증 없이도 작동)
            const token = localStorage.getItem('token');
            const headers = { 'Content-Type': 'application/json' };
            if (token) {
                headers['Authorization'] = `Bearer ${token}`;
            }

            const payload = {
                request_params: this.currentRequestParams,
                story_data: this.currentStoryData
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
            alert(`영상 생성 실패: ${error.message}`);
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
                    this.addLog('SUCCESS', '🎉 영상 생성 완료!');
                    this.updateProgress(100, '완료');
                    this.updateStepStatus('complete', '완료');
                    this.stopPolling();
                    this.isGenerating = false;

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
                    alert(`영상 생성 실패: ${data.error_message}`);

                } else if (data.status === 'processing') {
                    // 진행 중 상태 업데이트
                    const progress = data.progress || 25;
                    const message = data.message || '영상 생성 중...';

                    this.updateProgress(progress, message);

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
                    alert('서버와의 연결이 끊어졌습니다.\n서버가 재시작 중일 수 있습니다.\n잠시 후 페이지를 새로고침 해주세요.');
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
                            this.addLog('SUCCESS', '🎉 영상 생성 완료!');
                            this.updateStepStatus('complete', '완료');
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
                console.log(`[Fetch] Requesting manifest from: ${targetUrl}`);

                const response = await fetch(targetUrl);

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
                    server_url: urlToUse
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
        document.getElementById('result-video-container').innerHTML = `<div class="error-box"><p>${message}</p></div>`;
    }

    async showResults(data) {
        // 결과 섹션 표시
        this.showSection('result');
        this.setNavActive('nav-create');

        // 헤더 텍스트 업데이트
        const headerText = document.getElementById('result-header-text');
        const videoContainer = document.getElementById('result-video-container');
        const downloadBtn = document.getElementById('download-btn');

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
            headerText.textContent = "🎉 영상 생성 완료!";

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
                    <p>${data.error_message || '알 수 없는 오류'}</p>
                </div>`;
            downloadBtn.style.display = 'none';
        }

        // 최적화 패키지 (아래는 공통)
        if (data.title_candidates?.length > 0) this.displayTitleCandidates(data.title_candidates);
        if (data.thumbnail_texts?.length > 0) this.displayThumbnailTexts(data.thumbnail_texts);
        if (data.hashtags?.length > 0) this.displayHashtags(data.hashtags);

        // 씬 목록 로드
        await this.loadSceneList(data.project_id);

        // UI 전환
        this.showSection('result');
        this.setNavActive('nav-create');

        this.addLog('SUCCESS', '✅ 모든 정보 로드 완료!');
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
            item.style.cursor = 'pointer';
            item.title = '클릭하여 복사';
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
            item.style.cursor = 'pointer';
            item.title = '클릭하여 복사';
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

        // 모든 섹션 숨기기
        document.getElementById('input-section').classList.add('hidden');
        document.getElementById('progress-section').classList.add('hidden');
        document.getElementById('result-section').classList.add('hidden');
        document.getElementById('review-section').classList.add('hidden');
        document.getElementById('history-section').classList.add('hidden');
        document.getElementById('image-preview-section').classList.add('hidden');
        // MV 섹션들
        document.getElementById('mv-section')?.classList.add('hidden');
        document.getElementById('mv-analysis-section')?.classList.add('hidden');
        document.getElementById('mv-progress-section')?.classList.add('hidden');
        document.getElementById('mv-result-section')?.classList.add('hidden');

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
            case 'mv-result':
                document.getElementById('mv-result-section')?.classList.remove('hidden');
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
                errorMsg = `<div class="scene-error-message">❌ ${scene.error_message}</div>`;
            }

            card.innerHTML = `
                <div class="scene-card-header">
                    <span class="scene-card-title">Scene ${scene.scene_id}</span>
                    ${statusBadge}
                </div>

                <div class="scene-card-narration">
                    ${scene.narration || '내레이션 없음'}
                </div>

                <div class="scene-card-visual">
                    📸 ${scene.generation_method || 'unknown'}
                </div>

                ${errorMsg}

                <div class="scene-card-actions">
                    <button class="btn-regenerate" data-scene-id="${scene.scene_id}" data-project-id="${projectId}"
                        ${scene.status === 'regenerating' ? 'disabled' : ''}>
                        🔄 재생성
                    </button>
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
    }

    async regenerateScene(projectId, sceneId) {
        const card = document.querySelector(`[data-scene-id="${sceneId}"]`);
        const btn = card.querySelector('.btn-regenerate');

        try {
            // UI 업데이트
            btn.disabled = true;
            btn.textContent = '⏳ 재생성 중...';
            card.classList.add('regenerating');

            this.addLog('INFO', `Scene ${sceneId} 재생성 시작...`);

            const baseUrl = this.getApiBaseUrl();
            const response = await fetch(`${baseUrl}/api/projects/${projectId}/scenes/${sceneId}/regenerate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
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
                        // JSON 파싱 실패 시 텍스트 그대로 사용
                        errorMsg = errorText || errorMsg;
                    }
                } catch (e) {
                    console.error("Error reading response error:", e);
                }
                throw new Error(errorMsg);
            }

            const result = await response.json();
            this.addLog('SUCCESS', `✅ Scene ${sceneId} 재생성 완료!`);

            // 씬 목록 새로고침
            await this.loadSceneList(projectId);

            // 재합성 버튼 표시
            const recomposeBtn = document.getElementById('recompose-btn');
            if (recomposeBtn) {
                recomposeBtn.style.display = 'block';
            }

            alert(`Scene ${sceneId} 재생성 완료!\n\n수정된 씬을 영상에 반영하려면 "영상 재합성" 버튼을 누르세요.`);

        } catch (error) {
            console.error('씬 재생성 실패:', error);
            this.addLog('ERROR', `❌ Scene ${sceneId} 재생성 실패: ${error.message}`);
            alert(`씬 재생성 실패: ${error.message}`);

            // UI 복구
            btn.disabled = false;
            btn.textContent = '🔄 재생성';
            card.classList.remove('regenerating');
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
                headers: { 'Content-Type': 'application/json' }
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

            alert('영상 재합성 완료!\n\n새로운 영상이 플레이어에 반영되었습니다.');

        } catch (error) {
            console.error('영상 재합성 실패:', error);
            this.addLog('ERROR', `❌ 영상 재합성 실패: ${error.message}`);
            alert(`영상 재합성 실패: ${error.message}`);

            btn.disabled = false;
            btn.innerHTML = '<span class="btn-icon">🔄</span> 영상 재합성 (수정된 씬 반영)';
        }
    }

    // ==================== History 기능 ====================
    async loadHistory() {
        try {
            let urlToUse = this.getApiBaseUrl();
            const response = await fetch(`${urlToUse}/api/history`);

            if (!response.ok) throw new Error('History 로드 실패');

            const data = await response.json();
            const historyGrid = document.getElementById('history-grid');
            historyGrid.innerHTML = '';

            if (data.projects.length === 0) {
                historyGrid.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: #888;">생성된 영상이 없습니다.</p>';
                return;
            }

            data.projects.forEach(project => {
                const card = document.createElement('div');
                card.className = 'history-card';
                card.innerHTML = `
                    <div class="history-thumb" style="background: #1a1a2e;">
                        ${project.thumbnail_url ? `<img src="${project.thumbnail_url}" alt="${project.title}">` : '<div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: #555;">📽️</div>'}
                    </div>
                    <div class="history-info">
                        <p class="history-title">${project.title}</p>
                        <p class="history-date">${new Date(project.created_at).toLocaleDateString('ko-KR')}</p>
                        <span class="history-status ${project.status === 'completed' ? 'completed' : ''}">${project.status === 'completed' ? '✅ 완료' : '⏳ 처리 중'}</span>
                    </div>
                `;

                card.style.cursor = 'pointer';
                card.onclick = () => {
                    // status 상관없이 상세 페이지로 이동 (오류 났거나 생성 중이어도 확인 가능하도록)
                    this.fetchAndShowResults(project.project_id);
                };

                historyGrid.appendChild(card);
            });

        } catch (error) {
            console.error('History 로드 실패:', error);
            document.getElementById('history-grid').innerHTML = '<p style="color: #f66;">History 로드 실패</p>';
        }
    }

    // ==================== 이미지 URL 경로 변환 ====================
    resolveImageUrl(imagePath) {
        if (!imagePath) return '';
        if (imagePath.startsWith('http')) return imagePath;
        // outputs/xxx → /media/xxx 변환 (FastAPI StaticFiles 마운트: /media = outputs/)
        if (imagePath.startsWith('outputs/')) {
            return `${this.getApiBaseUrl()}/media/${imagePath.slice('outputs/'.length)}`;
        }
        if (imagePath.startsWith('/')) {
            return `${this.getApiBaseUrl()}${imagePath}`;
        }
        return `${this.getApiBaseUrl()}/media/${imagePath}`;
    }

    // ==================== 이미지 생성 워크플로우 ====================

    async startImageGeneration() {
        if (!this.currentStoryData) {
            alert('스토리 데이터가 없습니다.');
            return;
        }

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
                headers: { 'Content-Type': 'application/json' },
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
                    try { errorDetail = await response.text(); } catch (e2) {}
                }
                throw new Error(`${response.status}: ${errorDetail}`);
            }

            const result = await response.json();
            this.projectId = result.project_id;
            console.log('[Image Generation] Response:', JSON.stringify(result));

            // 즉시 프리뷰 화면으로 전환 (플레이스홀더 표시)
            this.renderImagePreviewPlaceholders(this.currentStoryData.scenes, result.total_scenes);
            this.showSection('image-preview');
            console.log('[Image Generation] Section switched to image-preview');

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
            alert(`이미지 생성 실패: ${error.message}`);
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
                        <button class="btn-image-action btn-hook" disabled>☆ Hook</button>
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

        this.imagePollingInterval = setInterval(async () => {
            try {
                const response = await fetch(`${apiUrl}/api/status/images/${projectId}`);
                if (!response.ok) {
                    console.warn(`[Image Polling] HTTP ${response.status}`);
                    return;
                }

                const data = await response.json();
                const { completed, total, scenes, status, error_message } = data;
                console.log(`[Image Polling] status=${status}, completed=${completed}/${total}, scenes=${scenes?.length || 0}`);

                // 상태별 메시지 표시
                let statusMsg = `Scene ${completed}/${total} 완료`;
                if (status === 'not_found') {
                    statusMsg = '프로젝트 초기화 중...';
                } else if (status === 'preparing') {
                    statusMsg = '스타일/캐릭터 앵커 준비 중...';
                } else if (status === 'generating_images' && completed === 0) {
                    statusMsg = '이미지 생성 시작 중...';
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
                        const hookBtn = card.querySelector('.btn-hook');
                        if (regenBtn && regenBtn.disabled) {
                            regenBtn.disabled = false;
                            regenBtn.setAttribute('onclick', `app.regenerateImage('${projectId}', ${scene.scene_id})`);
                        }
                        if (i2vBtn && i2vBtn.disabled) {
                            i2vBtn.disabled = false;
                            i2vBtn.setAttribute('onclick', `app.convertToVideo('${projectId}', ${scene.scene_id})`);
                        }
                        if (hookBtn && hookBtn.disabled) {
                            hookBtn.disabled = false;
                            hookBtn.setAttribute('onclick', `app.toggleHookVideo('${projectId}', ${scene.scene_id})`);
                        }
                    }
                });

                // 전체 완료 체크 (preparing/generating 단계에서는 완료 아님)
                if (status === 'images_ready' || (status !== 'preparing' && status !== 'generating_images' && status !== 'not_found' && completed === total && total > 0)) {
                    clearInterval(this.imagePollingInterval);
                    this.imagePollingInterval = null;

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

            if (scene.hook_video_enabled) card.classList.add('hook-video');

            const imagePath = scene.assets?.image_path || scene.image_path || '';
            const imageUrl = this.resolveImageUrl(imagePath);

            card.innerHTML = `
                <div class="image-card-header">
                    <span class="image-card-title">Scene ${scene.scene_id}</span>
                    ${scene.hook_video_enabled ? '<span class="hook-badge">🎥 HOOK</span>' : ''}
                </div>
                <img src="${imageUrl}?t=${Date.now()}" alt="Scene ${scene.scene_id}">
                <div class="image-card-body">
                    <div class="image-narration">${scene.narration || scene.sentence || ''}</div>
                    <div class="image-actions">
                        <button class="btn-image-action btn-regenerate" onclick="app.regenerateImage('${projectId}', ${scene.scene_id})">🔄 재생성</button>
                        <button class="btn-image-action btn-i2v" onclick="app.convertToVideo('${projectId}', ${scene.scene_id})" ${scene.i2v_converted ? 'disabled' : ''}>${scene.i2v_converted ? '✅ I2V' : '🎬 I2V'}</button>
                        <button class="btn-image-action btn-hook ${scene.hook_video_enabled ? 'active' : ''}" onclick="app.toggleHookVideo('${projectId}', ${scene.scene_id})">${scene.hook_video_enabled ? '⭐ Hook' : '☆ Hook'}</button>
                    </div>
                </div>
            `;

            grid.appendChild(card);
        });

        this.projectId = projectId;
    }

    async regenerateImage(projectId, sceneId) {
        const card = document.querySelector(`[data-scene-id="${sceneId}"]`);
        if (!card) {
            console.error(`[regenerateImage] Card not found for scene ${sceneId}`);
            alert('씬 카드를 찾을 수 없습니다.');
            return;
        }

        const btn = card.querySelector('.btn-regenerate');
        if (btn) {
            btn.textContent = '⏳...';
            btn.disabled = true;
        }

        try {
            const response = await fetch(`${this.getApiBaseUrl()}/api/regenerate/image/${projectId}/${sceneId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });

            if (!response.ok) {
                let errorDetail = response.statusText;
                try {
                    const errorBody = await response.json();
                    errorDetail = errorBody.detail || errorBody.error || errorBody.message || JSON.stringify(errorBody);
                } catch (e) {
                    try { errorDetail = await response.text(); } catch (e2) {}
                }
                throw new Error(`${response.status}: ${errorDetail}`);
            }

            const result = await response.json();
            const img = card.querySelector('img');
            if (img) {
                const imageUrl = this.resolveImageUrl(result.image_path);
                img.src = `${imageUrl}?t=${Date.now()}`;
            }

        } catch (error) {
            console.error('[regenerateImage] Error:', error);
            alert(`재생성 실패: ${error.message}`);
        } finally {
            if (btn) {
                btn.textContent = '🔄 재생성';
                btn.disabled = false;
            }
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
                    testImg.src = `${this.getApiBaseUrl()}${d.test_image_url}?t=${Date.now()}`;
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
                testImg.src = `${this.getApiBaseUrl()}${working.test_image_url}?t=${Date.now()}`;
                testImg.style.cssText = 'max-width:200px; margin-top:8px; border-radius:8px;';
                resultDiv.appendChild(testImg);
            }
        } catch (error) {
            resultDiv.textContent = `테스트 실패: ${error.message}`;
        }

        btn.textContent = '🧪 이미지 생성 테스트';
        btn.disabled = false;
    }

    async convertToVideo(projectId, sceneId) {
        const card = document.querySelector(`[data-scene-id="${sceneId}"]`);
        const btn = card.querySelector('.btn-i2v');
        btn.textContent = '⏳...';
        btn.disabled = true;

        try {
            const response = await fetch(`${this.getApiBaseUrl()}/api/convert/i2v/${projectId}/${sceneId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ motion_prompt: "camera slowly pans and zooms" })
            });

            if (!response.ok) {
                let errorDetail = response.statusText;
                try {
                    const errorBody = await response.json();
                    errorDetail = errorBody.detail || errorBody.error || errorBody.message || JSON.stringify(errorBody);
                } catch (e) {
                    try { errorDetail = await response.text(); } catch (e2) {}
                }
                throw new Error(`${response.status}: ${errorDetail}`);
            }

            btn.textContent = '✅ I2V';
            alert(`Scene ${sceneId} I2V 변환 완료!`);

        } catch (error) {
            alert(`I2V 실패: ${error.message}`);
            btn.textContent = '🎬 I2V';
            btn.disabled = false;
        }
    }

    async toggleHookVideo(projectId, sceneId) {
        const card = document.querySelector(`[data-scene-id="${sceneId}"]`);
        const btn = card.querySelector('.btn-hook');
        const isHook = card.classList.contains('hook-video');

        try {
            const response = await fetch(`${this.getApiBaseUrl()}/api/toggle/hook-video/${projectId}/${sceneId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enable: !isHook })
            });

            if (!response.ok) {
                let errorDetail = response.statusText;
                try {
                    const errorBody = await response.json();
                    errorDetail = errorBody.detail || errorBody.error || errorBody.message || JSON.stringify(errorBody);
                } catch (e) {
                    try { errorDetail = await response.text(); } catch (e2) {}
                }
                throw new Error(`${response.status}: ${errorDetail}`);
            }

            if (!isHook) {
                card.classList.add('hook-video');
                btn.classList.add('active');
                btn.textContent = '⭐ Hook';
                const header = card.querySelector('.image-card-header');
                if (!header.querySelector('.hook-badge')) {
                    header.innerHTML += '<span class="hook-badge">🎥 HOOK</span>';
                }
            } else {
                card.classList.remove('hook-video');
                btn.classList.remove('active');
                btn.textContent = '☆ Hook';
                card.querySelector('.hook-badge')?.remove();
            }

        } catch (error) {
            alert(`Hook 설정 실패: ${error.message}`);
        }
    }

    async startFinalGenerationAfterImageReview() {
        if (!this.projectId) {
            alert('프로젝트 ID가 없습니다.');
            return;
        }

        // 중복 생성 방지
        if (this.isGenerating) {
            alert('이미 영상 생성이 진행 중입니다.');
            return;
        }

        this.isGenerating = true;

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
            const token = localStorage.getItem('token');
            const headers = { 'Content-Type': 'application/json' };
            if (token) {
                headers['Authorization'] = `Bearer ${token}`;
            }

            const response = await fetch(`${this.getApiBaseUrl()}/api/generate/video`, {
                method: 'POST',
                headers: headers,
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
                    try { errorDetail = await response.text(); } catch (e2) {}
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
            alert(`영상 생성 실패: ${error.message}`);
            this.isGenerating = false;
            // 에러 시 이미지 프리뷰 화면으로 복귀
            this.showSection('image-preview');
        }
    }

    // ==================== Music Video Mode ====================

    initMVEventListeners() {
        // MV 네비게이션
        document.getElementById('nav-mv')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.showSection('mv');
            this.setNavActive('nav-mv');
        });

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

        // MV 새로 만들기
        document.getElementById('mv-new-btn')?.addEventListener('click', () => {
            this.resetMVUI();
        });

        // MV 로그 클리어
        document.getElementById('mv-clear-log-btn')?.addEventListener('click', () => {
            document.getElementById('mv-log-content').innerHTML = '';
        });
    }

    async uploadAndAnalyzeMusic() {
        const fileInput = document.getElementById('mv-music-file');
        const file = fileInput.files[0];

        if (!file) {
            alert('음악 파일을 선택해주세요.');
            return;
        }

        const btn = document.getElementById('mv-upload-btn');
        const originalText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = '<span class="btn-icon">⏳</span> 분석 중...';

        try {
            const formData = new FormData();
            formData.append('music_file', file);
            formData.append('lyrics', document.getElementById('mv-lyrics').value || '');
            formData.append('concept', document.getElementById('mv-concept').value || '');
            formData.append('genre', document.getElementById('mv-genre').value);
            formData.append('mood', document.getElementById('mv-mood').value);
            formData.append('style', document.getElementById('mv-style').value);

            const baseUrl = this.getApiBaseUrl();
            const response = await fetch(`${baseUrl}/api/mv/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                let errorMsg = '음악 업로드 실패';
                try {
                    const error = await response.json();
                    errorMsg = error.detail || error.message || errorMsg;
                } catch (e) {}
                throw new Error(errorMsg);
            }

            const result = await response.json();
            this.mvProjectId = result.project_id;
            this.mvAnalysis = result.music_analysis;

            // Gemini로 추출된 가사가 있고, 사용자가 직접 입력하지 않았으면 자동 채우기
            const lyricsInput = document.getElementById('mv-lyrics');
            if (result.extracted_lyrics && !lyricsInput.value.trim()) {
                lyricsInput.value = result.extracted_lyrics;
                console.log(`[MV] Auto-filled lyrics: ${result.extracted_lyrics.length} chars`);
            }

            this.mvRequestParams = {
                lyrics: lyricsInput.value || '',
                concept: document.getElementById('mv-concept').value || '',
                genre: document.getElementById('mv-genre').value,
                mood: document.getElementById('mv-mood').value,
                style: document.getElementById('mv-style').value
            };

            this.renderMVAnalysisResult(result);
            this.showSection('mv-analysis');

        } catch (error) {
            console.error('MV 업로드 실패:', error);
            alert(`오류: ${error.message}`);
        } finally {
            btn.disabled = false;
            btn.innerHTML = originalText;
        }
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
        if (!this.mvProjectId) {
            alert('프로젝트 ID가 없습니다. 다시 업로드해주세요.');
            return;
        }

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
        btn.disabled = true;
        btn.innerHTML = '<span class="btn-icon">⏳</span> 생성 요청 중...';

        try {
            const baseUrl = this.getApiBaseUrl();
            const response = await fetch(`${baseUrl}/api/mv/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    project_id: this.mvProjectId,
                    lyrics: this.mvRequestParams?.lyrics || '',
                    concept: this.mvRequestParams?.concept || '',
                    genre: this.mvRequestParams?.genre || 'fantasy',
                    mood: this.mvRequestParams?.mood || 'epic',
                    style: this.mvRequestParams?.style || 'cinematic',
                    scene_descriptions: sceneDescriptions
                })
            });

            if (!response.ok) {
                let errorMsg = 'MV 생성 요청 실패';
                try {
                    const error = await response.json();
                    errorMsg = error.detail || error.message || errorMsg;
                } catch (e) {}
                throw new Error(errorMsg);
            }

            const result = await response.json();

            // 진행 화면으로 전환
            this.showSection('mv-progress');
            this.mvAddLog('INFO', `✅ MV 생성 시작 (Project: ${this.mvProjectId})`);
            this.mvAddLog('INFO', `📊 총 ${result.total_scenes}개 씬, 예상 소요: ${Math.ceil(result.estimated_time_sec / 60)}분`);

            // 진행률 초기화
            this.updateMVProgress(5, '씬 프롬프트 생성 중...');
            this.updateMVStepStatus('scenes', '진행 중');

            // 폴링 시작
            this.startMVPolling(this.mvProjectId);

        } catch (error) {
            console.error('MV 생성 요청 실패:', error);
            alert(`오류: ${error.message}`);
            btn.disabled = false;
            btn.innerHTML = '<span class="btn-icon">🎬</span> 뮤직비디오 생성 시작';
        }
    }

    startMVPolling(projectId) {
        if (this.mvPollingInterval) {
            clearInterval(this.mvPollingInterval);
        }

        const baseUrl = this.getApiBaseUrl();

        this.mvPollingFailCount = 0;
        this.mvPollingInterval = setInterval(async () => {
            try {
                const response = await fetch(`${baseUrl}/api/mv/status/${projectId}`);

                if (!response.ok) {
                    console.warn(`MV status check failed: ${response.status}`);
                    return;
                }

                // 연결 성공 시 실패 카운터 리셋
                this.mvPollingFailCount = 0;

                const data = await response.json();

                // 상태별 처리
                if (data.status === 'completed') {
                    this.mvAddLog('SUCCESS', '🎉 뮤직비디오 생성 완료!');
                    this.updateMVProgress(100, '완료');
                    this.stopMVPolling();
                    this.fetchMVResult(projectId);

                } else if (data.status === 'failed') {
                    this.mvAddLog('ERROR', `❌ 오류: ${data.error_message || '알 수 없는 오류'}`);
                    this.updateMVProgress(0, '실패');
                    this.stopMVPolling();
                    alert(`뮤직비디오 생성 실패: ${data.error_message || '알 수 없는 오류'}`);

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
                    alert('서버와의 연결이 끊어졌습니다.\n서버가 재시작 중일 수 있습니다.\n잠시 후 페이지를 새로고침 해주세요.');
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

    async fetchMVResult(projectId) {
        try {
            const baseUrl = this.getApiBaseUrl();
            const response = await fetch(`${baseUrl}/api/mv/result/${projectId}`);

            if (!response.ok) {
                throw new Error('결과 조회 실패');
            }

            const result = await response.json();
            this.showMVResult(result);

        } catch (error) {
            console.error('MV 결과 조회 실패:', error);
            this.mvAddLog('ERROR', `결과 조회 실패: ${error.message}`);
        }
    }

    showMVResult(data) {
        this.showSection('mv-result');

        document.getElementById('mv-result-project-id').textContent = data.project_id;

        const durationMin = Math.floor(data.duration_sec / 60);
        const durationSec = Math.floor(data.duration_sec % 60);
        document.getElementById('mv-result-duration').textContent = `${durationMin}:${durationSec.toString().padStart(2, '0')}`;
        document.getElementById('mv-result-scene-count').textContent = `${data.scenes?.length || 0}개`;

        // 비디오 플레이어
        const baseUrl = this.getApiBaseUrl();
        const video = document.getElementById('mv-result-video');
        video.src = `${baseUrl}/api/mv/stream/${data.project_id}`;

        // 다운로드 버튼
        const downloadBtn = document.getElementById('mv-download-btn');
        downloadBtn.href = `${baseUrl}/api/mv/download/${data.project_id}`;
        downloadBtn.download = `musicvideo_${data.project_id}.mp4`;

        // 씬 그리드
        this.renderMVResultSceneGrid(data.scenes || []);
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
                    <img src="${imageUrl}" alt="Scene ${scene.scene_id}"
                        onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
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

    renderMVResultSceneGrid(scenes) {
        const grid = document.getElementById('mv-result-scene-grid');
        grid.innerHTML = '';

        scenes.forEach(scene => {
            const card = document.createElement('div');
            card.className = 'scene-card';

            const imageUrl = scene.image_path ? this.resolveImageUrl(scene.image_path) : '';

            card.innerHTML = `
                ${imageUrl ? `<img src="${imageUrl}?t=${Date.now()}" alt="Scene ${scene.scene_id}">` : '<div class="scene-placeholder">📷</div>'}
                <div class="scene-info">
                    <span class="scene-narration">Scene ${scene.scene_id}</span>
                    <span class="scene-visual">${scene.lyrics_text || ''}</span>
                </div>
            `;

            grid.appendChild(card);
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

    resetMVUI() {
        this.mvProjectId = null;
        this.mvAnalysis = null;
        this.mvRequestParams = null;
        this.stopMVPolling();

        document.getElementById('mv-form').reset();
        this.showSection('mv');
        this.setNavActive('nav-mv');
    }
}

// 앱 초기화
const app = new StorycutApp();

// MV 이벤트 리스너 초기화
app.initMVEventListeners();
