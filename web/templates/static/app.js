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
        const backToStoryBtn = document.getElementById('back-to-story-btn');
        if (backToStoryBtn) {
            backToStoryBtn.addEventListener('click', () => {
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
            // 스토리 생성은 Worker에서 처리
            const workerUrl = this.getWorkerUrl();
            const response = await fetch(`${workerUrl}/api/generate/story`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || '스토리 생성 실패');
            }

            const result = await response.json();

            // 새로운 비동기 방식: project_id를 받고 즉시 progress 화면으로 전환
            if (result.project_id && result.status === 'processing') {
                // 즉시 progress 화면으로 전환
                btn.disabled = false;
                btn.innerHTML = originalBtnText;
                this.showSection('progress');
                this.updateStepStatus('story', '스토리 생성 중...');
                document.getElementById('status-message').textContent = '스토리를 생성하고 있습니다...';
                document.getElementById('progress-percentage').textContent = '10%';
                document.getElementById('progress-bar').style.width = '10%';

                // 폴링으로 완료 대기
                const storyData = await this.pollStoryStatus(result.project_id, workerUrl);

                if (storyData) {
                    this.currentStoryData = storyData;
                    this.currentRequestParams = requestData;

                    // 스토리 리뷰 페이지로 이동
                    this.updateStepStatus('story', '완료');
                    this.renderStoryReview(this.currentStoryData);
                    this.showSection('review');
                    this.setNavActive('nav-create');
                } else {
                    throw new Error('스토리 생성 시간 초과 또는 실패');
                }
            } else if (result.story_data) {
                // 하위 호환: 기존 동기 방식 (Railway 백엔드)
                this.currentStoryData = result.story_data;
                this.currentRequestParams = result.request_params;

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

            let urlToUse = this.getApiBaseUrl();

            // 인증 토큰 가져오기
            const token = localStorage.getItem('token');
            if (!token) {
                alert('로그인이 필요합니다.');
                this.showSection('input'); // 로그인 화면이 없으므로 일단 입력 화면으로
                // 실제로는 로그인 모달을 띄우거나 로그인 페이지로 이동해야 함
                return;
            }

            const payload = {
                request_params: this.currentRequestParams,
                story_data: this.currentStoryData
            };

            this.addLog('INFO', '📤 영상 생성 요청 전송 중...');

            const response = await fetch(`${urlToUse}/api/generate/video`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
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

            // 진행률 초기화
            this.resetProgress();

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

    resetProgress() {
        this.updateProgress(5, '초기화 중...');

        // 단계 초기화
        document.querySelectorAll('.step').forEach(el => {
            el.classList.remove('active', 'completed');
            el.querySelector('.step-status').textContent = '대기 중';
        });

        // 첫 번째 단계 활성화
        const firstStep = document.querySelector('[data-step="story"]');
        if (firstStep) {
            firstStep.classList.add('active');
            firstStep.querySelector('.step-status').textContent = '진행 중';
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
        this.pollingInterval = setInterval(async () => {
            try {
                let urlToUse = this.getApiBaseUrl();
                const response = await fetch(`${urlToUse}/api/status/${projectId}`);

                if (!response.ok) {
                    console.error(`Status check failed: ${response.status}`);
                    return;
                }

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
                console.error('Polling error:', error);
                // 일시적 오류는 무시하고 계속 재시도
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
        // 모든 섹션 숨기기
        document.getElementById('input-section').classList.add('hidden');
        document.getElementById('progress-section').classList.add('hidden');
        document.getElementById('result-section').classList.add('hidden');
        document.getElementById('review-section').classList.add('hidden');
        document.getElementById('history-section').classList.add('hidden');

        // 선택한 섹션 표시
        switch (sectionName) {
            case 'input':
                document.getElementById('input-section').classList.remove('hidden');
                break;
            case 'review':
                document.getElementById('review-section').classList.remove('hidden');
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
        }
    }

    resetUI() {
        this.projectId = null;
        this.currentStoryData = null;
        this.currentRequestParams = null;
        this.isGenerating = false;
        this.stopPolling();

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

    // ==================== 이미지 생성 워크플로우 ====================

    async startImageGeneration() {
        if (!this.currentStoryData) {
            alert('스토리 데이터가 없습니다.');
            return;
        }

        const apiUrl = this.getApiBaseUrl();

        try {
            const title = document.getElementById('review-title').value;
            this.currentStoryData.title = title;

            document.querySelectorAll('.review-card').forEach((card, index) => {
                const sceneId = parseInt(card.dataset.sceneId);
                const scene = this.currentStoryData.scenes.find(s => s.scene_id === sceneId);
                if (scene) {
                    scene.narration = card.querySelector('.review-textarea[name="narration"]').value;
                    scene.visual_description = card.querySelector('.visual-textarea').value;
                }
            });

            console.log('[Image Generation] Starting...');

            const response = await fetch(`${apiUrl}/api/generate/images`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    project_id: this.projectId,
                    story_data: this.currentStoryData,
                    request_params: this.currentRequestParams
                })
            });

            if (!response.ok) throw new Error(`Failed: ${response.statusText}`);

            const result = await response.json();
            this.renderImagePreview(result);
            this.showSection('image-preview');

        } catch (error) {
            console.error('[Image Generation] Error:', error);
            alert(`이미지 생성 실패: ${error.message}`);
        }
    }

    renderImagePreview(data) {
        const grid = document.getElementById('image-preview-grid');
        grid.innerHTML = '';

        const scenes = data.scenes || data.story_data?.scenes || [];

        scenes.forEach(scene => {
            const card = document.createElement('div');
            card.className = 'image-card';
            card.dataset.sceneId = scene.scene_id;

            if (scene.hook_video_enabled) card.classList.add('hook-video');

            const imagePath = scene.assets?.image_path || scene.image_path || '';
            const imageUrl = imagePath.startsWith('http') ? imagePath : `${this.getApiBaseUrl()}${imagePath}`;

            card.innerHTML = `
                <div class="image-card-header">
                    <span class="image-card-title">Scene ${scene.scene_id}</span>
                    ${scene.hook_video_enabled ? '<span class="hook-badge">🎥 HOOK</span>' : ''}
                </div>
                <img src="${imageUrl}?t=${Date.now()}" alt="Scene ${scene.scene_id}">
                <div class="image-card-body">
                    <div class="image-narration">${scene.narration || scene.sentence || ''}</div>
                    <div class="image-actions">
                        <button class="btn-image-action btn-regenerate" onclick="app.regenerateImage('${this.projectId}', ${scene.scene_id})">🔄 재생성</button>
                        <button class="btn-image-action btn-i2v" onclick="app.convertToVideo('${this.projectId}', ${scene.scene_id})" ${scene.i2v_converted ? 'disabled' : ''}>${scene.i2v_converted ? '✅ I2V' : '🎬 I2V'}</button>
                        <button class="btn-image-action btn-hook ${scene.hook_video_enabled ? 'active' : ''}" onclick="app.toggleHookVideo('${this.projectId}', ${scene.scene_id})">${scene.hook_video_enabled ? '⭐ Hook' : '☆ Hook'}</button>
                    </div>
                </div>
            `;

            grid.appendChild(card);
        });

        this.projectId = data.project_id;
    }

    async regenerateImage(projectId, sceneId) {
        const card = document.querySelector(`[data-scene-id="${sceneId}"]`);
        const btn = card.querySelector('.btn-regenerate');
        btn.textContent = '⏳...';
        btn.disabled = true;

        try {
            const response = await fetch(`${this.getApiBaseUrl()}/api/regenerate/image/${projectId}/${sceneId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });

            if (!response.ok) throw new Error(`Failed: ${response.statusText}`);

            const result = await response.json();
            const img = card.querySelector('img');
            img.src = `${this.getApiBaseUrl()}${result.image_path}?t=${Date.now()}`;

            btn.textContent = '🔄 재생성';
            btn.disabled = false;

        } catch (error) {
            alert(`재생성 실패: ${error.message}`);
            btn.textContent = '🔄 재생성';
            btn.disabled = false;
        }
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

            if (!response.ok) throw new Error(`Failed: ${response.statusText}`);

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

            if (!response.ok) throw new Error(`Failed: ${response.statusText}`);

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

        this.showSection('progress');
        this.resetProgress();

        try {
            const response = await fetch(`${this.getApiBaseUrl()}/api/generate/video`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    project_id: this.projectId,
                    story_data: this.currentStoryData,
                    request_params: this.currentRequestParams
                })
            });

            if (!response.ok) throw new Error(`Failed: ${response.statusText}`);

            const result = await response.json();
            this.projectId = result.project_id;

            this.connectWebSocket(this.projectId);
            this.startPolling(this.projectId);

        } catch (error) {
            alert(`영상 생성 실패: ${error.message}`);
        }
    }
}

// 앱 초기화
const app = new StorycutApp();
