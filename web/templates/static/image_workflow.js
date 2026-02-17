// ==================== 이미지 생성 워크플로우 메서드 ====================
// StorycutApp 클래스에 추가할 메서드들

// 이미지만 생성 (Step 2A)
async startImageGeneration() {
    if (!this.currentStoryData) {
        this.showToast('스토리 데이터가 없습니다.', 'warning');
        return;
    }

    const apiUrl = this.getApiBaseUrl();

    try {
        // 수정된 스토리 데이터 가져오기 (사용자가 review에서 수정했을 수 있음)
        const title = document.getElementById('review-title').value;
        this.currentStoryData.title = title;

        // Scene 데이터 업데이트
        document.querySelectorAll('.review-card').forEach((card, index) => {
            const sceneId = parseInt(card.dataset.sceneId);
            const scene = this.currentStoryData.scenes.find(s => s.scene_id === sceneId);
            if (scene) {
                scene.narration = card.querySelector('.review-textarea[name="narration"]').value;
                scene.visual_description = card.querySelector('.visual-textarea').value;
            }
        });

        console.log('[Image Generation] Starting image-only generation...');

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
            throw new Error(`Image generation failed: ${response.statusText}`);
        }

        const result = await response.json();
        console.log('[Image Generation] Success:', result);

        // 이미지 프리뷰 섹션으로 이동
        this.renderImagePreview(result);
        this.showSection('image-preview');

    } catch (error) {
        console.error('[Image Generation] Error:', error);
        this.showToast('이미지 생성에 실패했습니다. 다시 시도해주세요.', 'error');
    }
}

// 이미지 프리뷰 렌더링
renderImagePreview(data) {
    const grid = document.getElementById('image-preview-grid');
    grid.innerHTML = '';

    const scenes = data.scenes || data.story_data?.scenes || [];

    scenes.forEach(scene => {
        const card = document.createElement('div');
        card.className = 'image-card';
        card.dataset.sceneId = scene.scene_id;

        if (scene.hook_video_enabled) {
            card.classList.add('hook-video');
        }

        const imagePath = scene.assets?.image_path || scene.image_path || '';
        const imageUrl = imagePath.startsWith('http') ? imagePath : `${this.getApiBaseUrl()}${imagePath}`;

        card.innerHTML = `
            <div class="image-card-header">
                <span class="image-card-title">Scene ${scene.scene_id}</span>
                ${scene.hook_video_enabled ? '<span class="hook-badge">🎥 HOOK</span>' : ''}
            </div>
            <img src="${imageUrl}" alt="Scene ${scene.scene_id}" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22300%22 height=%22220%22%3E%3Crect fill=%22%23252a34%22 width=%22300%22 height=%22220%22/%3E%3Ctext x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 fill=%22%23666%22%3EImage Loading...%3C/text%3E%3C/svg%3E'">
            <div class="image-card-body">
                <div class="image-narration">${scene.narration || scene.sentence || ''}</div>
                <div class="image-actions">
                    <button class="btn-image-action btn-regenerate" onclick="app.regenerateImage(${this.projectId}, ${scene.scene_id})">
                        🔄 재생성
                    </button>
                    <button class="btn-image-action btn-i2v" onclick="app.convertToVideo(${this.projectId}, ${scene.scene_id})" ${scene.i2v_converted ? 'disabled' : ''}>
                        ${scene.i2v_converted ? '✅ I2V 완료' : '🎬 I2V 변환'}
                    </button>
                    <button class="btn-image-action btn-hook ${scene.hook_video_enabled ? 'active' : ''}" onclick="app.toggleHookVideo(${this.projectId}, ${scene.scene_id})">
                        ${scene.hook_video_enabled ? '⭐ Hook' : '☆ Hook'}
                    </button>
                </div>
            </div>
        `;

        grid.appendChild(card);
    });

    // 프로젝트 ID 저장
    this.projectId = data.project_id;
}

// 이미지 재생성
async regenerateImage(projectId, sceneId) {
    const apiUrl = this.getApiBaseUrl();
    const card = document.querySelector(`[data-scene-id="${sceneId}"]`);
    const btn = card.querySelector('.btn-regenerate');

    const originalText = btn.textContent;
    btn.textContent = '⏳ 생성 중...';
    btn.disabled = true;

    try {
        console.log(`[Regenerate] Regenerating image for scene ${sceneId}...`);

        const response = await fetch(`${apiUrl}/api/regenerate/image/${projectId}/${sceneId}`, {
            method: 'POST',
            headers: this.getAuthHeaders(),
            body: JSON.stringify({})
        });

        if (!response.ok) {
            throw new Error(`Regeneration failed: ${response.statusText}`);
        }

        const result = await response.json();
        console.log('[Regenerate] Success:', result);

        // 이미지 업데이트
        const img = card.querySelector('img');
        const newImageUrl = result.image_path.startsWith('http')
            ? result.image_path
            : `${apiUrl}${result.image_path}`;
        img.src = `${newImageUrl}?t=${Date.now()}`; // Cache busting

        btn.textContent = originalText;
        btn.disabled = false;

    } catch (error) {
        console.error('[Regenerate] Error:', error);
        this.showToast('이미지 재생성에 실패했습니다.', 'error');
        btn.textContent = originalText;
        btn.disabled = false;
    }
}

// I2V 변환
async convertToVideo(projectId, sceneId) {
    const apiUrl = this.getApiBaseUrl();
    const card = document.querySelector(`[data-scene-id="${sceneId}"]`);
    const btn = card.querySelector('.btn-i2v');

    const originalText = btn.textContent;
    btn.textContent = '⏳ 변환 중...';
    btn.disabled = true;

    try {
        console.log(`[I2V] Converting scene ${sceneId} to video...`);

        const response = await fetch(`${apiUrl}/api/convert/i2v/${projectId}/${sceneId}`, {
            method: 'POST',
            headers: this.getAuthHeaders(),
            body: JSON.stringify({
                motion_prompt: "camera slowly pans and zooms"
            })
        });

        if (!response.ok) {
            throw new Error(`I2V conversion failed: ${response.statusText}`);
        }

        const result = await response.json();
        console.log('[I2V] Success:', result);

        btn.textContent = '✅ I2V 완료';
        btn.disabled = true;

        this.showToast(`Scene ${sceneId} I2V 변환 완료!`, 'success');

    } catch (error) {
        console.error('[I2V] Error:', error);
        this.showToast(`I2V 변환 실패: ${error.message}`, 'error');
        btn.textContent = originalText;
        btn.disabled = false;
    }
}

// Hook Video 토글
async toggleHookVideo(projectId, sceneId) {
    const apiUrl = this.getApiBaseUrl();
    const card = document.querySelector(`[data-scene-id="${sceneId}"]`);
    const btn = card.querySelector('.btn-hook');
    const isCurrentlyHook = card.classList.contains('hook-video');

    const newState = !isCurrentlyHook;

    try {
        console.log(`[Hook] Toggling hook video for scene ${sceneId} to ${newState}...`);

        const response = await fetch(`${apiUrl}/api/toggle/hook-video/${projectId}/${sceneId}`, {
            method: 'POST',
            headers: this.getAuthHeaders(),
            body: JSON.stringify({ enable: newState })
        });

        if (!response.ok) {
            throw new Error(`Hook toggle failed: ${response.statusText}`);
        }

        const result = await response.json();
        console.log('[Hook] Success:', result);

        // UI 업데이트
        if (newState) {
            card.classList.add('hook-video');
            btn.classList.add('active');
            btn.textContent = '⭐ Hook';

            // Header에 badge 추가
            const header = card.querySelector('.image-card-header');
            if (!header.querySelector('.hook-badge')) {
                const badge = document.createElement('span');
                badge.className = 'hook-badge';
                badge.textContent = '🎥 HOOK';
                header.appendChild(badge);
            }
        } else {
            card.classList.remove('hook-video');
            btn.classList.remove('active');
            btn.textContent = '☆ Hook';

            // Badge 제거
            const badge = card.querySelector('.hook-badge');
            if (badge) badge.remove();
        }

    } catch (error) {
        console.error('[Hook] Error:', error);
        this.showToast('Hook Video 설정에 실패했습니다.', 'error');
    }
}

// 이미지 승인 후 최종 영상 생성
async startFinalGenerationAfterImageReview() {
    if (!this.projectId) {
        this.showToast('프로젝트 ID가 없습니다.', 'warning');
        return;
    }

    // 기존 startFinalGeneration과 동일하게 진행
    // 이미 이미지가 생성되어 있으므로 바로 영상 합성으로 진행
    this.showSection('progress');
    this.resetProgress();

    const apiUrl = this.getApiBaseUrl();

    try {
        console.log('[Final Generation] Starting final video generation...');

        const response = await fetch(`${apiUrl}/api/generate/video`, {
            method: 'POST',
            headers: this.getAuthHeaders(),
            body: JSON.stringify({
                project_id: this.projectId,
                story_data: this.currentStoryData,
                request_params: this.currentRequestParams
            })
        });

        if (!response.ok) {
            throw new Error(`Video generation failed: ${response.statusText}`);
        }

        const result = await response.json();
        console.log('[Final Generation] Started:', result);

        this.projectId = result.project_id;

        // WebSocket + Polling
        this.connectWebSocket(this.projectId);
        this.startPolling(this.projectId);

    } catch (error) {
        console.error('[Final Generation] Error:', error);
        this.showToast('영상 생성 시작에 실패했습니다.', 'error');
    }
}
