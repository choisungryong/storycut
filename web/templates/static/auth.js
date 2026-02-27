// Auth Logic + Clip System + Google OAuth + Profile Modal
const API_BASE_URL = window.location.hostname === 'localhost' ? '' : 'https://web-production-bb6bf.up.railway.app';
const WORKER_URL = 'https://storycut-worker.twinspa0713.workers.dev';

// ==================== Clip Costs (synced with Worker) ====================
const CLIP_COSTS = {
    video: 25,
    script_video: 25,
    mv: 15,
    image_regen: 1,
    i2v: 30,
    mv_recompose: 8,
};

const CLIP_LABELS = {
    video: 'AI 스토리 영상',
    script_video: '스크립트 영상',
    mv: '뮤직비디오',
    image_regen: '이미지 재생성',
    i2v: 'I2V 변환',
    mv_recompose: 'MV 재합성',
};

// ==================== Google OAuth ====================

let _googleClientId = null;

async function fetchGoogleClientId() {
    if (_googleClientId) return _googleClientId;
    try {
        const url = `${WORKER_URL}/api/config/google-client-id`;
        const res = await fetch(url);
        if (!res.ok) return null;
        const data = await res.json();
        _googleClientId = data.client_id || null;
        return _googleClientId;
    } catch {
        return null;
    }
}

let _gisInitialized = false;

async function initGoogleSignIn() {
    const clientId = await fetchGoogleClientId();
    if (!clientId) return;

    google.accounts.id.initialize({
        client_id: clientId,
        callback: handleGoogleCredential,
    });
    _gisInitialized = true;

    // 공식 Google 버튼 렌더링 (One Tap 쿨다운 억제 문제 회피)
    const container = document.getElementById('google-signin-btn');
    if (container) {
        container.innerHTML = '';
        google.accounts.id.renderButton(container, {
            type: 'standard',
            theme: 'outline',
            size: 'large',
            shape: 'rectangular',
            text: 'continue_with',
            width: 350,
            logo_alignment: 'left',
        });
    }
}

async function googleSignIn() {
    if (!_gisInitialized) {
        await initGoogleSignIn();
        if (!_gisInitialized) {
            showAuthError('Google 로그인이 설정되지 않았습니다. 이메일 로그인을 사용해 주세요.');
            return;
        }
    }
    // renderButton이 이미 처리하므로 prompt는 폴백으로만 사용
    google.accounts.id.prompt((notification) => {
        if (notification.isNotDisplayed() || notification.isSkippedMoment()) {
            showAuthError('Google 로그인 팝업이 차단되었습니다. 위 Google 버튼을 직접 클릭해 주세요.');
        }
    });
}

// GIS callback
async function handleGoogleCredential(response) {
    const idToken = response.credential;
    if (!idToken) {
        showAuthError('Google 인증에 실패했습니다.');
        return;
    }

    try {
        const url = `${WORKER_URL}/api/auth/google`;

        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id_token: idToken }),
        });
        const data = await res.json();

        if (!res.ok) throw new Error(data.error || 'Google 로그인 실패');

        localStorage.setItem('token', data.token);
        localStorage.setItem('user', JSON.stringify(data.user));
        window.location.href = '/';
    } catch (err) {
        showAuthError(err.message);
    }
}

// Expose for GIS callback
window.handleGoogleCredential = handleGoogleCredential;

// ==================== Auth Error Display ====================

function showAuthError(message) {
    const el = document.getElementById('auth-error');
    if (el) {
        el.textContent = message;
        el.classList.add('visible');
    } else {
        window.showToast ? window.showToast(message, 'error') : _authToast(message, 'error');
    }
}

// ==================== DOMContentLoaded ====================

document.addEventListener('DOMContentLoaded', () => {
    // Login form
    const loginForm = document.getElementById('login-form');
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = loginForm.email.value;
            const password = loginForm.password.value;

            try {
                const res = await fetch(`${WORKER_URL}/api/auth/login`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, password })
                });
                const data = await res.json();

                if (!res.ok) throw new Error(data.error || '로그인 실패');

                localStorage.setItem('token', data.token);
                localStorage.setItem('user', JSON.stringify(data.user));

                window.location.href = '/';

            } catch (err) {
                showAuthError(err.message);
            }
        });
    }

    // Signup form
    const signupForm = document.getElementById('signup-form');
    if (signupForm) {
        signupForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = signupForm.username.value;
            const email = signupForm.email.value;
            const password = signupForm.password.value;

            try {
                const res = await fetch(`${WORKER_URL}/api/auth/register`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, email, password })
                });
                const data = await res.json();

                if (!res.ok) throw new Error(data.error || '회원가입 실패');

                showAuthError('');
                // Auto-login after signup
                const loginRes = await fetch(`${WORKER_URL}/api/auth/login`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, password })
                });
                const loginData = await loginRes.json();
                if (loginRes.ok) {
                    localStorage.setItem('token', loginData.token);
                    localStorage.setItem('user', JSON.stringify(loginData.user));
                    window.location.href = '/';
                } else {
                    _authToast('회원가입이 완료되었습니다! 로그인해 주세요.', 'success');
                    window.location.href = '/login.html';
                }
            } catch (err) {
                showAuthError(err.message);
            }
        });
    }

    // Google Sign-In 버튼 렌더링 (로그인/회원가입 페이지)
    if (loginForm || signupForm) {
        if (typeof google !== 'undefined' && google.accounts) {
            initGoogleSignIn();
        } else {
            // GIS 라이브러리 로드 대기
            window.addEventListener('load', () => {
                if (typeof google !== 'undefined' && google.accounts) {
                    initGoogleSignIn();
                }
            });
        }
    }

    // Auth check (index.html)
    if (!loginForm && !signupForm) {
        checkAuth();
        loadPreferences();
    }
});

// ==================== Auth Check ====================

function checkAuth() {
    const token = localStorage.getItem('token');
    const user = JSON.parse(localStorage.getItem('user') || 'null');

    const path = window.location.pathname;
    if (!token) {
        // 루트(/)는 랜딩 페이지로, 앱 직접 접근 시에는 로그인 페이지로
        if (path === '/') {
            window.location.href = '/landing.html';
            return;
        }
        if (path === '/index.html' || path === '/app') {
            window.location.href = '/login.html';
            return;
        }
    }

    if (user) {
        renderUserHeader(user);
        fetchClipBalance();
    }
}

// ==================== User Header ====================

function renderUserHeader(user) {
    const navRight = document.querySelector('.app-nav__right');
    if (!navRight) return;

    // Remove existing user info
    const existing = document.getElementById('user-header-info');
    if (existing) existing.remove();

    const displayName = user.username || user.email?.split('@')[0] || '사용자';

    const userInfo = document.createElement('div');
    userInfo.id = 'user-header-info';
    userInfo.className = 'nav-user-info';
    userInfo.innerHTML = `
        <a href="/pricing.html" id="clip-badge" class="nav-clip-badge">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <text x="12" y="16" text-anchor="middle" font-size="12" font-weight="bold" fill="currentColor">C</text>
            </svg>
            <span id="clip-count">${user.credits ?? user.clips ?? '...'}</span>
        </a>
        <button class="nav-user-btn" onclick="openProfileModal()">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/>
                <circle cx="12" cy="7" r="4"/>
            </svg>
            <span>${displayName}</span>
        </button>
    `;

    navRight.insertBefore(userInfo, navRight.firstChild);
}

// ==================== Profile Modal ====================

function openProfileModal() {
    const user = JSON.parse(localStorage.getItem('user') || 'null');
    if (!user) return;

    const modal = document.getElementById('profile-modal');
    if (!modal) return;

    const displayName = user.username || user.email?.split('@')[0] || '사용자';
    const planName = (user.plan_name || user.plan_id || 'Free').charAt(0).toUpperCase() +
        (user.plan_name || user.plan_id || 'Free').slice(1);
    const memberSince = user.created_at ? new Date(user.created_at).toLocaleDateString() : '없음';

    // Header
    const nameEl = document.getElementById('profile-name');
    const emailEl = document.getElementById('profile-email');
    if (nameEl) nameEl.textContent = displayName;
    if (emailEl) emailEl.textContent = user.email || '';

    // Account tab
    const accEmail = document.getElementById('profile-account-email');
    if (accEmail) accEmail.textContent = user.email || '-';

    const googleStatus = document.getElementById('profile-google-status');
    if (googleStatus) {
        const isGoogle = user.auth_provider === 'google' || user.google_id;
        googleStatus.innerHTML = isGoogle
            ? '<span class="profile-badge profile-badge-connected">연결됨</span>'
            : '<span class="profile-badge profile-badge-disconnected">미연결</span>';
    }

    const planEl = document.getElementById('profile-plan');
    if (planEl) planEl.textContent = planName;

    const sinceEl = document.getElementById('profile-since');
    if (sinceEl) sinceEl.textContent = memberSince;

    // Clips tab
    const clipsEl = document.getElementById('profile-clips');
    if (clipsEl) clipsEl.textContent = user.clips ?? '...';

    const clipsPlan = document.getElementById('profile-clips-plan');
    if (clipsPlan) clipsPlan.textContent = planName;

    // Reset to account tab
    switchProfileTab('account');

    // Load data async
    loadProfileStats();
    loadClipHistory();
    loadPreferencesToForm();

    modal.classList.add('active');
}

function closeProfileModal() {
    const modal = document.getElementById('profile-modal');
    if (modal) modal.classList.remove('active');
    // Hide name edit if open
    const editEl = document.getElementById('profile-name-edit');
    if (editEl) editEl.style.display = 'none';
    const nameEl = document.getElementById('profile-name');
    if (nameEl) nameEl.style.display = '';
    const editBtn = document.querySelector('.profile-edit-name-btn');
    if (editBtn) editBtn.style.display = '';
}

function switchProfileTab(tab) {
    document.querySelectorAll('.profile-tab').forEach(t => {
        t.classList.toggle('active', t.dataset.tab === tab);
    });
    document.querySelectorAll('.profile-tab-panel').forEach(p => {
        p.classList.toggle('active', p.id === `profile-tab-${tab}`);
    });
}

function toggleUsernameEdit() {
    const nameEl = document.getElementById('profile-name');
    const editEl = document.getElementById('profile-name-edit');
    const editBtn = document.querySelector('.profile-edit-name-btn');
    const input = document.getElementById('profile-name-input');

    if (editEl.style.display === 'none') {
        input.value = nameEl.textContent;
        nameEl.style.display = 'none';
        editBtn.style.display = 'none';
        editEl.style.display = 'flex';
        input.focus();
    } else {
        editEl.style.display = 'none';
        nameEl.style.display = '';
        editBtn.style.display = '';
    }
}

async function updateUsername() {
    const input = document.getElementById('profile-name-input');
    const newName = input.value.trim();
    if (!newName) return;

    try {
        const token = localStorage.getItem('token');
        const res = await fetch(`${API_BASE_URL}/api/user/profile`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                ...(token ? { 'Authorization': `Bearer ${token}` } : {})
            },
            body: JSON.stringify({ username: newName })
        });
        if (!res.ok) throw new Error('업데이트 실패');

        // Update local storage
        const user = JSON.parse(localStorage.getItem('user') || '{}');
        user.username = newName;
        localStorage.setItem('user', JSON.stringify(user));

        // Update UI
        document.getElementById('profile-name').textContent = newName;
        toggleUsernameEdit();
        renderUserHeader(user);
        _authToast('이름이 변경되었습니다', 'success');
    } catch (err) {
        _authToast('이름 변경에 실패했습니다', 'error');
    }
}

async function loadProfileStats() {
    try {
        const token = localStorage.getItem('token');
        const res = await fetch(`${API_BASE_URL}/api/user/stats`, {
            headers: token ? { 'Authorization': `Bearer ${token}` } : {}
        });
        if (!res.ok) return;
        const data = await res.json();

        const el = (id) => document.getElementById(id);
        if (el('profile-stat-videos')) el('profile-stat-videos').textContent = data.total_videos ?? 0;
        if (el('profile-stat-mvs')) el('profile-stat-mvs').textContent = data.total_mv ?? 0;
        if (el('profile-stat-clips-used')) el('profile-stat-clips-used').textContent = data.total_clips_used ?? data.total_credits_used ?? 0;
        if (el('profile-stat-since')) el('profile-stat-since').textContent = data.member_since || '없음';

        const extraEl = document.getElementById('profile-stats-extra');
        if (data.top_style || data.top_genre) {
            if (extraEl) extraEl.style.display = '';
            if (el('profile-stat-top-style')) el('profile-stat-top-style').textContent = data.top_style || '-';
            if (el('profile-stat-top-genre')) el('profile-stat-top-genre').textContent = data.top_genre || '-';
        }
    } catch {
        // Stats unavailable
    }
}

async function loadClipHistory() {
    try {
        const token = localStorage.getItem('token');
        const res = await fetch(`${API_BASE_URL}/api/user/history`, {
            headers: token ? { 'Authorization': `Bearer ${token}` } : {}
        });
        if (!res.ok) return;
        const data = await res.json();

        const listEl = document.getElementById('profile-history-list');
        if (!listEl) return;

        const items = data.history || [];
        if (items.length === 0) {
            listEl.innerHTML = '<p class="profile-empty-state">아직 크레딧 사용 내역이 없습니다.</p>';
            return;
        }

        listEl.innerHTML = items.map(item => `
            <div class="profile-history-item">
                <span class="history-action">${item.action || '-'}</span>
                <span class="history-date">${item.date || ''}</span>
                <span class="history-clips">-${item.clips || item.credits || 0}</span>
            </div>
        `).join('');
    } catch {
        // History unavailable
    }
}

// ==================== Preferences ====================

function savePreferences() {
    const prefs = {
        style: document.getElementById('pref-style')?.value || '',
        genre: document.getElementById('pref-genre')?.value || '',
        language: document.getElementById('pref-language')?.value || '',
    };
    localStorage.setItem('klippa_preferences', JSON.stringify(prefs));
    _authToast('설정이 저장되었습니다', 'success');
}

function loadPreferencesToForm() {
    const prefs = JSON.parse(localStorage.getItem('klippa_preferences') || '{}');
    const prefStyle = document.getElementById('pref-style');
    const prefGenre = document.getElementById('pref-genre');
    const prefLang = document.getElementById('pref-language');
    if (prefStyle) prefStyle.value = prefs.style || '';
    if (prefGenre) prefGenre.value = prefs.genre || '';
    if (prefLang) prefLang.value = prefs.language || '';
}

function loadPreferences() {
    const prefs = JSON.parse(localStorage.getItem('klippa_preferences') || '{}');
    if (prefs.style) {
        const styleEl = document.getElementById('style');
        if (styleEl) {
            const option = Array.from(styleEl.options).find(o => o.value === prefs.style);
            if (option) styleEl.value = prefs.style;
        }
    }
    if (prefs.genre) {
        const genreEl = document.getElementById('genre');
        if (genreEl) {
            const option = Array.from(genreEl.options).find(o => o.value === prefs.genre);
            if (option) genreEl.value = prefs.genre;
        }
    }
}

// ==================== Toast ====================

function _authToast(message, type = 'error') {
    let container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
}

// ==================== Clip Functions ====================

async function fetchClipBalance() {
    const token = localStorage.getItem('token');
    if (!token) return null;

    try {
        const res = await fetch(`${WORKER_URL}/api/clips/balance`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (!res.ok) {
            if (res.status === 401) {
                logout();
                return null;
            }
            if (!window._clipErrorShown) {
                window._clipErrorShown = true;
                _authToast('크레딧 서비스를 사용할 수 없습니다. 크레딧 확인 없이 진행합니다.', 'error');
            }
            return null;
        }

        const data = await res.json();

        const user = JSON.parse(localStorage.getItem('user') || '{}');
        user.clips = data.credits;
        user.plan_id = data.plan_id;
        user.plan_name = data.plan_name;
        if (data.gemini3) user.gemini3 = data.gemini3;
        localStorage.setItem('user', JSON.stringify(user));

        updateClipDisplay(data.credits);

        return data;
    } catch (err) {
        if (!window._clipErrorShown) {
            window._clipErrorShown = true;
            _authToast('크레딧 서비스를 사용할 수 없습니다. 크레딧 확인 없이 진행합니다.', 'error');
        }
        return null;
    }
}

function updateClipDisplay(clips) {
    const el = document.getElementById('clip-count');
    if (el) el.textContent = clips;
}

async function checkClipsBeforeAction(action) {
    const cost = CLIP_COSTS[action];
    if (!cost) return true;

    const user = JSON.parse(localStorage.getItem('user') || '{}');

    if (user.clips === undefined || user.clips === null) return true;

    const clips = user.clips;

    if (clips < cost) {
        showInsufficientClipsModal(action, cost, clips);
        return false;
    }

    return true;
}

function showInsufficientClipsModal(action, cost, available) {
    const existing = document.getElementById('clip-modal');
    if (existing) existing.remove();

    const label = CLIP_LABELS[action] || action;

    const modal = document.createElement('div');
    modal.id = 'clip-modal';
    modal.style.cssText = `
        position: fixed; inset: 0; z-index: 10000;
        display: flex; align-items: center; justify-content: center;
        background: rgba(0,0,0,0.7); backdrop-filter: blur(4px);
    `;
    modal.innerHTML = `
        <div style="
            background: #1e1e2e; border: 1px solid #333; border-radius: 16px;
            padding: 32px; max-width: 420px; width: 90%; text-align: center;
        ">
            <div style="font-size: 48px; margin-bottom: 16px;">&#x26A0;</div>
            <h3 style="color: #f59e0b; margin: 0 0 12px;">크레딧 부족</h3>
            <p style="color: #ccc; margin: 0 0 8px;">
                <strong>${label}</strong>에 <strong style="color:#f59e0b">${cost}</strong> 크레딧이 필요합니다.
            </p>
            <p style="color: #888; margin: 0 0 24px;">
                현재 잔액: <strong style="color:#ef4444">${available}</strong> 크레딧
            </p>
            <div style="display: flex; gap: 10px; justify-content: center;">
                <a href="/pricing.html" style="
                    padding: 10px 24px; background: linear-gradient(135deg, #f59e0b, #d97706);
                    border: none; border-radius: 8px; color: #000; font-weight: 600;
                    text-decoration: none; cursor: pointer;
                ">크레딧 충전</a>
                <button onclick="document.getElementById('clip-modal').remove()" style="
                    padding: 10px 24px; background: #333; border: 1px solid #555;
                    border-radius: 8px; color: #ccc; cursor: pointer;
                ">취소</button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.remove();
    });
}

function showPlanUpgradeModal(action, message) {
    const existing = document.getElementById('clip-modal');
    if (existing) existing.remove();

    const label = CLIP_LABELS[action] || action;

    const modal = document.createElement('div');
    modal.id = 'clip-modal';
    modal.style.cssText = `
        position: fixed; inset: 0; z-index: 10000;
        display: flex; align-items: center; justify-content: center;
        background: rgba(0,0,0,0.7); backdrop-filter: blur(4px);
    `;
    modal.innerHTML = `
        <div style="
            background: #1e1e2e; border: 1px solid #333; border-radius: 16px;
            padding: 32px; max-width: 420px; width: 90%; text-align: center;
        ">
            <div style="font-size: 48px; margin-bottom: 16px;">&#x1F512;</div>
            <h3 style="color: #a78bfa; margin: 0 0 12px;">플랜 업그레이드 필요</h3>
            <p style="color: #ccc; margin: 0 0 8px;">
                <strong>${label}</strong>은(는) 현재 플랜에서 사용할 수 없습니다.
            </p>
            <p style="color: #888; margin: 0 0 24px;">
                ${message || '이 기능을 사용하려면 유료 플랜으로 업그레이드하세요.'}
            </p>
            <div style="display: flex; gap: 10px; justify-content: center;">
                <a href="/pricing.html" style="
                    padding: 10px 24px; background: linear-gradient(135deg, #a78bfa, #7c3aed);
                    border: none; border-radius: 8px; color: #fff; font-weight: 600;
                    text-decoration: none; cursor: pointer;
                ">요금제 보기</a>
                <button onclick="document.getElementById('clip-modal').remove()" style="
                    padding: 10px 24px; background: #333; border: 1px solid #555;
                    border-radius: 8px; color: #ccc; cursor: pointer;
                ">취소</button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.remove();
    });
}

function showGemini3SurchargeModal(action, required, available, surcharge) {
    const existing = document.getElementById('clip-modal');
    if (existing) existing.remove();

    const label = CLIP_LABELS[action] || action;

    const modal = document.createElement('div');
    modal.id = 'clip-modal';
    modal.style.cssText = `
        position: fixed; inset: 0; z-index: 10000;
        display: flex; align-items: center; justify-content: center;
        background: rgba(0,0,0,0.7); backdrop-filter: blur(4px);
    `;
    modal.innerHTML = `
        <div style="
            background: #1e1e2e; border: 1px solid #333; border-radius: 16px;
            padding: 32px; max-width: 440px; width: 90%; text-align: center;
        ">
            <div style="font-size: 48px; margin-bottom: 16px;">&#x2728;</div>
            <h3 style="color: #f59e0b; margin: 0 0 12px;">Gemini 추가 요금</h3>
            <p style="color: #ccc; margin: 0 0 8px;">
                이번 달 무료 Gemini 사용량을 모두 소진했습니다.
            </p>
            <p style="color: #ccc; margin: 0 0 8px;">
                <strong>${label}</strong>에 Gemini 사용 시
                <span style="color: #f59e0b; font-weight: 700;">${required} 크레딧</span>이
                필요합니다 (+${surcharge} 추가 요금 포함).
            </p>
            <p style="color: #888; margin: 0 0 8px;">
                보유: <strong>${available} 크레딧</strong>
            </p>
            <p style="color: #777; font-size: 13px; margin: 0 0 24px;">
                추가 요금을 피하려면 Gemini 2.5로 전환하거나, 크레딧을 충전하세요.
            </p>
            <div style="display: flex; gap: 10px; justify-content: center;">
                <a href="/pricing.html" style="
                    padding: 10px 24px; background: linear-gradient(135deg, #f59e0b, #d97706);
                    border: none; border-radius: 8px; color: #fff; font-weight: 600;
                    text-decoration: none; cursor: pointer;
                ">크레딧 충전하기</a>
                <button onclick="document.getElementById('clip-modal').remove()" style="
                    padding: 10px 24px; background: #333; border: 1px solid #555;
                    border-radius: 8px; color: #ccc; cursor: pointer;
                ">취소</button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.remove();
    });
}

/**
 * Get Gemini status for current user.
 * Returns { allowed, free_limit, used, surcharge_per_image, willSurcharge }
 */
function getGemini3Status() {
    const user = JSON.parse(localStorage.getItem('user') || '{}');
    const g3 = user.gemini3 || {};
    return {
        allowed: g3.allowed || false,
        freeLimit: g3.free_limit ?? 0,
        used: g3.used || 0,
        surchargePerImage: g3.surcharge_per_image || 2,
        willSurcharge: g3.allowed && g3.free_limit >= 0 && (g3.used || 0) >= (g3.free_limit || 0),
    };
}

/**
 * Central API error handler for 402/403 responses from Worker.
 * Returns true if the error was handled (modal shown), false otherwise.
 */
async function handleApiError(response, action) {
    if (response.ok) return false;

    if (response.status === 402) {
        try {
            const data = await response.json();
            if (data.gemini3_surcharge > 0) {
                showGemini3SurchargeModal(action || data.action || 'unknown', data.required || 0, data.available || 0, data.gemini3_surcharge);
            } else {
                showInsufficientClipsModal(action || data.action || 'unknown', data.required || 0, data.available || 0);
            }
        } catch (e) {
            showInsufficientClipsModal(action || 'unknown', 0, 0);
        }
        return true;
    }

    if (response.status === 403) {
        try {
            const data = await response.json();
            showPlanUpgradeModal(action || data.action || 'unknown', data.error || '');
        } catch (e) {
            showPlanUpgradeModal(action || 'unknown', '');
        }
        return true;
    }

    return false;
}

function deductLocalClips(action) {
    const cost = CLIP_COSTS[action] || 0;
    const user = JSON.parse(localStorage.getItem('user') || '{}');
    user.clips = Math.max(0, (user.clips || 0) - cost);
    localStorage.setItem('user', JSON.stringify(user));
    updateClipDisplay(user.clips);
}

// Legacy compatibility aliases
const CREDIT_COSTS = CLIP_COSTS;
const CREDIT_LABELS = CLIP_LABELS;
function fetchCreditBalance() { return fetchClipBalance(); }
function updateCreditDisplay(v) { return updateClipDisplay(v); }
function checkCreditsBeforeAction(a) { return checkClipsBeforeAction(a); }
function showInsufficientCreditsModal(a, c, v) { return showInsufficientClipsModal(a, c, v); }
function deductLocalCredits(a) { return deductLocalClips(a); }

function logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    window.location.href = '/login.html';
}
