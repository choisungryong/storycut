// Auth Logic + Credit System
const API_BASE_URL = window.location.hostname === 'localhost' ? '' : 'https://web-production-bb6bf.up.railway.app';
const WORKER_URL = window.location.hostname === 'localhost' ? '' : 'https://storycut-worker.twinspa0713.workers.dev';

// ==================== 크레딧 비용 (Worker와 동기화) ====================
const CREDIT_COSTS = {
    video: 5,
    script_video: 5,
    mv: 10,
    image_regen: 1,
    i2v: 2,
    mv_recompose: 2,
};

const CREDIT_LABELS = {
    video: 'AI 스토리 영상',
    script_video: '스크립트 영상',
    mv: '뮤직비디오',
    image_regen: '이미지 재생성',
    i2v: 'I2V 변환',
    mv_recompose: 'MV 리컴포즈',
};

document.addEventListener('DOMContentLoaded', () => {
    // 로그인 폼
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

                if (!res.ok) throw new Error(data.error || 'Login failed');

                localStorage.setItem('token', data.token);
                localStorage.setItem('user', JSON.stringify(data.user));

                window.location.href = '/index.html';

            } catch (err) {
                alert(err.message);
            }
        });
    }

    // 회원가입 폼
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

                if (!res.ok) throw new Error(data.error || 'Registration failed');

                alert('Registration complete! Please log in.');
                window.location.href = '/login.html';

            } catch (err) {
                alert(err.message);
            }
        });
    }

    // 인증 체크 (index.html 용)
    if (!loginForm && !signupForm) {
        checkAuth();
    }
});

function checkAuth() {
    const token = localStorage.getItem('token');
    const user = JSON.parse(localStorage.getItem('user') || 'null');

    const path = window.location.pathname;
    if (!token && (path === '/' || path === '/index.html')) {
        window.location.href = '/login.html';
        return;
    }

    if (user && document.querySelector('.header')) {
        renderUserHeader(user);
        // 서버에서 최신 크레딧 가져오기
        fetchCreditBalance();
    }
}

function renderUserHeader(user) {
    const header = document.querySelector('.header');
    // 기존 userInfo 제거
    const existing = document.getElementById('user-header-info');
    if (existing) existing.remove();

    const userInfo = document.createElement('div');
    userInfo.id = 'user-header-info';
    userInfo.style.cssText = `
        display: flex; align-items: center; justify-content: flex-end; gap: 12px;
        padding: 8px 0; margin-bottom: 8px;
    `;
    userInfo.innerHTML = `
        <a href="/pricing.html" id="credit-badge" style="
            display: inline-flex; align-items: center; gap: 6px;
            background: linear-gradient(135deg, #f59e0b22, #f59e0b11);
            border: 1px solid #f59e0b44;
            padding: 6px 14px; border-radius: 20px;
            color: #f59e0b; font-weight: 600; font-size: 0.9rem;
            text-decoration: none; cursor: pointer; transition: all 0.2s;
        ">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2"/>
                <text x="12" y="16" text-anchor="middle" font-size="12" font-weight="bold" fill="currentColor">C</text>
            </svg>
            <span id="credit-count">${user.credits ?? '...'}</span>
        </a>
        <span style="color: #888; font-size: 0.85rem;">${user.username || user.email?.split('@')[0] || 'User'}</span>
        <button onclick="logout()" style="
            padding: 5px 12px; background: #ff4b4b; border: none;
            border-radius: 6px; color: white; cursor: pointer; font-size: 0.8rem;
        ">Logout</button>
    `;
    header.insertBefore(userInfo, header.firstChild);
}

async function fetchCreditBalance() {
    const token = localStorage.getItem('token');
    if (!token) return null;

    try {
        const res = await fetch(`${WORKER_URL}/api/credits/balance`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (!res.ok) {
            if (res.status === 401) {
                // 토큰 만료 → 로그아웃
                logout();
                return null;
            }
            return null;
        }

        const data = await res.json();

        // localStorage 업데이트
        const user = JSON.parse(localStorage.getItem('user') || '{}');
        user.credits = data.credits;
        user.plan_id = data.plan_id;
        user.plan_name = data.plan_name;
        localStorage.setItem('user', JSON.stringify(user));

        // UI 업데이트
        updateCreditDisplay(data.credits);

        return data;
    } catch (err) {
        console.warn('Failed to fetch credit balance:', err);
        return null;
    }
}

function updateCreditDisplay(credits) {
    const el = document.getElementById('credit-count');
    if (el) el.textContent = credits;
}

/**
 * 크레딧 사전 확인 - 생성 액션 전에 호출
 * @param {string} action - 'video', 'script_video', 'mv', 'image_regen', 'i2v', 'mv_recompose'
 * @returns {boolean} true if sufficient credits
 */
async function checkCreditsBeforeAction(action) {
    const cost = CREDIT_COSTS[action];
    if (!cost) return true;

    const user = JSON.parse(localStorage.getItem('user') || '{}');

    // Worker 미배포 등으로 크레딧 정보가 없으면 체크 스킵
    if (user.credits === undefined || user.credits === null) return true;

    const credits = user.credits;

    if (credits < cost) {
        showInsufficientCreditsModal(action, cost, credits);
        return false;
    }

    return true;
}

function showInsufficientCreditsModal(action, cost, available) {
    // 기존 모달 제거
    const existing = document.getElementById('credit-modal');
    if (existing) existing.remove();

    const label = CREDIT_LABELS[action] || action;

    const modal = document.createElement('div');
    modal.id = 'credit-modal';
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
            <h3 style="color: #f59e0b; margin: 0 0 12px;">Credit Insufficient</h3>
            <p style="color: #ccc; margin: 0 0 8px;">
                <strong>${label}</strong> requires <strong style="color:#f59e0b">${cost}</strong> credits.
            </p>
            <p style="color: #888; margin: 0 0 24px;">
                Current balance: <strong style="color:#ef4444">${available}</strong> credits
            </p>
            <div style="display: flex; gap: 10px; justify-content: center;">
                <a href="/pricing.html" style="
                    padding: 10px 24px; background: linear-gradient(135deg, #f59e0b, #d97706);
                    border: none; border-radius: 8px; color: #000; font-weight: 600;
                    text-decoration: none; cursor: pointer;
                ">Get Credits</a>
                <button onclick="document.getElementById('credit-modal').remove()" style="
                    padding: 10px 24px; background: #333; border: 1px solid #555;
                    border-radius: 8px; color: #ccc; cursor: pointer;
                ">Cancel</button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.remove();
    });
}

/**
 * 크레딧 차감 후 잔액 업데이트 (성공적인 생성 요청 후 호출)
 */
function deductLocalCredits(action) {
    const cost = CREDIT_COSTS[action] || 0;
    const user = JSON.parse(localStorage.getItem('user') || '{}');
    user.credits = Math.max(0, (user.credits || 0) - cost);
    localStorage.setItem('user', JSON.stringify(user));
    updateCreditDisplay(user.credits);
}

function logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    window.location.href = '/login.html';
}
