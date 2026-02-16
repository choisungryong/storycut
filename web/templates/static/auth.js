// Auth Logic + Credit System + Google OAuth + Profile Modal
const API_BASE_URL = window.location.hostname === 'localhost' ? '' : 'https://web-production-bb6bf.up.railway.app';
const WORKER_URL = window.location.hostname === 'localhost' ? '' : 'https://storycut-worker.twinspa0713.workers.dev';

// ==================== Credit Costs (synced with Worker) ====================
const CREDIT_COSTS = {
    video: 5,
    script_video: 5,
    mv: 10,
    image_regen: 1,
    i2v: 2,
    mv_recompose: 2,
};

const CREDIT_LABELS = {
    video: 'AI Story Video',
    script_video: 'Script Video',
    mv: 'Music Video',
    image_regen: 'Image Regen',
    i2v: 'I2V Conversion',
    mv_recompose: 'MV Recompose',
};

// ==================== Google OAuth ====================

let _googleClientId = null;

async function fetchGoogleClientId() {
    if (_googleClientId) return _googleClientId;
    try {
        const url = window.location.hostname === 'localhost'
            ? '/api/config/google-client-id'
            : `${WORKER_URL}/api/config/google-client-id`;
        const res = await fetch(url);
        if (!res.ok) return null;
        const data = await res.json();
        _googleClientId = data.client_id || null;
        return _googleClientId;
    } catch {
        return null;
    }
}

async function googleSignIn() {
    const clientId = await fetchGoogleClientId();
    if (!clientId) {
        showAuthError('Google login is not configured. Please use email login.');
        return;
    }

    // Use Google Identity Services popup flow
    google.accounts.id.initialize({
        client_id: clientId,
        callback: handleGoogleCredential,
    });
    google.accounts.id.prompt();
}

// GIS callback
async function handleGoogleCredential(response) {
    const idToken = response.credential;
    if (!idToken) {
        showAuthError('Google authentication failed.');
        return;
    }

    try {
        const url = window.location.hostname === 'localhost'
            ? '/api/auth/google'
            : `${WORKER_URL}/api/auth/google`;

        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id_token: idToken }),
        });
        const data = await res.json();

        if (!res.ok) throw new Error(data.error || 'Google login failed');

        localStorage.setItem('token', data.token);
        localStorage.setItem('user', JSON.stringify(data.user));
        window.location.href = '/app';
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
        alert(message);
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

                if (!res.ok) throw new Error(data.error || 'Login failed');

                localStorage.setItem('token', data.token);
                localStorage.setItem('user', JSON.stringify(data.user));

                window.location.href = '/app';

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

                if (!res.ok) throw new Error(data.error || 'Registration failed');

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
                    window.location.href = '/app';
                } else {
                    alert('Registration complete! Please log in.');
                    window.location.href = '/login.html';
                }
            } catch (err) {
                showAuthError(err.message);
            }
        });
    }

    // Auth check (index.html)
    if (!loginForm && !signupForm) {
        checkAuth();
    }
});

// ==================== Auth Check ====================

function checkAuth() {
    const token = localStorage.getItem('token');
    const user = JSON.parse(localStorage.getItem('user') || 'null');

    const path = window.location.pathname;
    if (!token && (path === '/' || path === '/index.html' || path === '/app')) {
        window.location.href = '/login.html';
        return;
    }

    if (user) {
        renderUserHeader(user);
        fetchCreditBalance();
    }
}

// ==================== User Header ====================

function renderUserHeader(user) {
    const navRight = document.querySelector('.app-nav__right');
    if (!navRight) return;

    // Remove existing user info
    const existing = document.getElementById('user-header-info');
    if (existing) existing.remove();

    const displayName = user.username || user.email?.split('@')[0] || 'User';

    const userInfo = document.createElement('div');
    userInfo.id = 'user-header-info';
    userInfo.className = 'nav-user-info';
    userInfo.innerHTML = `
        <a href="/pricing.html" id="credit-badge" class="nav-credit-badge">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <text x="12" y="16" text-anchor="middle" font-size="12" font-weight="bold" fill="currentColor">C</text>
            </svg>
            <span id="credit-count">${user.credits ?? '...'}</span>
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

    // Fill data
    const nameEl = document.getElementById('profile-name');
    const emailEl = document.getElementById('profile-email');
    const creditsEl = document.getElementById('profile-credits');
    const planEl = document.getElementById('profile-plan');
    const sinceEl = document.getElementById('profile-since');

    if (nameEl) nameEl.textContent = user.username || user.email?.split('@')[0] || 'User';
    if (emailEl) emailEl.textContent = user.email || '';
    if (creditsEl) creditsEl.textContent = user.credits ?? '...';
    if (planEl) planEl.textContent = (user.plan_name || user.plan_id || 'Free').charAt(0).toUpperCase() +
        (user.plan_name || user.plan_id || 'Free').slice(1);
    if (sinceEl) {
        const created = user.created_at ? new Date(user.created_at).toLocaleDateString() : 'N/A';
        sinceEl.textContent = created;
    }

    modal.classList.add('active');
}

function closeProfileModal() {
    const modal = document.getElementById('profile-modal');
    if (modal) modal.classList.remove('active');
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

// ==================== Credit Functions ====================

async function fetchCreditBalance() {
    const token = localStorage.getItem('token');
    if (!token) return null;

    try {
        const res = await fetch(`${WORKER_URL}/api/credits/balance`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (!res.ok) {
            if (res.status === 401) {
                logout();
                return null;
            }
            if (!window._creditErrorShown) {
                window._creditErrorShown = true;
                _authToast('Credit service unavailable. Proceeding without credit check.', 'error');
            }
            return null;
        }

        const data = await res.json();

        const user = JSON.parse(localStorage.getItem('user') || '{}');
        user.credits = data.credits;
        user.plan_id = data.plan_id;
        user.plan_name = data.plan_name;
        localStorage.setItem('user', JSON.stringify(user));

        updateCreditDisplay(data.credits);

        return data;
    } catch (err) {
        if (!window._creditErrorShown) {
            window._creditErrorShown = true;
            _authToast('Credit service unavailable. Proceeding without credit check.', 'error');
        }
        return null;
    }
}

function updateCreditDisplay(credits) {
    const el = document.getElementById('credit-count');
    if (el) el.textContent = credits;
}

async function checkCreditsBeforeAction(action) {
    const cost = CREDIT_COSTS[action];
    if (!cost) return true;

    const user = JSON.parse(localStorage.getItem('user') || '{}');

    if (user.credits === undefined || user.credits === null) return true;

    const credits = user.credits;

    if (credits < cost) {
        showInsufficientCreditsModal(action, cost, credits);
        return false;
    }

    return true;
}

function showInsufficientCreditsModal(action, cost, available) {
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
