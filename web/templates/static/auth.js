// Auth Logic
const API_BASE_URL = window.location.hostname === 'localhost' ? '' : 'https://web-production-bb6bf.up.railway.app';

document.addEventListener('DOMContentLoaded', () => {
    // 로그인 폼
    const loginForm = document.getElementById('login-form');
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = loginForm.email.value;
            const password = loginForm.password.value;

            try {
                const res = await fetch(`${API_BASE_URL}/api/auth/login`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, password })
                });
                const data = await res.json();

                if (!res.ok) throw new Error(data.error || '로그인 실패');

                // 토큰/정보 저장
                localStorage.setItem('token', data.token);
                localStorage.setItem('user', JSON.stringify(data.user));

                alert(`환영합니다, ${data.user.username}님!`);
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
                const res = await fetch(`${API_BASE_URL}/api/auth/register`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, email, password })
                });
                const data = await res.json();

                if (!res.ok) throw new Error(data.error || '회원가입 실패');

                alert('가입이 완료되었습니다! 로그인해주세요.');
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
    const user = JSON.parse(localStorage.getItem('user'));

    // 비로그인 상태면 로그인 페이지로
    // (선택 사항: 둘러보기 허용하려면 막지 않아도 됨. 일단 막음)
    // 비로그인 상태면 로그인 페이지로 강제 이동
    const path = window.location.pathname;
    if (!token && (path === '/' || path === '/index.html')) {
        window.location.href = '/login.html';
    }

    // 로그인 상태면 상단에 정보 표시 (간단히)
    if (user && document.querySelector('.header')) {
        const userInfo = document.createElement('div');
        userInfo.style.position = 'absolute';
        userInfo.style.top = '1rem';
        userInfo.style.right = '1rem';
        userInfo.style.color = '#fff';
        userInfo.innerHTML = `
            <span>👤 ${user.username} | 💰 ${user.credits} Credits</span>
            <button onclick="logout()" style="margin-left:10px; padding: 5px 10px; background:#ff4b4b; border:none; border-radius:4px; color:white; cursor:pointer;">로그아웃</button>
        `;
        document.querySelector('.header').appendChild(userInfo);
    }
}

function logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    window.location.href = '/login.html';
}
