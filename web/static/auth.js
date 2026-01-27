// STORYCUT 인증 모듈

class AuthManager {
    constructor() {
        this.apiUrl = window.location.hostname === 'localhost'
            ? 'http://localhost:8000'
            : 'https://api.storycut.com';

        this.token = localStorage.getItem('storycut_token');
        this.user = null;
    }

    /**
     * 사용자 로그인
     */
    async login(email, password) {
        try {
            const response = await fetch(`${this.apiUrl}/api/auth/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email, password })
            });

            if (!response.ok) {
                throw new Error('로그인 실패');
            }

            const data = await response.json();
            this.token = data.token;
            this.user = data.user;

            localStorage.setItem('storycut_token', this.token);
            localStorage.setItem('storycut_user', JSON.stringify(this.user));

            return this.user;
        } catch (error) {
            console.error('Login error:', error);
            throw error;
        }
    }

    /**
     * 사용자 회원가입
     */
    async register(email, password) {
        try {
            const response = await fetch(`${this.apiUrl}/api/auth/register`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email, password })
            });

            if (!response.ok) {
                throw new Error('회원가입 실패');
            }

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Register error:', error);
            throw error;
        }
    }

    /**
     * 로그아웃
     */
    logout() {
        this.token = null;
        this.user = null;
        localStorage.removeItem('storycut_token');
        localStorage.removeItem('storycut_user');
        window.location.reload();
    }

    /**
     * 현재 사용자 정보 가져오기
     */
    async getCurrentUser() {
        if (!this.token) {
            return null;
        }

        // 캐시된 사용자 정보
        const cachedUser = localStorage.getItem('storycut_user');
        if (cachedUser) {
            this.user = JSON.parse(cachedUser);
            return this.user;
        }

        // API에서 최신 정보 가져오기
        try {
            const response = await fetch(`${this.apiUrl}/api/auth/me`, {
                headers: {
                    'Authorization': `Bearer ${this.token}`,
                }
            });

            if (!response.ok) {
                this.logout();
                return null;
            }

            this.user = await response.json();
            localStorage.setItem('storycut_user', JSON.stringify(this.user));
            return this.user;
        } catch (error) {
            console.error('Get user error:', error);
            return null;
        }
    }

    /**
     * 인증 헤더 반환
     */
    getAuthHeaders() {
        if (!this.token) {
            return {};
        }

        return {
            'Authorization': `Bearer ${this.token}`,
        };
    }

    /**
     * 로그인 여부 확인
     */
    isAuthenticated() {
        return !!this.token;
    }

    /**
     * 크레딧 충전 페이지로 이동
     */
    goToCreditsPage() {
        window.location.href = '/credits.html';
    }

    /**
     * 크레딧 잔액 확인
     */
    async getCredits() {
        const user = await this.getCurrentUser();
        return user ? user.credits : 0;
    }
}

// 전역 AuthManager 인스턴스
window.authManager = new AuthManager();
