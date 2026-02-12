// 크레딧 잔액을 직접 확인하는 스크립트
// 브라우저 콘솔에서 실행하세요

(async function () {
    const token = localStorage.getItem('auth_token');
    if (!token) {
        console.log('❌ 로그인되어 있지 않습니다. 먼저 로그인해주세요.');
        return;
    }

    const WORKER_URL = 'https://storycut-worker.twinspa0713.workers.dev';

    try {
        const response = await fetch(`${WORKER_URL}/api/credits/balance`, {
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`API 오류: ${response.status}`);
        }

        const data = await response.json();
        console.log('📊 크레딧 정보:', data);
        console.log('💰 현재 크레딧:', data.credits);

        // 화면 업데이트
        if (typeof fetchCreditBalance === 'function') {
            fetchCreditBalance();
            console.log('🔄 화면 크레딧 표시 업데이트 완료');
        }

    } catch (error) {
        console.error('❌ 크레딧 조회 실패:', error);
    }
})();
