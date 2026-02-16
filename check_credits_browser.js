// Clip balance check script
// Run in browser console

(async function () {
    const token = localStorage.getItem('auth_token');
    if (!token) {
        console.log('Not logged in. Please log in first.');
        return;
    }

    const WORKER_URL = 'https://storycut-worker.twinspa0713.workers.dev';

    try {
        const response = await fetch(`${WORKER_URL}/api/clips/balance`, {
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();
        console.log('Clip info:', data);
        console.log('Current clips:', data.clips);

        // Update display
        if (typeof fetchClipBalance === 'function') {
            fetchClipBalance();
            console.log('Display updated');
        }

    } catch (error) {
        console.error('Clip query failed:', error);
    }
})();
