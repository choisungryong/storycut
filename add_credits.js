// Script to add clips to a user account via admin API
// Usage: node add_credits.js

const WORKER_URL = 'YOUR_WORKER_URL_HERE'; // e.g., 'https://your-worker.workers.dev'
const ADMIN_EMAIL = 'YOUR_ADMIN_EMAIL_HERE'; // Your admin email (must be in ADMIN_EMAILS env var)
const ADMIN_PASSWORD = 'YOUR_ADMIN_PASSWORD_HERE'; // Admin password

const TARGET_EMAIL = 'neopioneer0713@gmail.com';
const CLIPS_TO_ADD = 200;
const REASON = 'Manual clip grant';

async function main() {
    try {
        console.log('1. Logging in as admin...');

        // Login to get token
        const loginResponse = await fetch(`${WORKER_URL}/api/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                email: ADMIN_EMAIL,
                password: ADMIN_PASSWORD,
            }),
        });

        if (!loginResponse.ok) {
            const error = await loginResponse.json();
            throw new Error(`Login failed: ${JSON.stringify(error)}`);
        }

        const loginData = await loginResponse.json();
        const token = loginData.token;

        console.log('Login successful');
        console.log(`2. Adding ${CLIPS_TO_ADD} clips to ${TARGET_EMAIL}...`);

        // Grant clips
        const grantResponse = await fetch(`${WORKER_URL}/api/admin/grant-clips`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`,
            },
            body: JSON.stringify({
                target_email: TARGET_EMAIL,
                amount: CLIPS_TO_ADD,
                reason: REASON,
            }),
        });

        if (!grantResponse.ok) {
            const error = await grantResponse.json();
            throw new Error(`Clip grant failed: ${JSON.stringify(error)}`);
        }

        const grantData = await grantResponse.json();
        console.log('Clips added successfully!');
        console.log('Result:', grantData);

    } catch (error) {
        console.error('Error:', error.message);
        process.exit(1);
    }
}

main();
