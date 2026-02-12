-- SQL Script to add 200 credits to neopioneer0713@gmail.com
-- This script is compatible with the actual database schema

-- 1. Check current user credits
SELECT id, email, credits FROM users WHERE email = 'neopioneer0713@gmail.com';

-- 2. Add 200 credits to the user
UPDATE users 
SET credits = credits + 200
WHERE email = 'neopioneer0713@gmail.com';

-- 3. Record the transaction
INSERT INTO credit_transactions (
    user_id, 
    amount, 
    type, 
    description
)
SELECT 
    id,
    200,
    'purchase',
    'Manual credit grant - 200 credits added'
FROM users 
WHERE email = 'neopioneer0713@gmail.com';

-- 4. Verify the update
SELECT id, email, credits FROM users WHERE email = 'neopioneer0713@gmail.com';
