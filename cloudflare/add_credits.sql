-- SQL Script to add 200 clips to neopioneer0713@gmail.com
-- Note: DB column is still 'credits' until migration 002 is applied

-- 1. Check current user clips
SELECT id, email, credits AS clips FROM users WHERE email = 'neopioneer0713@gmail.com';

-- 2. Add 200 clips to the user
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
    'Manual clip grant - 200 clips added'
FROM users
WHERE email = 'neopioneer0713@gmail.com';

-- 4. Verify the update
SELECT id, email, credits AS clips FROM users WHERE email = 'neopioneer0713@gmail.com';
