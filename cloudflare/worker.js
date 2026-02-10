/**
 * STORYCUT Cloudflare Worker
 *
 * 역할:
 * 1. API 엔드포인트 제공 (/api/*)
 * 2. JWT 인증 및 크레딧 검증
 * 3. Railway 백엔드 프록시 + 크레딧 차감
 * 4. R2에서 영상 서빙
 * 5. D1에 메타데이터 저장
 * 6. PortOne(포트원) V2 결제 연동
 * 7. 크레딧 관리 (DeeVid 모델)
 */

// ==================== 크레딧 & 플랜 설정 ====================

const CREDIT_COSTS = {
  video: 5,
  script_video: 5,
  mv: 10,
  image_regen: 1,
  i2v: 2,
  mv_recompose: 2,
};

const PLANS = {
  free:    { name: 'Free',    monthlyCredits: 0,    priceKrw: 0,       yearlyPriceKrw: 0 },
  lite:    { name: 'Lite',    monthlyCredits: 200,  priceKrw: 11900,   yearlyPriceKrw: 119000 },
  pro:     { name: 'Pro',     monthlyCredits: 600,  priceKrw: 29900,   yearlyPriceKrw: 299000 },
  premium: { name: 'Premium', monthlyCredits: 3000, priceKrw: 139000,  yearlyPriceKrw: 1390000 },
};

const CREDIT_PACKS = {
  small:  { credits: 50,  priceKrw: 5900 },
  medium: { credits: 200, priceKrw: 17900 },
  large:  { credits: 500, priceKrw: 35900 },
};

const SIGNUP_BONUS_CREDITS = 20;

// ==================== 메인 엔트리 ====================

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    };

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    try {
      return await routeRequest(url, request, env, ctx, corsHeaders);
    } catch (error) {
      console.error('Unhandled error:', error);
      return jsonResponse({ error: 'Internal server error' }, 500, corsHeaders);
    }
  },

  // 정기결제 자동 갱신 (Cron Trigger — 매일 09:00 KST)
  async scheduled(event, env, ctx) {
    ctx.waitUntil(processSubscriptionRenewals(env));
  },
};

// ==================== 라우터 ====================

async function routeRequest(url, request, env, ctx, cors) {
  const path = url.pathname;
  const method = request.method;

  // Public routes
  if (path === '/api/health') return handleHealth(env, cors);
  if (path === '/api/payments/plans' && method === 'GET') return handleGetPlans(cors);
  if (path === '/api/payments/webhook' && method === 'POST') return handlePortoneWebhook(request, env, cors);

  // Auth routes
  if (path === '/api/auth/register' && method === 'POST') return handleRegister(request, env, cors);
  if (path === '/api/auth/login' && method === 'POST') return handleLogin(request, env, cors);

  // ---------- 인증 필요 라우트 ----------
  const user = await authenticateUser(request, env);
  if (!user) return jsonResponse({ error: 'Unauthorized' }, 401, cors);

  // Credits
  if (path === '/api/credits/balance' && method === 'GET') return handleCreditBalance(user, env, cors);
  if (path === '/api/credits/history' && method === 'GET') return handleCreditHistory(user, url, env, cors);
  if (path === '/api/credits/check' && method === 'POST') return handleCreditCheck(request, user, cors);

  // Payments (PortOne)
  if (path === '/api/payments/prepare' && method === 'POST') return handlePaymentPrepare(request, user, env, cors);
  if (path === '/api/payments/complete' && method === 'POST') return handlePaymentComplete(request, user, env, cors);
  if (path === '/api/payments/subscribe' && method === 'POST') return handleSubscribe(request, user, env, cors);
  if (path === '/api/payments/cancel-subscription' && method === 'POST') return handleCancelSubscription(user, env, cors);

  // Admin
  if (path === '/api/admin/grant-credits' && method === 'POST') return handleAdminGrantCredits(request, user, env, cors);

  // Generation (크레딧 차감 포함)
  if (path === '/api/generate' && method === 'POST') return handleGenerate(request, user, env, ctx, cors);

  // Video download
  if (path.startsWith('/api/video/')) return handleVideoDownload(url, env, cors);

  // Status
  if (path.startsWith('/api/status/')) return handleStatus(url, env, cors);

  // ---- Railway 프록시 라우트 (크레딧 차감 후 프록시) ----

  // 일반 영상 생성 프록시
  if (path === '/api/generate/story' && method === 'POST') {
    return proxyToRailway(request, env, user, 'video', path, cors);
  }
  if (path === '/api/generate/from-script' && method === 'POST') {
    return proxyToRailway(request, env, user, 'script_video', path, cors);
  }
  if (path === '/api/generate/video' && method === 'POST') {
    return proxyToRailway(request, env, user, null, path, cors);
  }
  if (path === '/api/generate/images' && method === 'POST') {
    return proxyToRailway(request, env, user, null, path, cors);
  }

  // I2V 변환
  if (path.match(/^\/api\/convert\/i2v\//) && method === 'POST') {
    return proxyToRailway(request, env, user, 'i2v', path, cors);
  }

  // 이미지 재생성
  if (path.match(/^\/api\/regenerate\/image\//) && method === 'POST') {
    return proxyToRailway(request, env, user, 'image_regen', path, cors);
  }
  if (path.match(/^\/api\/projects\/[^/]+\/scenes\/[^/]+\/regenerate/) && method === 'POST') {
    return proxyToRailway(request, env, user, 'image_regen', path, cors);
  }

  // MV 생성
  if (path === '/api/mv/generate' && method === 'POST') {
    return proxyToRailway(request, env, user, 'mv', path, cors);
  }

  // MV 씬 이미지 재생성
  if (path.match(/^\/api\/mv\/scenes\/[^/]+\/[^/]+\/regenerate/) && method === 'POST') {
    return proxyToRailway(request, env, user, 'image_regen', path, cors);
  }

  // MV I2V
  if (path.match(/^\/api\/mv\/scenes\/[^/]+\/[^/]+\/i2v/) && method === 'POST') {
    return proxyToRailway(request, env, user, 'i2v', path, cors);
  }

  // MV 리컴포즈
  if (path.match(/^\/api\/mv\/[^/]+\/recompose/) && method === 'POST') {
    return proxyToRailway(request, env, user, 'mv_recompose', path, cors);
  }

  // 기타 Railway 프록시 (크레딧 차감 없음)
  if (path.startsWith('/api/')) {
    return proxyToRailway(request, env, user, null, path, cors);
  }

  return jsonResponse({ error: 'Not Found' }, 404, cors);
}

// ==================== Railway 프록시 ====================

const RAILWAY_URL = 'https://web-production-bb6bf.up.railway.app';

async function proxyToRailway(request, env, user, creditAction, path, cors) {
  if (creditAction) {
    const cost = CREDIT_COSTS[creditAction];
    if (cost && user.credits < cost) {
      return jsonResponse({
        error: 'Insufficient credits',
        required: cost,
        available: user.credits,
        action: creditAction,
      }, 402, cors);
    }

    if (cost) {
      await deductCredits(env, user.id, cost, creditAction, `${creditAction} generation`);
    }
  }

  const railwayUrl = `${RAILWAY_URL}${path}`;
  const headers = new Headers(request.headers);
  headers.set('X-User-Id', user.id);
  if (creditAction) {
    headers.set('X-Credits-Charged', String(CREDIT_COSTS[creditAction] || 0));
  }

  try {
    const response = await fetch(railwayUrl, {
      method: request.method,
      headers,
      body: request.method !== 'GET' ? await request.clone().arrayBuffer() : undefined,
    });

    const responseHeaders = new Headers(response.headers);
    Object.entries(cors).forEach(([k, v]) => responseHeaders.set(k, v));

    return new Response(response.body, {
      status: response.status,
      headers: responseHeaders,
    });
  } catch (error) {
    if (creditAction) {
      const cost = CREDIT_COSTS[creditAction];
      if (cost) {
        await refundCredits(env, user.id, cost, creditAction, `${creditAction} proxy failed - refund`);
      }
    }
    return jsonResponse({ error: 'Backend unavailable' }, 502, cors);
  }
}

// ==================== 인증 ====================

async function authenticateUser(request, env) {
  const authHeader = request.headers.get('Authorization');
  if (!authHeader || !authHeader.startsWith('Bearer ')) return null;

  const token = authHeader.substring(7);

  try {
    const payload = decodeJWT(token, env.JWT_SECRET);
    if (!payload) return null;

    const result = await env.DB.prepare(
      `SELECT id, email, credits, plan_id, plan_expires_at, monthly_credits,
              stripe_customer_id, stripe_subscription_id
       FROM users WHERE id = ?`
    ).bind(payload.sub).first();

    return result;
  } catch (error) {
    try {
      const result = await env.DB.prepare(
        `SELECT id, email, credits, plan_id, plan_expires_at, monthly_credits,
                stripe_customer_id, stripe_subscription_id
         FROM users WHERE api_token = ?`
      ).bind(token).first();
      return result;
    } catch (e) {
      console.error('Auth error:', e);
      return null;
    }
  }
}

function decodeJWT(token, secret) {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) return null;

    const payload = JSON.parse(atob(parts[1].replace(/-/g, '+').replace(/_/g, '/')));

    if (payload.exp && payload.exp < Math.floor(Date.now() / 1000)) return null;

    return payload;
  } catch (e) {
    return null;
  }
}

async function createJWT(payload, secret) {
  const header = { alg: 'HS256', typ: 'JWT' };
  const now = Math.floor(Date.now() / 1000);
  const tokenPayload = {
    ...payload,
    iat: now,
    exp: now + 86400,
  };

  const encodedHeader = btoa(JSON.stringify(header)).replace(/=/g, '').replace(/\+/g, '-').replace(/\//g, '_');
  const encodedPayload = btoa(JSON.stringify(tokenPayload)).replace(/=/g, '').replace(/\+/g, '-').replace(/\//g, '_');

  const data = `${encodedHeader}.${encodedPayload}`;

  const key = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(secret || 'storycut-secret-key'),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign']
  );
  const signature = await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(data));
  const encodedSignature = btoa(String.fromCharCode(...new Uint8Array(signature)))
    .replace(/=/g, '').replace(/\+/g, '-').replace(/\//g, '_');

  return `${data}.${encodedSignature}`;
}

// ==================== 회원가입 / 로그인 ====================

async function handleRegister(request, env, cors) {
  try {
    const { username, email, password } = await request.json();
    if (!email || !password) {
      return jsonResponse({ error: 'Email and password required' }, 400, cors);
    }

    const existing = await env.DB.prepare('SELECT id FROM users WHERE email = ?').bind(email).first();
    if (existing) {
      return jsonResponse({ error: 'Email already registered' }, 409, cors);
    }

    const userId = crypto.randomUUID();
    const passwordHash = await hashPassword(password);

    await env.DB.prepare(
      `INSERT INTO users (id, email, password_hash, credits, plan_id, created_at)
       VALUES (?, ?, ?, ?, 'free', ?)`
    ).bind(userId, email, passwordHash, SIGNUP_BONUS_CREDITS, new Date().toISOString()).run();

    await recordTransaction(env, userId, SIGNUP_BONUS_CREDITS, 'signup_bonus', null, 'Signup bonus credits');

    return jsonResponse({ message: 'Registered successfully', user: { id: userId, email, credits: SIGNUP_BONUS_CREDITS } }, 201, cors);
  } catch (error) {
    console.error('Register error:', error);
    return jsonResponse({ error: error.message }, 500, cors);
  }
}

async function handleLogin(request, env, cors) {
  try {
    const { email, password } = await request.json();
    if (!email || !password) {
      return jsonResponse({ error: 'Email and password required' }, 400, cors);
    }

    const user = await env.DB.prepare(
      'SELECT id, email, password_hash, credits, plan_id FROM users WHERE email = ?'
    ).bind(email).first();

    if (!user || !user.password_hash) {
      return jsonResponse({ error: 'Invalid credentials' }, 401, cors);
    }

    const valid = await verifyPassword(password, user.password_hash);
    if (!valid) {
      return jsonResponse({ error: 'Invalid credentials' }, 401, cors);
    }

    const token = await createJWT({ sub: user.id, email: user.email }, env.JWT_SECRET);

    return jsonResponse({
      token,
      user: {
        id: user.id,
        email: user.email,
        username: user.email.split('@')[0],
        credits: user.credits,
        plan_id: user.plan_id || 'free',
      },
    }, 200, cors);
  } catch (error) {
    console.error('Login error:', error);
    return jsonResponse({ error: error.message }, 500, cors);
  }
}

// ==================== 패스워드 해싱 ====================

async function hashPassword(password) {
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const key = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(password),
    'PBKDF2',
    false,
    ['deriveBits']
  );
  const hash = await crypto.subtle.deriveBits(
    { name: 'PBKDF2', salt, iterations: 100000, hash: 'SHA-256' },
    key,
    256
  );
  const saltHex = Array.from(salt).map(b => b.toString(16).padStart(2, '0')).join('');
  const hashHex = Array.from(new Uint8Array(hash)).map(b => b.toString(16).padStart(2, '0')).join('');
  return `${saltHex}:${hashHex}`;
}

async function verifyPassword(password, stored) {
  const [saltHex, hashHex] = stored.split(':');
  if (!saltHex || !hashHex) return false;

  const salt = new Uint8Array(saltHex.match(/.{2}/g).map(h => parseInt(h, 16)));
  const key = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(password),
    'PBKDF2',
    false,
    ['deriveBits']
  );
  const hash = await crypto.subtle.deriveBits(
    { name: 'PBKDF2', salt, iterations: 100000, hash: 'SHA-256' },
    key,
    256
  );
  const computedHex = Array.from(new Uint8Array(hash)).map(b => b.toString(16).padStart(2, '0')).join('');
  return computedHex === hashHex;
}

// ==================== 크레딧 관리 ====================

async function handleCreditBalance(user, env, cors) {
  const freshUser = await env.DB.prepare(
    `SELECT credits, plan_id, plan_expires_at, monthly_credits FROM users WHERE id = ?`
  ).bind(user.id).first();

  const planInfo = PLANS[freshUser.plan_id] || PLANS.free;

  return jsonResponse({
    credits: freshUser.credits,
    plan_id: freshUser.plan_id || 'free',
    plan_name: planInfo.name,
    plan_expires_at: freshUser.plan_expires_at,
    monthly_credits: freshUser.monthly_credits,
    costs: CREDIT_COSTS,
  }, 200, cors);
}

async function handleCreditHistory(user, url, env, cors) {
  const page = parseInt(url.searchParams.get('page') || '1');
  const limit = Math.min(parseInt(url.searchParams.get('limit') || '20'), 50);
  const offset = (page - 1) * limit;

  const total = await env.DB.prepare(
    'SELECT COUNT(*) as count FROM credit_transactions WHERE user_id = ?'
  ).bind(user.id).first();

  const transactions = await env.DB.prepare(
    `SELECT id, amount, type, action, description, created_at
     FROM credit_transactions WHERE user_id = ?
     ORDER BY created_at DESC LIMIT ? OFFSET ?`
  ).bind(user.id, limit, offset).all();

  return jsonResponse({
    transactions: transactions.results,
    pagination: {
      page,
      limit,
      total: total.count,
      pages: Math.ceil(total.count / limit),
    },
  }, 200, cors);
}

async function handleCreditCheck(request, user, cors) {
  const { action } = await request.json();
  const cost = CREDIT_COSTS[action];

  if (cost === undefined) {
    return jsonResponse({ error: 'Unknown action' }, 400, cors);
  }

  const sufficient = user.credits >= cost;

  return jsonResponse({
    action,
    cost,
    available: user.credits,
    sufficient,
  }, 200, cors);
}

async function deductCredits(env, userId, amount, action, description, projectId) {
  await env.DB.batch([
    env.DB.prepare('UPDATE users SET credits = credits - ?, updated_at = ? WHERE id = ?')
      .bind(amount, new Date().toISOString(), userId),
    env.DB.prepare(
      `INSERT INTO credit_transactions (user_id, project_id, amount, type, action, description, created_at)
       VALUES (?, ?, ?, 'usage', ?, ?, ?)`
    ).bind(userId, projectId || null, -amount, action, description, new Date().toISOString()),
  ]);
}

async function refundCredits(env, userId, amount, action, description, projectId) {
  await env.DB.batch([
    env.DB.prepare('UPDATE users SET credits = credits + ?, updated_at = ? WHERE id = ?')
      .bind(amount, new Date().toISOString(), userId),
    env.DB.prepare(
      `INSERT INTO credit_transactions (user_id, project_id, amount, type, action, description, created_at)
       VALUES (?, ?, ?, 'refund', ?, ?, ?)`
    ).bind(userId, projectId || null, amount, action, description, new Date().toISOString()),
  ]);
}

async function addCredits(env, userId, amount, type, action, description) {
  await env.DB.batch([
    env.DB.prepare('UPDATE users SET credits = credits + ?, updated_at = ? WHERE id = ?')
      .bind(amount, new Date().toISOString(), userId),
    env.DB.prepare(
      `INSERT INTO credit_transactions (user_id, amount, type, action, description, created_at)
       VALUES (?, ?, ?, ?, ?, ?)`
    ).bind(userId, amount, type, action, description, new Date().toISOString()),
  ]);
}

async function recordTransaction(env, userId, amount, type, action, description) {
  await env.DB.prepare(
    `INSERT INTO credit_transactions (user_id, amount, type, action, description, created_at)
     VALUES (?, ?, ?, ?, ?, ?)`
  ).bind(userId, amount, type, action, description, new Date().toISOString()).run();
}

// ==================== PortOne(포트원) V2 결제 ====================

const PORTONE_API_URL = 'https://api.portone.io';

async function handleGetPlans(cors) {
  return jsonResponse({ plans: PLANS, credit_packs: CREDIT_PACKS, costs: CREDIT_COSTS }, 200, cors);
}

/**
 * 결제 사전 준비 — D1에 pending 결제 생성, 프론트에 paymentId 반환
 * 프론트는 이 paymentId로 PortOne SDK 결제 팝업을 연다
 */
async function handlePaymentPrepare(request, user, env, cors) {
  const { type, plan_id, pack_type } = await request.json();

  const paymentId = `pay_${Date.now()}_${crypto.randomUUID().slice(0, 8)}`;
  let orderName, totalAmount, credits;

  if (type === 'credit_pack') {
    const pack = CREDIT_PACKS[pack_type];
    if (!pack) return jsonResponse({ error: 'Invalid pack type' }, 400, cors);

    orderName = `StoryCut 크레딧팩 ${pack_type} (${pack.credits}cr)`;
    totalAmount = pack.priceKrw;
    credits = pack.credits;

    // pending 결제 기록
    await env.DB.prepare(
      `INSERT INTO payments (id, user_id, amount_usd, credits, status, payment_type, created_at)
       VALUES (?, ?, ?, ?, 'pending', 'one_time', ?)`
    ).bind(paymentId, user.id, totalAmount, credits, new Date().toISOString()).run();

  } else if (type === 'subscription') {
    const plan = PLANS[plan_id];
    if (!plan || plan_id === 'free') return jsonResponse({ error: 'Invalid plan' }, 400, cors);

    orderName = `StoryCut ${plan.name} 구독 (월 ${plan.monthlyCredits}cr)`;
    totalAmount = plan.priceKrw;
    credits = plan.monthlyCredits;

    await env.DB.prepare(
      `INSERT INTO payments (id, user_id, amount_usd, credits, status, payment_type, created_at)
       VALUES (?, ?, ?, ?, 'pending', 'subscription', ?)`
    ).bind(paymentId, user.id, totalAmount, credits, new Date().toISOString()).run();

  } else {
    return jsonResponse({ error: 'Invalid type (credit_pack or subscription)' }, 400, cors);
  }

  return jsonResponse({
    payment_id: paymentId,
    order_name: orderName,
    total_amount: totalAmount,
    currency: 'KRW',
    store_id: env.PORTONE_STORE_ID || '',
    channel_key: env.PORTONE_CHANNEL_KEY || '',
  }, 200, cors);
}

/**
 * 크레딧팩 결제 완료 검증 — 프론트에서 SDK 결제 후 호출
 * PortOne API로 결제 상태 확인 → 크레딧 충전
 */
async function handlePaymentComplete(request, user, env, cors) {
  const { payment_id, pack_type } = await request.json();

  if (!payment_id) return jsonResponse({ error: 'payment_id required' }, 400, cors);

  // 1) pending 결제 확인
  const payment = await env.DB.prepare(
    'SELECT * FROM payments WHERE id = ? AND user_id = ? AND status = ?'
  ).bind(payment_id, user.id, 'pending').first();

  if (!payment) return jsonResponse({ error: 'Payment not found or already processed' }, 404, cors);

  // 2) 포트원 API로 결제 상태 검증
  const verified = await verifyPortonePayment(env, payment_id, payment.amount_usd);
  if (!verified.success) {
    return jsonResponse({ error: verified.error || 'Payment verification failed' }, 400, cors);
  }

  // 3) 크레딧 충전
  const credits = payment.credits;
  await addCredits(env, user.id, credits, 'purchase', 'credit_pack',
    `${pack_type || 'pack'} (${credits}cr) - ${payment.amount_usd.toLocaleString()}원`);

  // 4) 결제 완료 처리
  await env.DB.prepare(
    `UPDATE payments SET status = 'completed', stripe_payment_id = ?, completed_at = ? WHERE id = ?`
  ).bind(verified.portone_payment_id || payment_id, new Date().toISOString(), payment_id).run();

  // 5) credit_packs 기록
  if (pack_type) {
    await env.DB.prepare(
      `INSERT INTO credit_packs (user_id, pack_type, credits, amount_usd, stripe_payment_id, created_at)
       VALUES (?, ?, ?, ?, ?, ?)`
    ).bind(user.id, pack_type, credits, payment.amount_usd, payment_id, new Date().toISOString()).run();
  }

  // 최신 잔액 조회
  const freshUser = await env.DB.prepare('SELECT credits FROM users WHERE id = ?').bind(user.id).first();

  return jsonResponse({
    message: `${credits} credits added`,
    credits_added: credits,
    credits_total: freshUser.credits,
  }, 200, cors);
}

/**
 * 정기결제(구독) — 빌링키 발급 후 호출
 * 프론트에서 PortOne SDK로 빌링키 발급 → 여기서 첫 결제 + 구독 등록
 */
async function handleSubscribe(request, user, env, cors) {
  const { billing_key, plan_id, payment_id } = await request.json();

  if (!billing_key || !plan_id) {
    return jsonResponse({ error: 'billing_key and plan_id required' }, 400, cors);
  }

  const plan = PLANS[plan_id];
  if (!plan || plan_id === 'free') return jsonResponse({ error: 'Invalid plan' }, 400, cors);

  // 기존 구독 확인
  if (user.plan_id && user.plan_id !== 'free') {
    return jsonResponse({ error: 'Already subscribed. Cancel current plan first.' }, 400, cors);
  }

  // 빌링키로 첫 결제 실행
  const firstPaymentId = payment_id || `sub_first_${Date.now()}_${crypto.randomUUID().slice(0, 8)}`;

  const chargeResult = await chargeWithBillingKey(env, billing_key, firstPaymentId, plan.priceKrw,
    `StoryCut ${plan.name} 구독 (첫 달)`);

  if (!chargeResult.success) {
    return jsonResponse({ error: `Payment failed: ${chargeResult.error}` }, 402, cors);
  }

  // 구독 등록 + 크레딧 충전
  const now = new Date();
  const nextMonth = new Date(now);
  nextMonth.setMonth(nextMonth.getMonth() + 1);

  await env.DB.batch([
    env.DB.prepare(
      `UPDATE users SET plan_id = ?, monthly_credits = ?, credits = credits + ?,
       stripe_subscription_id = ?, plan_expires_at = ?, updated_at = ? WHERE id = ?`
    ).bind(plan_id, plan.monthlyCredits, plan.monthlyCredits, billing_key,
      nextMonth.toISOString(), now.toISOString(), user.id),
    env.DB.prepare(
      `INSERT INTO subscriptions (user_id, plan_id, stripe_subscription_id, stripe_price_id, status,
       current_period_start, current_period_end, created_at, updated_at)
       VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?)`
    ).bind(user.id, plan_id, billing_key, String(plan.priceKrw),
      now.toISOString(), nextMonth.toISOString(), now.toISOString(), now.toISOString()),
  ]);

  await addCredits(env, user.id, plan.monthlyCredits, 'subscription', 'plan_renewal',
    `${plan.name} 구독 시작 (${plan.monthlyCredits}cr)`);

  // 결제 기록
  await env.DB.prepare(
    `INSERT INTO payments (id, user_id, amount_usd, credits, status, payment_type, stripe_payment_id, created_at, completed_at)
     VALUES (?, ?, ?, ?, 'completed', 'subscription', ?, ?, ?)`
  ).bind(firstPaymentId, user.id, plan.priceKrw, plan.monthlyCredits,
    billing_key, now.toISOString(), now.toISOString()).run();

  const freshUser = await env.DB.prepare('SELECT credits FROM users WHERE id = ?').bind(user.id).first();

  return jsonResponse({
    message: `${plan.name} 구독 시작! ${plan.monthlyCredits}cr 충전됨`,
    plan_id: plan_id,
    plan_name: plan.name,
    credits_added: plan.monthlyCredits,
    credits_total: freshUser.credits,
    next_renewal: nextMonth.toISOString(),
  }, 200, cors);
}

/**
 * 구독 취소
 */
async function handleCancelSubscription(user, env, cors) {
  if (!user.plan_id || user.plan_id === 'free') {
    return jsonResponse({ error: 'No active subscription' }, 400, cors);
  }

  const now = new Date().toISOString();

  await env.DB.batch([
    env.DB.prepare(
      `UPDATE users SET plan_id = 'free', stripe_subscription_id = NULL,
       monthly_credits = 0, updated_at = ? WHERE id = ?`
    ).bind(now, user.id),
    env.DB.prepare(
      `UPDATE subscriptions SET status = 'cancelled', cancelled_at = ?, updated_at = ?
       WHERE user_id = ? AND status = 'active'`
    ).bind(now, now, user.id),
  ]);

  return jsonResponse({ message: 'Subscription cancelled' }, 200, cors);
}

/**
 * PortOne Webhook (결제 알림 — 백업용)
 */
async function handlePortoneWebhook(request, env, cors) {
  try {
    const body = await request.json();

    // Transaction.Paid 이벤트만 처리
    if (body.type === 'Transaction.Paid') {
      const paymentId = body.data?.paymentId;
      if (paymentId) {
        // 이미 complete 처리된 결제인지 확인
        const payment = await env.DB.prepare(
          'SELECT * FROM payments WHERE id = ? AND status = ?'
        ).bind(paymentId, 'pending').first();

        if (payment) {
          // complete 처리 (프론트에서 complete 호출 못한 경우 보충)
          await addCredits(env, payment.user_id, payment.credits, 'purchase', 'credit_pack',
            `Webhook: ${payment.credits}cr - ${payment.amount_usd}원`);

          await env.DB.prepare(
            `UPDATE payments SET status = 'completed', completed_at = ? WHERE id = ?`
          ).bind(new Date().toISOString(), paymentId).run();
        }
      }
    }

    return jsonResponse({ received: true }, 200, cors);
  } catch (error) {
    console.error('Webhook error:', error);
    return jsonResponse({ error: 'Webhook processing failed' }, 500, cors);
  }
}

// ==================== PortOne API 유틸리티 ====================

/**
 * 포트원 결제 검증
 */
async function verifyPortonePayment(env, paymentId, expectedAmount) {
  const apiSecret = env.PORTONE_API_SECRET;
  if (!apiSecret) {
    // 포트원 미설정 시 검증 스킵 (개발용)
    console.warn('PORTONE_API_SECRET not set, skipping verification');
    return { success: true, portone_payment_id: paymentId };
  }

  try {
    const response = await fetch(`${PORTONE_API_URL}/payments/${encodeURIComponent(paymentId)}`, {
      headers: { 'Authorization': `PortOne ${apiSecret}` },
    });

    if (!response.ok) {
      const err = await response.text();
      return { success: false, error: `PortOne API error: ${response.status} ${err}` };
    }

    const data = await response.json();

    // 상태 확인
    if (data.status !== 'PAID') {
      return { success: false, error: `Payment status: ${data.status}` };
    }

    // 금액 확인
    if (data.amount?.total !== expectedAmount) {
      return { success: false, error: `Amount mismatch: expected ${expectedAmount}, got ${data.amount?.total}` };
    }

    return { success: true, portone_payment_id: data.id || paymentId };
  } catch (error) {
    return { success: false, error: `Verification failed: ${error.message}` };
  }
}

/**
 * 빌링키로 결제 (정기결제용)
 */
async function chargeWithBillingKey(env, billingKey, paymentId, amount, orderName) {
  const apiSecret = env.PORTONE_API_SECRET;
  if (!apiSecret) {
    console.warn('PORTONE_API_SECRET not set, skipping billing charge');
    return { success: true };
  }

  try {
    const response = await fetch(`${PORTONE_API_URL}/payments/${encodeURIComponent(paymentId)}/billing-key`, {
      method: 'POST',
      headers: {
        'Authorization': `PortOne ${apiSecret}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        billingKey: billingKey,
        orderName: orderName,
        amount: { total: amount },
        currency: 'KRW',
      }),
    });

    if (!response.ok) {
      const err = await response.text();
      return { success: false, error: `Billing charge failed: ${response.status} ${err}` };
    }

    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

// ==================== 정기결제 자동 갱신 (Cron) ====================

async function processSubscriptionRenewals(env) {
  const now = new Date();

  // 만료된 구독 조회
  const expiredSubs = await env.DB.prepare(
    `SELECT s.id, s.user_id, s.plan_id, s.stripe_subscription_id as billing_key,
            s.stripe_price_id as price_krw
     FROM subscriptions s
     JOIN users u ON s.user_id = u.id
     WHERE s.status = 'active' AND u.plan_expires_at <= ?`
  ).bind(now.toISOString()).all();

  for (const sub of expiredSubs.results || []) {
    const plan = PLANS[sub.plan_id];
    if (!plan || !sub.billing_key) continue;

    const paymentId = `renewal_${Date.now()}_${crypto.randomUUID().slice(0, 8)}`;

    try {
      const result = await chargeWithBillingKey(env, sub.billing_key, paymentId,
        plan.priceKrw, `StoryCut ${plan.name} 월간 갱신`);

      if (result.success) {
        const nextMonth = new Date(now);
        nextMonth.setMonth(nextMonth.getMonth() + 1);

        await addCredits(env, sub.user_id, plan.monthlyCredits, 'subscription', 'plan_renewal',
          `${plan.name} 월간 갱신 (${plan.monthlyCredits}cr)`);

        await env.DB.batch([
          env.DB.prepare(
            'UPDATE users SET plan_expires_at = ?, updated_at = ? WHERE id = ?'
          ).bind(nextMonth.toISOString(), now.toISOString(), sub.user_id),
          env.DB.prepare(
            `UPDATE subscriptions SET current_period_start = ?, current_period_end = ?, updated_at = ?
             WHERE id = ?`
          ).bind(now.toISOString(), nextMonth.toISOString(), now.toISOString(), sub.id),
        ]);

        await env.DB.prepare(
          `INSERT INTO payments (id, user_id, amount_usd, credits, status, payment_type, stripe_payment_id, created_at, completed_at)
           VALUES (?, ?, ?, ?, 'completed', 'subscription', ?, ?, ?)`
        ).bind(paymentId, sub.user_id, plan.priceKrw, plan.monthlyCredits,
          sub.billing_key, now.toISOString(), now.toISOString()).run();

        console.log(`Renewal success: user=${sub.user_id}, plan=${sub.plan_id}`);
      } else {
        // 결제 실패 — 구독 만료
        console.error(`Renewal failed: user=${sub.user_id}, error=${result.error}`);

        await env.DB.batch([
          env.DB.prepare(
            `UPDATE users SET plan_id = 'free', stripe_subscription_id = NULL,
             monthly_credits = 0, updated_at = ? WHERE id = ?`
          ).bind(now.toISOString(), sub.user_id),
          env.DB.prepare(
            `UPDATE subscriptions SET status = 'expired', updated_at = ? WHERE id = ?`
          ).bind(now.toISOString(), sub.id),
        ]);
      }
    } catch (error) {
      console.error(`Renewal error: user=${sub.user_id}`, error);
    }
  }
}

// ==================== 관리자 API ====================

async function handleAdminGrantCredits(request, user, env, cors) {
  const adminEmails = (env.ADMIN_EMAILS || '').split(',').map(e => e.trim());
  if (!adminEmails.includes(user.email)) {
    return jsonResponse({ error: 'Forbidden' }, 403, cors);
  }

  const { target_user_id, target_email, amount, reason } = await request.json();

  let targetId = target_user_id;
  if (!targetId && target_email) {
    const target = await env.DB.prepare('SELECT id FROM users WHERE email = ?').bind(target_email).first();
    if (!target) return jsonResponse({ error: 'Target user not found' }, 404, cors);
    targetId = target.id;
  }

  if (!targetId || !amount) {
    return jsonResponse({ error: 'target_user_id/target_email and amount required' }, 400, cors);
  }

  await addCredits(env, targetId, amount, 'purchase', 'admin_grant', reason || `Admin grant by ${user.email}`);

  return jsonResponse({ message: `Granted ${amount} credits`, target_user_id: targetId }, 200, cors);
}

// ==================== 기존 핸들러 ====================

async function handleHealth(env, cors) {
  return jsonResponse({
    status: 'ok',
    version: '3.0-portone',
    timestamp: new Date().toISOString(),
  }, 200, cors);
}

async function handleGenerate(request, user, env, ctx, cors) {
  try {
    const body = await request.json();

    const creditAction = body.mode === 'mv' ? 'mv' : 'video';
    const cost = CREDIT_COSTS[creditAction];

    if (user.credits < cost) {
      return jsonResponse({
        error: 'Insufficient credits',
        required: cost,
        available: user.credits,
        action: creditAction,
      }, 402, cors);
    }

    const projectId = generateProjectId();

    await env.DB.prepare(
      `INSERT INTO projects (id, user_id, status, input_data, created_at)
       VALUES (?, ?, ?, ?, ?)`
    ).bind(projectId, user.id, 'queued', JSON.stringify(body), new Date().toISOString()).run();

    if (env.VIDEO_QUEUE) {
      await env.VIDEO_QUEUE.send({
        projectId,
        userId: user.id,
        input: body,
        timestamp: Date.now(),
      });
    }

    await deductCredits(env, user.id, cost, creditAction, `${creditAction} generation`, projectId);

    return jsonResponse({
      project_id: projectId,
      status: 'queued',
      message: 'Generation started',
      credits_used: cost,
      credits_remaining: user.credits - cost,
    }, 200, cors);
  } catch (error) {
    console.error('Generate error:', error);
    return jsonResponse({ error: error.message }, 500, cors);
  }
}

async function handleVideoDownload(url, env, cors) {
  const projectId = url.pathname.split('/').pop();

  try {
    if (!env.R2_BUCKET) {
      return jsonResponse({ error: 'Storage not configured' }, 503, cors);
    }
    const object = await env.R2_BUCKET.get(`videos/${projectId}/final_video.mp4`);

    if (!object) {
      return new Response('Video not found', { status: 404, headers: cors });
    }

    return new Response(object.body, {
      headers: {
        ...cors,
        'Content-Type': 'video/mp4',
        'Content-Disposition': `attachment; filename="storycut_${projectId}.mp4"`,
        'Cache-Control': 'public, max-age=31536000',
      },
    });
  } catch (error) {
    console.error('Download error:', error);
    return new Response('Error downloading video', { status: 500, headers: cors });
  }
}

async function handleStatus(url, env, cors) {
  const projectId = url.pathname.split('/').pop();

  try {
    const result = await env.DB.prepare(
      `SELECT id, status, created_at, completed_at, error_message
       FROM projects WHERE id = ?`
    ).bind(projectId).first();

    if (!result) {
      return jsonResponse({ error: 'Project not found' }, 404, cors);
    }

    return jsonResponse(result, 200, cors);
  } catch (error) {
    console.error('Status error:', error);
    return jsonResponse({ error: error.message }, 500, cors);
  }
}

// ==================== 유틸리티 ====================

function jsonResponse(data, status, cors) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...cors, 'Content-Type': 'application/json' },
  });
}

function generateProjectId() {
  return crypto.randomUUID().replace(/-/g, '').substring(0, 12);
}
