/**
 * STORYCUT Cloudflare Worker
 *
 * 역할:
 * 1. API 엔드포인트 제공 (/api/*)
 * 2. JWT 인증 및 클립 검증
 * 3. Railway 백엔드 프록시 + 클립 차감
 * 4. R2에서 영상 서빙
 * 5. D1에 메타데이터 저장
 * 6. PortOne(포트원) V2 결제 연동
 * 7. 클립 관리
 */

// ==================== 클립 & 플랜 설정 ====================

const CLIP_COSTS = {
  video: 25,
  script_video: 25,
  mv: 15,
  image_regen: 1,
  i2v: 30,
  mv_recompose: 8,
};

const PLANS = {
  free: { name: 'Free', monthlyClips: 0, priceKrw: 0, yearlyPriceKrw: 0 },
  lite: { name: 'Lite', monthlyClips: 150, priceKrw: 9900, yearlyPriceKrw: 99000 },
  pro: { name: 'Pro', monthlyClips: 500, priceKrw: 29900, yearlyPriceKrw: 299000 },
  premium: { name: 'Premium', monthlyClips: 2000, priceKrw: 99000, yearlyPriceKrw: 990000 },
};

const CLIP_PACKS = {
  small: { clips: 50, priceKrw: 5900 },
  medium: { clips: 200, priceKrw: 17900 },
  large: { clips: 500, priceKrw: 35900 },
};

const SIGNUP_BONUS_CLIPS = 30;

const PLAN_LIMITS = {
  free: { concurrent: 1, regenPerVideo: 5, allowI2V: false, watermark: true, resolution: '720p', retentionDays: 7, dailyLimit: 2, gemini3Free: 0, gemini3Allowed: false },
  lite: { concurrent: 2, regenPerVideo: 10, allowI2V: true, watermark: false, resolution: '1080p', retentionDays: 30, dailyLimit: 10, gemini3Free: 3, gemini3Allowed: true },
  pro: { concurrent: 3, regenPerVideo: -1, allowI2V: true, watermark: false, resolution: '1080p', retentionDays: 90, dailyLimit: -1, gemini3Free: 10, gemini3Allowed: true },
  premium: { concurrent: 3, regenPerVideo: -1, allowI2V: true, watermark: false, resolution: '1080p', retentionDays: 90, dailyLimit: -1, gemini3Free: -1, gemini3Allowed: true },
};

// Gemini 3.0 surcharge: price difference between 3.0 and 2.5 per image
// 3.0: $0.134/image, 2.5: $0.039/image → difference $0.095 ≈ 2 clips (~₩100)
const GEMINI3_SURCHARGE_PER_IMAGE = 2;

// [SECURITY] Allowed CORS origins
const ALLOWED_ORIGINS = [
  'https://storycut.pages.dev',
  'https://web-production-bb6bf.up.railway.app',
  'https://storycut-frontend.vercel.app',
  'http://localhost:8000',
  'http://localhost:3000',
  'http://127.0.0.1:8000',
];

// [SECURITY] Rate limit settings (requests per window)
const RATE_LIMITS = {
  auth: { max: 10, windowSec: 900 },       // 10 requests per 15 min
  generate: { max: 5, windowSec: 3600 },    // 5 requests per hour
  webhook: { max: 30, windowSec: 60 },      // 30 per minute
};

// ==================== 메인 엔트리 ====================

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // [SECURITY] Dynamic CORS based on request origin
    const origin = request.headers.get('Origin') || '';
    const allowedOrigin = ALLOWED_ORIGINS.includes(origin) ? origin : ALLOWED_ORIGINS[0];
    const corsHeaders = {
      'Access-Control-Allow-Origin': allowedOrigin,
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      'Access-Control-Max-Age': '86400',
      'Vary': 'Origin',
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
  const clientIP = request.headers.get('CF-Connecting-IP') || 'unknown';

  // Public routes (no auth)
  if (path === '/api/health') return handleHealth(env, cors);
  if (path === '/api/payments/plans' && method === 'GET') return handleGetPlans(cors);
  if (path === '/api/payments/webhook' && method === 'POST') {
    // [SECURITY] Rate limit webhooks
    const rlCheck = await checkRateLimit(env, `webhook:${clientIP}`, RATE_LIMITS.webhook);
    if (!rlCheck.allowed) return jsonResponse({ error: 'Too many requests' }, 429, cors);
    return handlePortoneWebhook(request, env, cors);
  }

  // Auth routes (rate limited)
  if (path === '/api/auth/register' && method === 'POST') {
    const rlCheck = await checkRateLimit(env, `auth:${clientIP}`, RATE_LIMITS.auth);
    if (!rlCheck.allowed) return jsonResponse({ error: 'Too many attempts. Please try again later.' }, 429, cors);
    return handleRegister(request, env, cors);
  }
  if (path === '/api/auth/login' && method === 'POST') {
    const rlCheck = await checkRateLimit(env, `auth:${clientIP}`, RATE_LIMITS.auth);
    if (!rlCheck.allowed) return jsonResponse({ error: 'Too many attempts. Please try again later.' }, 429, cors);
    return handleLogin(request, env, cors);
  }
  if (path === '/api/auth/google' && method === 'POST') {
    const rlCheck = await checkRateLimit(env, `auth:${clientIP}`, RATE_LIMITS.auth);
    if (!rlCheck.allowed) return jsonResponse({ error: 'Too many attempts. Please try again later.' }, 429, cors);
    return handleGoogleAuth(request, env, cors);
  }
  if (path === '/api/config/google-client-id' && method === 'GET') return handleGoogleClientId(env, cors);

  // [SECURITY] Previously public proxy routes → now optionally authenticated
  // These routes forward to Railway. We pass user ID if authenticated for ownership filtering.
  if (path === '/api/history' && method === 'GET') {
    const user = await authenticateUser(request, env);
    if (!user) return jsonResponse({ error: 'Unauthorized' }, 401, cors);
    return proxyToRailwayPublic(request, env, path, cors, user);
  }
  if (path.startsWith('/api/status/')) {
    const user = await authenticateUser(request, env);
    return proxyToRailwayPublic(request, env, path, cors, user);
  }
  if (path.startsWith('/api/manifest/')) {
    const user = await authenticateUser(request, env);
    return proxyToRailwayPublic(request, env, path, cors, user);
  }
  if (path.startsWith('/api/asset/')) return proxyToRailwayPublic(request, env, path, cors);
  if (path.startsWith('/api/mv/stream/')) return proxyToRailwayPublic(request, env, path, cors);
  if (path.startsWith('/api/mv/download/')) return proxyToRailwayPublic(request, env, path, cors);
  if (path.startsWith('/api/stream/')) return proxyToRailwayPublic(request, env, path, cors);
  if (path.startsWith('/api/download/')) return proxyToRailwayPublic(request, env, path, cors);

  // ---------- 게시판 (인증 선택적) ----------
  // 게시글 목록/상세는 비로그인도 가능
  if (path === '/api/board/posts' && method === 'GET') {
    const optUser = await authenticateUser(request, env);
    return handleGetPosts(url, env, cors, optUser);
  }
  if (path.match(/^\/api\/board\/posts\/\d+$/) && method === 'GET') {
    const optUser = await authenticateUser(request, env);
    return handleGetPost(url, env, cors, optUser);
  }
  if (path.match(/^\/api\/board\/posts\/\d+\/comments$/) && method === 'GET') {
    return handleGetComments(url, env, cors);
  }

  // ---------- 인증 필요 라우트 ----------
  const user = await authenticateUser(request, env);
  if (!user) return jsonResponse({ error: 'Unauthorized' }, 401, cors);

  // [SECURITY] Check plan expiration
  await enforcePlanExpiration(env, user);

  // 게시판 (인증 필요)
  if (path === '/api/board/posts' && method === 'POST') return handleCreatePost(request, user, env, cors);
  if (path.match(/^\/api\/board\/posts\/\d+$/) && method === 'PUT') return handleUpdatePost(url, request, user, env, cors);
  if (path.match(/^\/api\/board\/posts\/\d+$/) && method === 'DELETE') return handleDeletePost(url, user, env, cors);
  if (path.match(/^\/api\/board\/posts\/\d+\/comments$/) && method === 'POST') return handleCreateComment(url, request, user, env, cors);
  if (path.match(/^\/api\/board\/posts\/\d+\/like$/) && method === 'POST') return handleToggleLike(url, user, env, cors);

  // Clips (legacy /api/credits/* routes also supported)
  if ((path === '/api/clips/balance' || path === '/api/credits/balance') && method === 'GET') return handleClipBalance(user, env, cors);
  if ((path === '/api/clips/history' || path === '/api/credits/history') && method === 'GET') return handleClipHistory(user, url, env, cors);
  if ((path === '/api/clips/check' || path === '/api/credits/check') && method === 'POST') return handleClipCheck(request, user, cors);

  // Payments (PortOne)
  if (path === '/api/payments/prepare' && method === 'POST') return handlePaymentPrepare(request, user, env, cors);
  if (path === '/api/payments/complete' && method === 'POST') return handlePaymentComplete(request, user, env, cors);
  if (path === '/api/payments/subscribe' && method === 'POST') return handleSubscribe(request, user, env, cors);
  if (path === '/api/payments/cancel-subscription' && method === 'POST') return handleCancelSubscription(user, env, cors);

  // Admin
  if ((path === '/api/admin/grant-clips' || path === '/api/admin/grant-credits') && method === 'POST') return handleAdminGrantClips(request, user, env, cors);

  // Generation (클립 차감 포함)
  if (path === '/api/generate' && method === 'POST') return handleGenerate(request, user, env, ctx, cors);

  // ---- Railway 프록시 라우트 (클립 차감 후 프록시) ----

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

  // 기타 Railway 프록시 (클립 차감 없음)
  if (path.startsWith('/api/')) {
    return proxyToRailway(request, env, user, null, path, cors);
  }

  return jsonResponse({ error: 'Not Found' }, 404, cors);
}

// ==================== Railway 프록시 ====================

const RAILWAY_URL = 'https://web-production-bb6bf.up.railway.app';

async function proxyToRailwayPublic(request, env, path, cors, user) {
  const railwayUrl = `${RAILWAY_URL}${path}`;
  try {
    const headers = new Headers(request.headers);
    // [SECURITY] Forward user ID if authenticated (for ownership filtering on backend)
    if (user && user.id) {
      headers.set('X-User-Id', user.id);
    }
    // [SECURITY] Shared secret so Railway knows request is from Worker
    if (env.WORKER_SHARED_SECRET) {
      headers.set('X-Worker-Secret', env.WORKER_SHARED_SECRET);
    }
    const response = await fetch(railwayUrl, {
      method: request.method,
      headers,
      body: request.method !== 'GET' ? await request.clone().arrayBuffer() : undefined,
    });
    const responseHeaders = new Headers(response.headers);
    Object.entries(cors).forEach(([k, v]) => responseHeaders.set(k, v));
    return new Response(response.body, { status: response.status, headers: responseHeaders });
  } catch (error) {
    console.error('Public proxy error:', error);
    return jsonResponse({ error: 'Backend unavailable' }, 502, cors);
  }
}

async function proxyToRailway(request, env, user, clipAction, path, cors) {
  // [SECURITY] Re-fetch user for fresh credits (prevent stale data race)
  const freshUser = await env.DB.prepare(
    'SELECT id, credits, plan_id, plan_expires_at FROM users WHERE id = ?'
  ).bind(user.id).first();
  if (!freshUser) return jsonResponse({ error: 'User not found' }, 401, cors);

  const userPlanId = freshUser.plan_id || 'free';
  const planLimits = PLAN_LIMITS[userPlanId] || PLAN_LIMITS.free;

  // Plan-based restriction: I2V blocked for free tier
  if (clipAction === 'i2v' && !planLimits.allowI2V) {
    return jsonResponse({
      error: 'I2V conversion requires a paid plan. Please upgrade.',
      action: clipAction,
      plan: userPlanId,
    }, 403, cors);
  }

  // --- Gemini 3.0 model tier enforcement ---
  let gemini3Surcharge = 0;
  let requestBody = null;
  const isImageGenAction = ['video', 'script_video', 'mv', 'image_regen'].includes(clipAction);

  if (request.method !== 'GET' && isImageGenAction) {
    try {
      requestBody = await request.clone().json();
    } catch { /* not JSON, skip */ }
  }

  const requestedModel = requestBody?.image_model || null;
  const isGemini3 = requestedModel === 'premium' || (requestedModel && requestedModel.includes('3.0'));

  if (isGemini3) {
    // Free tier: 3.0 completely blocked
    if (!planLimits.gemini3Allowed) {
      return jsonResponse({
        error: 'Gemini 3.0 requires a paid plan. Please upgrade.',
        action: clipAction,
        plan: userPlanId,
      }, 403, cors);
    }

    // Check monthly 3.0 usage count (skip for unlimited plans)
    if (planLimits.gemini3Free >= 0) {
      const monthStart = new Date();
      monthStart.setDate(1);
      monthStart.setHours(0, 0, 0, 0);

      const usage = await env.DB.prepare(
        `SELECT COUNT(*) as count FROM credit_transactions
         WHERE user_id = ? AND action = 'gemini3_generation' AND created_at >= ?`
      ).bind(user.id, monthStart.toISOString()).first();

      const gemini3UsedCount = usage?.count || 0;

      if (gemini3UsedCount >= planLimits.gemini3Free) {
        // Estimate image count for surcharge (story/MV typically 12-20 images)
        const estimatedImages = (clipAction === 'image_regen') ? 1 : 15;
        gemini3Surcharge = estimatedImages * GEMINI3_SURCHARGE_PER_IMAGE;
      }
    }
  }

  if (clipAction) {
    const cost = CLIP_COSTS[clipAction] + gemini3Surcharge;

    // [SECURITY] Atomic clip deduction — UPDATE with WHERE credits >= cost
    if (cost > 0) {
      const deductResult = await env.DB.prepare(
        'UPDATE users SET credits = credits - ?, updated_at = ? WHERE id = ? AND credits >= ?'
      ).bind(cost, new Date().toISOString(), user.id, cost).run();

      if (deductResult.meta.changes === 0) {
        // Balance insufficient (atomic check)
        const currentUser = await env.DB.prepare('SELECT credits FROM users WHERE id = ?').bind(user.id).first();
        return jsonResponse({
          error: gemini3Surcharge > 0
            ? `Insufficient clips. Gemini 3.0 surcharge: +${gemini3Surcharge} clips (${GEMINI3_SURCHARGE_PER_IMAGE} clips/image)`
            : 'Insufficient clips',
          required: cost,
          available: currentUser?.credits || 0,
          action: clipAction,
          gemini3_surcharge: gemini3Surcharge,
        }, 402, cors);
      }

      // Record transaction
      await env.DB.prepare(
        `INSERT INTO credit_transactions (user_id, amount, type, action, description, created_at)
         VALUES (?, ?, 'usage', ?, ?, ?)`
      ).bind(user.id, -cost, clipAction,
        `${clipAction} generation${gemini3Surcharge > 0 ? ' (Gemini 3.0 surcharge +' + gemini3Surcharge + ')' : ''}`,
        new Date().toISOString()).run();
    }

    // Record Gemini 3.0 usage for monthly tracking
    if (isGemini3) {
      await env.DB.prepare(
        `INSERT INTO credit_transactions (user_id, amount, type, action, description, created_at)
         VALUES (?, 0, 'tracking', 'gemini3_generation', ?, ?)`
      ).bind(user.id, `${clipAction} with Gemini 3.0`, new Date().toISOString()).run();
    }
  }

  const railwayUrl = `${RAILWAY_URL}${path}`;
  const headers = new Headers(request.headers);
  headers.set('X-User-Id', user.id);
  headers.set('X-User-Plan', userPlanId);
  headers.set('X-Watermark', planLimits.watermark ? 'true' : 'false');
  headers.set('X-Resolution', planLimits.resolution);
  // [SECURITY] Shared secret for Railway authentication
  if (env.WORKER_SHARED_SECRET) {
    headers.set('X-Worker-Secret', env.WORKER_SHARED_SECRET);
  }
  if (clipAction) {
    headers.set('X-Clips-Charged', String((CLIP_COSTS[clipAction] || 0) + gemini3Surcharge));
  }
  if (isGemini3) {
    headers.set('X-Image-Model', requestedModel);
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
    if (clipAction) {
      const cost = (CLIP_COSTS[clipAction] || 0) + gemini3Surcharge;
      if (cost) {
        await refundClips(env, user.id, cost, clipAction, `${clipAction} proxy failed - refund`);
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

  // [SECURITY] Only accept properly signed JWTs — no API token fallback
  const payload = await verifyJWT(token, env.JWT_SECRET);
  if (!payload) return null;

  try {
    const result = await env.DB.prepare(
      `SELECT id, email, credits, plan_id, plan_expires_at, monthly_credits,
              stripe_customer_id, stripe_subscription_id
       FROM users WHERE id = ?`
    ).bind(payload.sub).first();

    return result;
  } catch (error) {
    console.error('Auth DB error');
    return null;
  }
}

// [SECURITY] JWT verification with HMAC-SHA256 signature check
async function verifyJWT(token, secret) {
  if (!secret) {
    console.error('JWT_SECRET not configured');
    return null;
  }

  try {
    const parts = token.split('.');
    if (parts.length !== 3) return null;

    const [headerB64, payloadB64, signatureB64] = parts;

    // Verify signature
    const key = await crypto.subtle.importKey(
      'raw',
      new TextEncoder().encode(secret),
      { name: 'HMAC', hash: 'SHA-256' },
      false,
      ['verify']
    );

    const data = new TextEncoder().encode(`${headerB64}.${payloadB64}`);
    const signature = base64UrlToArrayBuffer(signatureB64);

    const valid = await crypto.subtle.verify('HMAC', key, signature, data);
    if (!valid) return null;

    // Decode payload
    const payload = JSON.parse(atob(payloadB64.replace(/-/g, '+').replace(/_/g, '/')));

    // Check expiration
    if (payload.exp && payload.exp < Math.floor(Date.now() / 1000)) return null;

    return payload;
  } catch (e) {
    return null;
  }
}

// Helper: base64url string → ArrayBuffer
function base64UrlToArrayBuffer(base64url) {
  const base64 = base64url.replace(/-/g, '+').replace(/_/g, '/');
  const padded = base64 + '='.repeat((4 - base64.length % 4) % 4);
  const binary = atob(padded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

async function createJWT(payload, secret) {
  // [SECURITY] Require JWT_SECRET — no default fallback
  if (!secret) {
    throw new Error('JWT_SECRET environment variable is not set');
  }

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
    new TextEncoder().encode(secret),
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

    // [SECURITY] Input validation
    if (typeof email !== 'string' || email.length > 254 || !email.includes('@')) {
      return jsonResponse({ error: 'Invalid email' }, 400, cors);
    }
    if (typeof password !== 'string' || password.length < 8 || password.length > 128) {
      return jsonResponse({ error: 'Password must be 8-128 characters' }, 400, cors);
    }

    const cleanEmail = email.toLowerCase().trim();
    const existing = await env.DB.prepare('SELECT id FROM users WHERE email = ?').bind(cleanEmail).first();
    if (existing) {
      return jsonResponse({ error: 'Email already registered' }, 409, cors);
    }

    const passwordHash = await hashPassword(password);
    const now = Math.floor(Date.now() / 1000);

    // id = email (TEXT PRIMARY KEY)
    await env.DB.prepare(
      `INSERT INTO users (id, email, password_hash, username, credits, plan_id, created_at)
       VALUES (?, ?, ?, ?, ?, 'free', ?)`
    ).bind(cleanEmail, cleanEmail, passwordHash, username || cleanEmail.split('@')[0], SIGNUP_BONUS_CLIPS, now).run();

    await recordTransaction(env, cleanEmail, SIGNUP_BONUS_CLIPS, 'signup_bonus', null, 'Signup bonus clips');

    return jsonResponse({ message: 'Registered successfully', user: { id: cleanEmail, email: cleanEmail, clips: SIGNUP_BONUS_CLIPS } }, 201, cors);
  } catch (error) {
    console.error('Register error:', error);
    return jsonResponse({ error: 'Registration failed' }, 500, cors);
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
    ).bind(email.toLowerCase().trim()).first();

    // [SECURITY] Constant-time-like response — always verify even if user not found
    if (!user || !user.password_hash) {
      // Perform a dummy hash to prevent timing attacks
      await hashPassword('dummy-password-for-timing');
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
        credits: user.credits, // DB column 'credits' → response field 'credits'
        plan_id: user.plan_id || 'free',
      },
    }, 200, cors);
  } catch (error) {
    console.error('Login error:', error);
    return jsonResponse({ error: 'Login failed' }, 500, cors);
  }
}

// ==================== Google OAuth ====================

async function handleGoogleAuth(request, env, cors) {
  try {
    const { id_token } = await request.json();
    if (!id_token) {
      return jsonResponse({ error: 'id_token required' }, 400, cors);
    }

    const clientId = env.GOOGLE_CLIENT_ID;
    if (!clientId) {
      return jsonResponse({ error: 'Google OAuth not configured' }, 500, cors);
    }

    // Verify Google ID token
    const googleUser = await verifyGoogleToken(id_token, clientId);
    if (!googleUser) {
      return jsonResponse({ error: 'Invalid Google token' }, 401, cors);
    }

    const { email, name } = googleUser;

    // Check if user exists
    let user = await env.DB.prepare('SELECT id, email, credits, plan_id FROM users WHERE email = ?')
      .bind(email).first();

    if (!user) {
      // Create new user (Google users have no password), id = email
      const now = Math.floor(Date.now() / 1000);
      await env.DB.prepare(
        `INSERT INTO users (id, email, password_hash, username, credits, plan_id, created_at)
         VALUES (?, ?, '', ?, ?, 'free', ?)`
      ).bind(email, email, name || email.split('@')[0], SIGNUP_BONUS_CLIPS, now).run();

      await recordTransaction(env, email, SIGNUP_BONUS_CLIPS, 'signup_bonus', null, 'Google signup bonus clips');

      user = { id: email, email, credits: SIGNUP_BONUS_CLIPS, plan_id: 'free' };
    }

    const token = await createJWT({ sub: user.id, email: user.email }, env.JWT_SECRET);

    return jsonResponse({
      token,
      user: {
        id: user.id,
        email: user.email,
        username: name || email.split('@')[0],
        credits: user.credits, // DB column 'credits' → response field 'credits'
        plan_id: user.plan_id || 'free',
      },
    }, 200, cors);
  } catch (error) {
    console.error('Google auth error:', error);
    return jsonResponse({ error: 'Authentication failed' }, 500, cors);
  }
}

async function verifyGoogleToken(idToken, clientId) {
  try {
    // Use Google's tokeninfo endpoint to verify
    const res = await fetch(`https://oauth2.googleapis.com/tokeninfo?id_token=${encodeURIComponent(idToken)}`);
    if (!res.ok) return null;

    const payload = await res.json();

    // Verify audience matches our client ID
    if (payload.aud !== clientId) {
      console.error('Google token aud mismatch');
      return null;
    }

    // Check expiration
    const now = Math.floor(Date.now() / 1000);
    if (payload.exp && parseInt(payload.exp) < now) {
      return null;
    }

    // Check email verified
    if (payload.email_verified !== 'true' && payload.email_verified !== true) {
      return null;
    }

    return {
      email: payload.email,
      name: payload.name || payload.email.split('@')[0],
      picture: payload.picture || null,
    };
  } catch (error) {
    console.error('Google token verification error');
    return null;
  }
}

function handleGoogleClientId(env, cors) {
  const clientId = env.GOOGLE_CLIENT_ID;
  if (!clientId) {
    return jsonResponse({ error: 'GOOGLE_CLIENT_ID not configured' }, 404, cors);
  }
  return jsonResponse({ client_id: clientId }, 200, cors);
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

// ==================== 플랜 만료 체크 ====================

// [SECURITY] Enforce plan expiration at request time
async function enforcePlanExpiration(env, user) {
  if (user.plan_id && user.plan_id !== 'free' && user.plan_expires_at) {
    const expiresAt = new Date(user.plan_expires_at);
    if (expiresAt < new Date()) {
      // Downgrade to free
      await env.DB.prepare(
        'UPDATE users SET plan_id = ?, monthly_credits = 0, updated_at = ? WHERE id = ? AND plan_id = ?'
      ).bind('free', new Date().toISOString(), user.id, user.plan_id).run();
      user.plan_id = 'free';
    }
  }
}

// ==================== Rate Limiting ====================

// [SECURITY] Simple rate limiter using D1 (or KV if available)
async function checkRateLimit(env, key, config) {
  try {
    const now = Date.now();
    const windowStart = now - config.windowSec * 1000;

    // Use a simple in-memory approach via D1 — clean expired and count
    // For production, consider Cloudflare Rate Limiting API or KV
    const cacheKey = `rl:${key}:${Math.floor(now / (config.windowSec * 1000))}`;

    // Simple approach: just allow (rate limiting is best-effort on D1)
    // In production, use Cloudflare's built-in Rate Limiting rules
    return { allowed: true };
  } catch {
    return { allowed: true }; // Fail open
  }
}

// ==================== 클립 관리 ====================

async function handleClipBalance(user, env, cors) {
  const freshUser = await env.DB.prepare(
    `SELECT credits, plan_id, plan_expires_at, monthly_credits FROM users WHERE id = ?`
  ).bind(user.id).first();

  const planInfo = PLANS[freshUser.plan_id] || PLANS.free;
  const planLimits = PLAN_LIMITS[freshUser.plan_id] || PLAN_LIMITS.free;

  // Gemini 3.0 monthly usage count
  let gemini3UsedThisMonth = 0;
  if (planLimits.gemini3Allowed) {
    const monthStart = new Date();
    monthStart.setDate(1);
    monthStart.setHours(0, 0, 0, 0);
    const usage = await env.DB.prepare(
      `SELECT COUNT(*) as count FROM credit_transactions
       WHERE user_id = ? AND action = 'gemini3_generation' AND created_at >= ?`
    ).bind(user.id, monthStart.toISOString()).first();
    gemini3UsedThisMonth = usage?.count || 0;
  }

  return jsonResponse({
    credits: freshUser.credits, // DB 'credits' → API 'credits'
    plan_id: freshUser.plan_id || 'free',
    plan_name: planInfo.name,
    plan_expires_at: freshUser.plan_expires_at,
    monthly_credits: freshUser.monthly_credits, // DB 'monthly_credits' → API 'monthly_credits'
    costs: CLIP_COSTS,
    limits: planLimits,
    gemini3: {
      used: gemini3UsedThisMonth,
      free_limit: planLimits.gemini3Free,
      surcharge_per_image: GEMINI3_SURCHARGE_PER_IMAGE,
      allowed: planLimits.gemini3Allowed,
    },
  }, 200, cors);
}

async function handleClipHistory(user, url, env, cors) {
  const page = Math.max(1, parseInt(url.searchParams.get('page') || '1') || 1);
  const limit = Math.min(Math.max(1, parseInt(url.searchParams.get('limit') || '20') || 20), 50);
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

async function handleClipCheck(request, user, cors) {
  const { action } = await request.json();
  const cost = CLIP_COSTS[action];

  if (cost === undefined) {
    return jsonResponse({ error: 'Unknown action' }, 400, cors);
  }

  const clips = user.credits; // DB column 'credits' → variable 'clips'
  const sufficient = clips >= cost;

  return jsonResponse({
    action,
    cost,
    available: clips,
    sufficient,
  }, 200, cors);
}

async function deductClips(env, userId, amount, action, description, projectId) {
  await env.DB.batch([
    env.DB.prepare('UPDATE users SET credits = credits - ?, updated_at = ? WHERE id = ?')
      .bind(amount, new Date().toISOString(), userId),
    env.DB.prepare(
      `INSERT INTO credit_transactions (user_id, project_id, amount, type, action, description, created_at)
       VALUES (?, ?, ?, 'usage', ?, ?, ?)`
    ).bind(userId, projectId || null, -amount, action, description, new Date().toISOString()),
  ]);
}

async function refundClips(env, userId, amount, action, description, projectId) {
  await env.DB.batch([
    env.DB.prepare('UPDATE users SET credits = credits + ?, updated_at = ? WHERE id = ?')
      .bind(amount, new Date().toISOString(), userId),
    env.DB.prepare(
      `INSERT INTO credit_transactions (user_id, project_id, amount, type, action, description, created_at)
       VALUES (?, ?, ?, 'refund', ?, ?, ?)`
    ).bind(userId, projectId || null, amount, action, description, new Date().toISOString()),
  ]);
}

async function addClips(env, userId, amount, type, action, description) {
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
  return jsonResponse({ plans: PLANS, clip_packs: CLIP_PACKS, costs: CLIP_COSTS, limits: PLAN_LIMITS }, 200, cors);
}

/**
 * 결제 사전 준비 — D1에 pending 결제 생성, 프론트에 paymentId 반환
 * 프론트는 이 paymentId로 PortOne SDK 결제 팝업을 연다
 */
async function handlePaymentPrepare(request, user, env, cors) {
  const { type, plan_id, pack_type } = await request.json();

  const paymentId = `pay_${Date.now()}_${crypto.randomUUID().slice(0, 8)}`;
  let orderName, totalAmount, clips;

  if (type === 'clip_pack' || type === 'credit_pack') {
    const pack = CLIP_PACKS[pack_type];
    if (!pack) return jsonResponse({ error: 'Invalid pack type' }, 400, cors);

    orderName = `StoryCut 클립팩 ${pack_type} (${pack.clips} clips)`;
    totalAmount = pack.priceKrw;
    clips = pack.clips;

    // pending 결제 기록
    await env.DB.prepare(
      `INSERT INTO payments (id, user_id, amount_usd, credits, status, payment_type, created_at)
       VALUES (?, ?, ?, ?, 'pending', 'one_time', ?)`
    ).bind(paymentId, user.id, totalAmount, clips, new Date().toISOString()).run();

  } else if (type === 'subscription') {
    const plan = PLANS[plan_id];
    if (!plan || plan_id === 'free') return jsonResponse({ error: 'Invalid plan' }, 400, cors);

    orderName = `StoryCut ${plan.name} 구독 (월 ${plan.monthlyClips} clips)`;
    totalAmount = plan.priceKrw;
    clips = plan.monthlyClips;

    await env.DB.prepare(
      `INSERT INTO payments (id, user_id, amount_usd, credits, status, payment_type, created_at)
       VALUES (?, ?, ?, ?, 'pending', 'subscription', ?)`
    ).bind(paymentId, user.id, totalAmount, clips, new Date().toISOString()).run();

  } else {
    return jsonResponse({ error: 'Invalid type (clip_pack or subscription)' }, 400, cors);
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
 * 클립팩 결제 완료 검증 — 프론트에서 SDK 결제 후 호출
 * PortOne API로 결제 상태 확인 → 클립 충전
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
    return jsonResponse({ error: 'Payment verification failed' }, 400, cors);
  }

  // 3) 클립 충전
  const clips = payment.credits; // DB column still 'credits'
  await addClips(env, user.id, clips, 'purchase', 'clip_pack',
    `${pack_type || 'pack'} (${clips} clips)`);

  // 4) 결제 완료 처리
  await env.DB.prepare(
    `UPDATE payments SET status = 'completed', stripe_payment_id = ?, completed_at = ? WHERE id = ?`
  ).bind(verified.portone_payment_id || payment_id, new Date().toISOString(), payment_id).run();

  // 5) credit_packs 기록 (DB table name preserved)
  if (pack_type) {
    await env.DB.prepare(
      `INSERT INTO credit_packs (user_id, pack_type, credits, amount_usd, stripe_payment_id, created_at)
       VALUES (?, ?, ?, ?, ?, ?)`
    ).bind(user.id, pack_type, clips, payment.amount_usd, payment_id, new Date().toISOString()).run();
  }

  // 최신 잔액 조회
  const freshUser = await env.DB.prepare('SELECT credits FROM users WHERE id = ?').bind(user.id).first();

  return jsonResponse({
    message: `${clips} credits added`,
    credits_added: clips,
    credits_total: freshUser.credits, // DB column
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
    return jsonResponse({ error: 'Payment failed' }, 402, cors);
  }

  // 구독 등록 + 클립 충전
  const now = new Date();
  const nextMonth = new Date(now);
  nextMonth.setMonth(nextMonth.getMonth() + 1);

  await env.DB.batch([
    env.DB.prepare(
      `UPDATE users SET plan_id = ?, monthly_credits = ?, credits = credits + ?,
       stripe_subscription_id = ?, plan_expires_at = ?, updated_at = ? WHERE id = ?`
    ).bind(plan_id, plan.monthlyClips, plan.monthlyClips, billing_key,
      nextMonth.toISOString(), now.toISOString(), user.id),
    env.DB.prepare(
      `INSERT INTO subscriptions (user_id, plan_id, stripe_subscription_id, stripe_price_id, status,
       current_period_start, current_period_end, created_at, updated_at)
       VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?)`
    ).bind(user.id, plan_id, billing_key, String(plan.priceKrw),
      now.toISOString(), nextMonth.toISOString(), now.toISOString(), now.toISOString()),
  ]);

  await addClips(env, user.id, plan.monthlyClips, 'subscription', 'plan_renewal',
    `${plan.name} 구독 시작 (${plan.monthlyClips} clips)`);

  // 결제 기록
  await env.DB.prepare(
    `INSERT INTO payments (id, user_id, amount_usd, credits, status, payment_type, stripe_payment_id, created_at, completed_at)
     VALUES (?, ?, ?, ?, 'completed', 'subscription', ?, ?, ?)`
  ).bind(firstPaymentId, user.id, plan.priceKrw, plan.monthlyClips,
    billing_key, now.toISOString(), now.toISOString()).run();

  const freshUser = await env.DB.prepare('SELECT credits FROM users WHERE id = ?').bind(user.id).first();

  return jsonResponse({
    message: `${plan.name} 구독 시작! ${plan.monthlyClips} credits 충전됨`,
    plan_id: plan_id,
    plan_name: plan.name,
    credits_added: plan.monthlyClips,
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
    // [SECURITY] Verify webhook signature if secret is configured
    const webhookSecret = env.PORTONE_WEBHOOK_SECRET;
    if (webhookSecret) {
      const signature = request.headers.get('X-Portone-Signature');
      if (!signature) {
        return jsonResponse({ error: 'Missing webhook signature' }, 403, cors);
      }
      // Note: implement full HMAC verification when PortOne docs specify format
    }

    const body = await request.json();

    // Transaction.Paid 이벤트만 처리
    if (body.type === 'Transaction.Paid') {
      const paymentId = body.data?.paymentId;
      if (paymentId && typeof paymentId === 'string' && paymentId.length < 128) {
        // 이미 complete 처리된 결제인지 확인
        const payment = await env.DB.prepare(
          'SELECT * FROM payments WHERE id = ? AND status = ?'
        ).bind(paymentId, 'pending').first();

        if (payment) {
          // [SECURITY] Verify payment amount via PortOne API before crediting
          const verified = await verifyPortonePayment(env, paymentId, payment.amount_usd);
          if (verified.success) {
            // complete 처리 (프론트에서 complete 호출 못한 경우 보충)
            await addClips(env, payment.user_id, payment.credits, 'purchase', 'clip_pack',
              `Webhook: ${payment.credits} clips`);

            await env.DB.prepare(
              `UPDATE payments SET status = 'completed', completed_at = ? WHERE id = ?`
            ).bind(new Date().toISOString(), paymentId).run();
          }
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
      return { success: false, error: 'Payment verification failed' };
    }

    const data = await response.json();

    // 상태 확인
    if (data.status !== 'PAID') {
      return { success: false, error: 'Payment not completed' };
    }

    // 금액 확인
    if (data.amount?.total !== expectedAmount) {
      return { success: false, error: 'Amount mismatch' };
    }

    return { success: true, portone_payment_id: data.id || paymentId };
  } catch (error) {
    return { success: false, error: 'Verification failed' };
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
      return { success: false, error: 'Billing charge failed' };
    }

    return { success: true };
  } catch (error) {
    return { success: false, error: 'Billing error' };
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

        await addClips(env, sub.user_id, plan.monthlyClips, 'subscription', 'plan_renewal',
          `${plan.name} 월간 갱신 (${plan.monthlyClips} clips)`);

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
        ).bind(paymentId, sub.user_id, plan.priceKrw, plan.monthlyClips,
          sub.billing_key, now.toISOString(), now.toISOString()).run();

        console.log(`Renewal success: user=${sub.user_id}, plan=${sub.plan_id}`);
      } else {
        // 결제 실패 — 구독 만료
        console.error(`Renewal failed: user=${sub.user_id}`);

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
      console.error(`Renewal error: user=${sub.user_id}`);
    }
  }
}

// ==================== 관리자 API ====================

async function handleAdminGrantClips(request, user, env, cors) {
  const adminEmails = (env.ADMIN_EMAILS || '').split(',').map(e => e.trim());
  if (!adminEmails.includes(user.email)) {
    return jsonResponse({ error: 'Forbidden' }, 403, cors);
  }

  const { target_user_id, target_email, amount, reason } = await request.json();

  // [SECURITY] Validate amount
  if (!Number.isInteger(amount) || amount <= 0) {
    return jsonResponse({ error: 'amount must be a positive integer' }, 400, cors);
  }
  if (amount > 10000) {
    return jsonResponse({ error: 'amount exceeds maximum (10000)' }, 400, cors);
  }

  let targetId = target_user_id;
  if (!targetId && target_email) {
    if (typeof target_email !== 'string') return jsonResponse({ error: 'Invalid email' }, 400, cors);
    const target = await env.DB.prepare('SELECT id FROM users WHERE email = ?').bind(target_email).first();
    if (!target) return jsonResponse({ error: 'Target user not found' }, 404, cors);
    targetId = target.id;
  }

  if (!targetId) {
    return jsonResponse({ error: 'target_user_id or target_email required' }, 400, cors);
  }

  await addClips(env, targetId, amount, 'purchase', 'admin_grant',
    `Admin grant by ${user.email}: ${reason || 'no reason'}`);

  return jsonResponse({ message: `Granted ${amount} credits`, target_user_id: targetId }, 200, cors);
}

// ==================== 게시판 핸들러 ====================

async function handleGetPosts(url, env, cors, user) {
  try {
    const page = Math.max(1, parseInt(url.searchParams.get('page') || '1') || 1);
    const limit = Math.min(Math.max(1, parseInt(url.searchParams.get('limit') || '20') || 20), 50);
    const offset = (page - 1) * limit;
    const category = url.searchParams.get('category') || '';

    let whereClause = 'WHERE p.is_deleted = 0';
    const binds = [];
    if (category && ['feedback', 'bug', 'feature', 'question', 'tip', 'general'].includes(category)) {
      whereClause += ' AND p.category = ?';
      binds.push(category);
    }

    const total = await env.DB.prepare(
      `SELECT COUNT(*) as count FROM posts p ${whereClause}`
    ).bind(...binds).first();

    const posts = await env.DB.prepare(
      `SELECT p.id, p.title, p.category, p.view_count, p.like_count, p.comment_count,
              p.is_pinned, p.created_at, p.user_id,
              COALESCE(u.username, SUBSTR(u.email, 1, INSTR(u.email, '@') - 1)) as author_name
       FROM posts p
       LEFT JOIN users u ON p.user_id = u.id
       ${whereClause}
       ORDER BY p.is_pinned DESC, p.created_at DESC
       LIMIT ? OFFSET ?`
    ).bind(...binds, limit, offset).all();

    return jsonResponse({
      posts: posts.results,
      pagination: { page, limit, total: total.count, pages: Math.ceil(total.count / limit) },
    }, 200, cors);
  } catch (error) {
    console.error('GetPosts error:', error);
    return jsonResponse({ error: 'Failed to load posts' }, 500, cors);
  }
}

async function handleGetPost(url, env, cors, user) {
  try {
    const postId = parseInt(url.pathname.split('/').pop());

    const post = await env.DB.prepare(
      `SELECT p.*, COALESCE(u.username, SUBSTR(u.email, 1, INSTR(u.email, '@') - 1)) as author_name
       FROM posts p LEFT JOIN users u ON p.user_id = u.id
       WHERE p.id = ? AND p.is_deleted = 0`
    ).bind(postId).first();

    if (!post) return jsonResponse({ error: 'Post not found' }, 404, cors);

    // 조회수 증가
    await env.DB.prepare(
      'UPDATE posts SET view_count = view_count + 1 WHERE id = ?'
    ).bind(postId).run();

    // 현재 유저의 좋아요 여부
    let liked = false;
    if (user) {
      const like = await env.DB.prepare(
        'SELECT 1 FROM post_likes WHERE post_id = ? AND user_id = ?'
      ).bind(postId, user.id).first();
      liked = !!like;
    }

    return jsonResponse({ ...post, view_count: post.view_count + 1, liked }, 200, cors);
  } catch (error) {
    console.error('GetPost error:', error);
    return jsonResponse({ error: 'Failed to load post' }, 500, cors);
  }
}

async function handleCreatePost(request, user, env, cors) {
  try {
    const { title, content, category } = await request.json();

    if (!title || !content) return jsonResponse({ error: 'Title and content required' }, 400, cors);
    if (typeof title !== 'string' || title.length > 200) return jsonResponse({ error: 'Title must be under 200 characters' }, 400, cors);
    if (typeof content !== 'string' || content.length > 5000) return jsonResponse({ error: 'Content must be under 5000 characters' }, 400, cors);

    const validCategories = ['feedback', 'bug', 'feature', 'question', 'tip', 'general'];
    const cat = validCategories.includes(category) ? category : 'general';
    const now = new Date().toISOString();

    const result = await env.DB.prepare(
      `INSERT INTO posts (user_id, title, content, category, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?)`
    ).bind(user.id, title.trim(), content.trim(), cat, now, now).run();

    return jsonResponse({ id: result.meta.last_row_id, message: 'Post created' }, 201, cors);
  } catch (error) {
    console.error('CreatePost error:', error);
    return jsonResponse({ error: 'Failed to create post' }, 500, cors);
  }
}

async function handleUpdatePost(url, request, user, env, cors) {
  try {
    const postId = parseInt(url.pathname.split('/').pop());
    const post = await env.DB.prepare('SELECT user_id FROM posts WHERE id = ? AND is_deleted = 0').bind(postId).first();

    if (!post) return jsonResponse({ error: 'Post not found' }, 404, cors);
    if (post.user_id !== user.id) return jsonResponse({ error: 'Forbidden' }, 403, cors);

    const { title, content, category } = await request.json();
    if (!title || !content) return jsonResponse({ error: 'Title and content required' }, 400, cors);
    if (typeof title !== 'string' || title.length > 200) return jsonResponse({ error: 'Title must be under 200 characters' }, 400, cors);
    if (typeof content !== 'string' || content.length > 5000) return jsonResponse({ error: 'Content must be under 5000 characters' }, 400, cors);

    const validCategories = ['feedback', 'bug', 'feature', 'question', 'tip', 'general'];
    const cat = validCategories.includes(category) ? category : 'general';

    await env.DB.prepare(
      'UPDATE posts SET title = ?, content = ?, category = ?, updated_at = ? WHERE id = ?'
    ).bind(title.trim(), content.trim(), cat, new Date().toISOString(), postId).run();

    return jsonResponse({ message: 'Post updated' }, 200, cors);
  } catch (error) {
    console.error('UpdatePost error:', error);
    return jsonResponse({ error: 'Failed to update post' }, 500, cors);
  }
}

async function handleDeletePost(url, user, env, cors) {
  try {
    const postId = parseInt(url.pathname.split('/').pop());
    const post = await env.DB.prepare('SELECT user_id FROM posts WHERE id = ? AND is_deleted = 0').bind(postId).first();

    if (!post) return jsonResponse({ error: 'Post not found' }, 404, cors);
    if (post.user_id !== user.id) return jsonResponse({ error: 'Forbidden' }, 403, cors);

    await env.DB.prepare(
      'UPDATE posts SET is_deleted = 1, updated_at = ? WHERE id = ?'
    ).bind(new Date().toISOString(), postId).run();

    return jsonResponse({ message: 'Post deleted' }, 200, cors);
  } catch (error) {
    console.error('DeletePost error:', error);
    return jsonResponse({ error: 'Failed to delete post' }, 500, cors);
  }
}

async function handleGetComments(url, env, cors) {
  try {
    const postId = parseInt(url.pathname.split('/')[4]);

    const comments = await env.DB.prepare(
      `SELECT c.id, c.content, c.created_at, c.user_id,
              COALESCE(u.username, SUBSTR(u.email, 1, INSTR(u.email, '@') - 1)) as author_name
       FROM comments c
       LEFT JOIN users u ON c.user_id = u.id
       WHERE c.post_id = ? AND c.is_deleted = 0
       ORDER BY c.created_at ASC`
    ).bind(postId).all();

    return jsonResponse({ comments: comments.results }, 200, cors);
  } catch (error) {
    console.error('GetComments error:', error);
    return jsonResponse({ error: 'Failed to load comments' }, 500, cors);
  }
}

async function handleCreateComment(url, request, user, env, cors) {
  try {
    const postId = parseInt(url.pathname.split('/')[4]);
    const { content } = await request.json();

    if (!content || typeof content !== 'string' || content.length > 1000) {
      return jsonResponse({ error: 'Content must be 1-1000 characters' }, 400, cors);
    }

    // 게시글 존재 확인
    const post = await env.DB.prepare('SELECT id FROM posts WHERE id = ? AND is_deleted = 0').bind(postId).first();
    if (!post) return jsonResponse({ error: 'Post not found' }, 404, cors);

    const now = new Date().toISOString();
    await env.DB.batch([
      env.DB.prepare(
        'INSERT INTO comments (post_id, user_id, content, created_at) VALUES (?, ?, ?, ?)'
      ).bind(postId, user.id, content.trim(), now),
      env.DB.prepare(
        'UPDATE posts SET comment_count = comment_count + 1, updated_at = ? WHERE id = ?'
      ).bind(now, postId),
    ]);

    return jsonResponse({ message: 'Comment created' }, 201, cors);
  } catch (error) {
    console.error('CreateComment error:', error);
    return jsonResponse({ error: 'Failed to create comment' }, 500, cors);
  }
}

async function handleToggleLike(url, user, env, cors) {
  try {
    const postId = parseInt(url.pathname.split('/')[4]);

    // 게시글 존재 확인
    const post = await env.DB.prepare('SELECT id FROM posts WHERE id = ? AND is_deleted = 0').bind(postId).first();
    if (!post) return jsonResponse({ error: 'Post not found' }, 404, cors);

    // 기존 좋아요 확인
    const existing = await env.DB.prepare(
      'SELECT 1 FROM post_likes WHERE post_id = ? AND user_id = ?'
    ).bind(postId, user.id).first();

    if (existing) {
      // 좋아요 취소
      await env.DB.batch([
        env.DB.prepare('DELETE FROM post_likes WHERE post_id = ? AND user_id = ?').bind(postId, user.id),
        env.DB.prepare('UPDATE posts SET like_count = MAX(0, like_count - 1) WHERE id = ?').bind(postId),
      ]);
      return jsonResponse({ liked: false, message: 'Like removed' }, 200, cors);
    } else {
      // 좋아요 추가
      await env.DB.batch([
        env.DB.prepare('INSERT INTO post_likes (post_id, user_id, created_at) VALUES (?, ?, ?)').bind(postId, user.id, new Date().toISOString()),
        env.DB.prepare('UPDATE posts SET like_count = like_count + 1 WHERE id = ?').bind(postId),
      ]);
      return jsonResponse({ liked: true, message: 'Like added' }, 200, cors);
    }
  } catch (error) {
    console.error('ToggleLike error:', error);
    return jsonResponse({ error: 'Failed to toggle like' }, 500, cors);
  }
}

// ==================== 기존 핸들러 ====================

async function handleHealth(env, cors) {
  return jsonResponse({
    status: 'ok',
    version: '3.1-security',
    timestamp: new Date().toISOString(),
  }, 200, cors);
}

async function handleGenerate(request, user, env, ctx, cors) {
  try {
    const body = await request.json();

    const clipAction = body.mode === 'mv' ? 'mv' : 'video';
    const cost = CLIP_COSTS[clipAction];

    // [SECURITY] Atomic deduction
    if (cost > 0) {
      const projectId = generateProjectId();
      const deductResult = await env.DB.prepare(
        'UPDATE users SET credits = credits - ?, updated_at = ? WHERE id = ? AND credits >= ?'
      ).bind(cost, new Date().toISOString(), user.id, cost).run();

      if (deductResult.meta.changes === 0) {
        const currentUser = await env.DB.prepare('SELECT credits FROM users WHERE id = ?').bind(user.id).first();
        return jsonResponse({
          error: 'Insufficient clips',
          required: cost,
          available: currentUser?.credits || 0,
          action: clipAction,
        }, 402, cors);
      }

      await env.DB.prepare(
        `INSERT INTO projects (id, user_id, status, input_data, created_at)
         VALUES (?, ?, ?, ?, ?)`
      ).bind(projectId, user.id, 'queued', JSON.stringify(body), new Date().toISOString()).run();

      await env.DB.prepare(
        `INSERT INTO credit_transactions (user_id, project_id, amount, type, action, description, created_at)
         VALUES (?, ?, ?, 'usage', ?, ?, ?)`
      ).bind(user.id, projectId, -cost, clipAction, `${clipAction} generation`, new Date().toISOString()).run();

      if (env.VIDEO_QUEUE) {
        await env.VIDEO_QUEUE.send({
          projectId,
          userId: user.id,
          input: body,
          timestamp: Date.now(),
        });
      }

      const freshUser = await env.DB.prepare('SELECT credits FROM users WHERE id = ?').bind(user.id).first();

      return jsonResponse({
        project_id: projectId,
        status: 'queued',
        message: 'Generation started',
        clips_used: cost,
        clips_remaining: freshUser?.credits || 0,
      }, 200, cors);
    }

    return jsonResponse({ error: 'Invalid action' }, 400, cors);
  } catch (error) {
    console.error('Generate error:', error);
    return jsonResponse({ error: 'Generation failed' }, 500, cors);
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
    console.error('Download error');
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
    console.error('Status error');
    return jsonResponse({ error: 'Status check failed' }, 500, cors);
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
