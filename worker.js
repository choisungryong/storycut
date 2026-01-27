
import { SignJWT, jwtVerify } from 'jose';
import bcrypt from 'bcryptjs';

/**
 * STORYCUT Cloudflare Worker
 *
 * 역할:
 * 1. API 엔드포인트 제공 (/api/*)
 * 2. 인증 및 크레딧 검증
 * 3. 비동기 작업 큐잉 (Queue)
 * 4. R2에서 영상 서빙
 * 5. D1에 메타데이터 저장
 */

// JWT 비밀키 (실제 운영 시에는 wrangler secret으로 설정해야 함)
const JWT_SECRET = new TextEncoder().encode('my-secret-salt-key-change-this');

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // CORS 헤더
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    };

    // OPTIONS 요청 처리
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    // [API] 회원가입
    if (url.pathname === '/api/auth/register' && request.method === 'POST') {
      try {
        const { email, password, username } = await request.json();

        if (!email || !password) {
          return new Response(JSON.stringify({ error: '아이디(이메일)와 비밀번호를 입력해주세요.' }), {
            status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' }
          });
        }

        // 비밀번호 해싱
        const passwordHash = await bcrypt.hash(password, 10);

        // DB 저장
        const result = await env.DB.prepare(
          `INSERT INTO users (email, password_hash, username) VALUES (?, ?, ?)`
        )
          .bind(email, passwordHash, username || email.split('@')[0])
          .run();

        if (!result.success) {
          throw new Error('회원가입 실패 (이미 존재하는 이메일일 수 있습니다)');
        }

        return new Response(JSON.stringify({ message: '회원가입 성공!' }), {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });

      } catch (err) {
        return new Response(JSON.stringify({ error: err.message }), {
          status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });
      }
    }

    // [API] 로그인
    if (url.pathname === '/api/auth/login' && request.method === 'POST') {
      try {
        const { email, password } = await request.json();

        // 사용자 조회
        const user = await env.DB.prepare(
          `SELECT * FROM users WHERE email = ?`
        )
          .bind(email)
          .first();

        if (!user || !(await bcrypt.compare(password, user.password_hash))) {
          return new Response(JSON.stringify({ error: '아이디 또는 비밀번호가 잘못되었습니다.' }), {
            status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' }
          });
        }

        // JWT 토큰 발급
        const token = await new SignJWT({
          id: user.id,
          email: user.email,
          username: user.username
        })
          .setProtectedHeader({ alg: 'HS256' })
          .setIssuedAt()
          .setExpirationTime('24h') // 24시간 유효
          .sign(JWT_SECRET);

        return new Response(JSON.stringify({
          token,
          user: { id: user.id, email: user.email, username: user.username, credits: user.credits }
        }), {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });

      } catch (err) {
        console.error(err);
        return new Response(JSON.stringify({ error: '로그인 처리 중 오류 발생' }), {
          status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });
      }
    }

    // [API] 사용자 정보 & 크레딧 확인
    if (url.pathname === '/api/auth/me' && request.method === 'GET') {
      // 토큰 검증
      const authHeader = request.headers.get('Authorization');
      if (!authHeader || !authHeader.startsWith('Bearer ')) {
        return new Response(JSON.stringify({ error: '로그인이 필요합니다.' }), { status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' } });
      }

      try {
        const token = authHeader.split(' ')[1];
        const { payload } = await jwtVerify(token, JWT_SECRET);

        // 최신 크레딧 정보 조회
        const user = await env.DB.prepare(`SELECT id, email, username, credits FROM users WHERE id = ?`)
          .bind(payload.id)
          .first();

        return new Response(JSON.stringify({ user }), {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: '유효하지 않은 토큰입니다.' }), { status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' } });
      }
    }

    // 라우팅
    if (url.pathname === '/api/health') {
      return handleHealth(env, corsHeaders);
    }

    // [기존 API] 영상 생성 (인증 적용)
    if (url.pathname === '/api/generate' && request.method === 'POST') {
      // 토큰 검증 (선택 사항: 배포 초기엔 없어도 되지만, 이제 인증 붙였으니 체크)
      const authHeader = request.headers.get('Authorization');
      let userId = null; // 비회원도 일단 허용? 아니면 차단? -> 일단 비회원용 임시 처리 or 차단
      let userCredits = 0;

      if (authHeader && authHeader.startsWith('Bearer ')) {
        try {
          const token = authHeader.split(' ')[1];
          const { payload } = await jwtVerify(token, JWT_SECRET);
          userId = payload.id;

          // 크레딧 확인
          const user = await env.DB.prepare(`SELECT credits FROM users WHERE id = ?`).bind(userId).first();
          userCredits = user ? user.credits : 0;
        } catch (e) {
          console.warn('Token verification failed', e);
        }
      }

      // (임시) 비회원이면 못 쓰게 막으려면 여기서 return 401 하면 됨.
      // 일단은 기존 로직 유지하되, userId가 있으면 DB 크레딧 차감하도록 수정

      return handleGenerate(request, env, ctx, corsHeaders, userId, userCredits);
    }

    if (url.pathname.startsWith('/api/video/')) {
      return handleVideoDownload(url, env, corsHeaders);
    }

    if (url.pathname.startsWith('/api/webhook/')) {
      return handleWebhook(request, url, env, corsHeaders);
    }

    if (url.pathname.startsWith('/api/status/')) {
      return handleStatus(url, env, corsHeaders);
    }

    // 정적 파일 (Pages에서 서빙)
    return new Response('Not Found', { status: 404, headers: corsHeaders });
  },

  /**
   * Queue Consumer
   * 비동기 작업 처리
   */
  async queue(batch, env) {
    console.log(`[Queue] Received batch of ${batch.messages.length} messages`);

    for (const message of batch.messages) {
      try {
        const { projectId, userId, input } = message.body;

        console.log(`[Queue] Processing project ${projectId} for user ${userId}`);

        // TODO: 여기서 실제 Python 백엔드 호출 또는 비디오 생성 로직 수행
        // 현재는 로그만 출력하고 성공 처리

        // 메시지 확인 (성공)
        message.ack();
      } catch (error) {
        console.error(`[Queue] Error processing message ${message.id}:`, error);

        // 실패 시 재시도 (Dead Letter Queue로 이동 전)
        message.retry();
      }
    }
  },
};

/**
 * 헬스 체크
 */
async function handleHealth(env, corsHeaders) {
  return new Response(
    JSON.stringify({
      status: 'ok',
      version: '2.0',
      timestamp: new Date().toISOString(),
    }),
    {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    }
  );
}

/**
 * 영상 생성 요청 처리
 *
 * 1. 사용자 인증 확인
 * 2. 크레딧 차감
 * 3. Queue에 작업 추가
 * 4. 프로젝트 ID 반환
 */
async function handleGenerate(request, env, ctx, corsHeaders, userId, userCredits) {
  try {
    // 요청 데이터 파싱
    const body = await request.json();
    const url = new URL(request.url);

    // 인증 확인 (이미 fetch에서 검증됨)
    if (!userId) {
      return new Response(
        JSON.stringify({ error: 'Unauthorized (재로그인 필요)' }),
        { status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // 크레딧 확인
    const creditRequired = calculateCredit(body);
    if (userCredits < creditRequired) {
      return new Response(
        JSON.stringify({
          error: 'Insufficient credits',
          required: creditRequired,
          available: userCredits,
        }),
        { status: 402, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // 프로젝트 ID 생성
    const projectId = generateProjectId();

    // D1에 프로젝트 메타데이터 저장
    await env.DB.prepare(
      `INSERT INTO projects (id, user_id, status, input_data, created_at)
       VALUES (?, ?, ?, ?, ?)`
    )
      .bind(projectId, userId, 'queued', JSON.stringify(body), new Date().toISOString())
      .run();

    // Queue에 작업 추가 (Queue가 있을 때만)
    if (env.VIDEO_QUEUE) {
      /*
      await env.VIDEO_QUEUE.send({
        projectId,
        userId: userId,
        input: body,
        timestamp: Date.now(),
      });
      */
      console.warn('Queue disabled. Direct backend call required here.');
    }

    // [Direct Call] Python 백엔드 호출 (Railway)
    const BACKEND_URL = env.BACKEND_URL || "https://web-production-bb6bf.up.railway.app";

    console.log(`Forwarding request to backend: ${BACKEND_URL}/api/generate`);

    // 백엔드에 요청 전달 (Fire-and-forget 방식)
    // 사용자는 기다리지 않고 바로 응답을 받지만, 백엔드는 작업을 수행함
    ctx.waitUntil(
      fetch(`${BACKEND_URL}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...body,
          project_id: projectId, // 이미 생성된 ID 전달
          webhook_url: `${url.origin}/api/webhook/${projectId}` // 완료 알림용 (나중에 구현)
        }),
      }).then(res => {
        console.log(`Backend response: ${res.status}`);
      }).catch(err => {
        console.error(`Backend call failed:`, err);
      })
    );

    // 크레딧 차감
    await env.DB.prepare(
      `UPDATE users SET credits = credits - ? WHERE id = ?`
    )
      .bind(creditRequired, userId)
      .run();

    return new Response(
      JSON.stringify({
        project_id: projectId,
        status: 'started',
        message: '영상 생성이 시작되었습니다.',
        server_url: BACKEND_URL, // 디버깅용
        credits_used: creditRequired,
        credits_remaining: userCredits - creditRequired,
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  } catch (error) {
    console.error('Generate error:', error);
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
}

/**
 * 영상 다운로드
 * R2 Storage에서 가져오기
 */
async function handleVideoDownload(url, env, corsHeaders) {
  const projectId = url.pathname.split('/').pop();

  try {
    // R2에서 영상 가져오기
    const object = await env.R2_BUCKET.get(`videos/${projectId}/final_video.mp4`);

    if (!object) {
      return new Response('Video not found', {
        status: 404,
        headers: corsHeaders,
      });
    }

    return new Response(object.body, {
      headers: {
        ...corsHeaders,
        'Content-Type': 'video/mp4',
        'Content-Disposition': `attachment; filename="storycut_${projectId}.mp4"`,
        'Cache-Control': 'public, max-age=31536000', // 1년 캐싱
      },
    });
  } catch (error) {
    console.error('Download error:', error);
    return new Response('Error downloading video', {
      status: 500,
      headers: corsHeaders,
    });
  }
}

/**
 * 프로젝트 상태 조회
 */
async function handleStatus(url, env, corsHeaders) {
  const projectId = url.pathname.split('/').pop();

  try {
    const result = await env.DB.prepare(
      `SELECT id, status, created_at, completed_at, error_message
       FROM projects WHERE id = ?`
    )
      .bind(projectId)
      .first();

    if (!result) {
      return new Response(
        JSON.stringify({ error: 'Project not found' }),
        { status: 404, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('Status error:', error);
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
}

/**
 * 사용자 인증
 */
async function authenticateUser(authHeader, env) {
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return null;
  }

  const token = authHeader.substring(7);

  // JWT 검증 (실제 구현 필요)
  // 여기서는 단순화된 버전
  try {
    const result = await env.DB.prepare(
      `SELECT id, email, credits FROM users WHERE api_token = ?`
    )
      .bind(token)
      .first();

    return result;
  } catch (error) {
    console.error('Auth error:', error);
    return null;
  }
}

/**
 * 크레딧 계산
 */
function calculateCredit(input) {
  let credit = 1; // 기본 1 크레딧

  // 길이에 따라 추가
  if (input.duration > 60) {
    credit += Math.ceil((input.duration - 60) / 30);
  }

  // Hook 비디오 사용 시 추가
  if (input.hook_scene1_video) {
    credit += 2;
  }

  return credit;
}

/**
 * 프로젝트 ID 생성
 */
function generateProjectId() {
  return Math.random().toString(36).substring(2, 10);
}

/**
 * Webhook Handler
 * Python 백엔드로부터 상태 업데이트 수신
 */
async function handleWebhook(request, url, env, corsHeaders) {
  const projectId = url.pathname.split('/').pop();

  if (request.method !== 'POST') {
    return new Response('Method Not Allowed', { status: 405, headers: corsHeaders });
  }

  try {
    const data = await request.json();
    // data: { status: 'completed'|'failed'|'processing', output_url: '...', error: '...' }

    console.log(`[Webhook] Update for project ${projectId}:`, data);

    if (data.status === 'completed') {
      await env.DB.prepare(
        `UPDATE projects
         SET status = ?, video_url = ?, completed_at = ?
         WHERE id = ?`
      )
        .bind('completed', data.output_url || data.video_url, new Date().toISOString(), projectId)
        .run();
    } else if (data.status === 'failed') {
      await env.DB.prepare(
        `UPDATE projects 
         SET status = ?, error_message = ? 
         WHERE id = ?`
      )
        .bind('failed', data.error || 'Unknown error', projectId)
        .run();
    } else {
      // processing 등 기타 상태
      await env.DB.prepare(
        `UPDATE projects SET status = ? WHERE id = ?`
      )
        .bind(data.status, projectId)
        .run();
    }

    return new Response(JSON.stringify({ success: true }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  } catch (error) {
    console.error('Webhook error:', error);
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }
}
