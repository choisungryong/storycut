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

    // 라우팅
    if (url.pathname === '/api/health') {
      return handleHealth(env, corsHeaders);
    }

    if (url.pathname === '/api/generate' && request.method === 'POST') {
      return handleGenerate(request, env, ctx, corsHeaders);
    }

    if (url.pathname.startsWith('/api/video/')) {
      return handleVideoDownload(url, env, corsHeaders);
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
async function handleGenerate(request, env, ctx, corsHeaders) {
  try {
    // 요청 데이터 파싱
    const body = await request.json();

    // 인증 토큰 확인
    const authHeader = request.headers.get('Authorization');
    const user = await authenticateUser(authHeader, env);

    if (!user) {
      return new Response(
        JSON.stringify({ error: 'Unauthorized' }),
        { status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // 크레딧 확인
    const creditRequired = calculateCredit(body);
    if (user.credits < creditRequired) {
      return new Response(
        JSON.stringify({
          error: 'Insufficient credits',
          required: creditRequired,
          available: user.credits,
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
      .bind(projectId, user.id, 'queued', JSON.stringify(body), new Date().toISOString())
      .run();

    // Queue에 작업 추가 (Queue가 있을 때만)
    if (env.VIDEO_QUEUE) {
      /*
      await env.VIDEO_QUEUE.send({
        projectId,
        userId: user.id,
        input: body,
        timestamp: Date.now(),
      });
      */
      console.warn('Queue disabled. Direct backend call required here.');
    } else {
      console.warn('VIDEO_QUEUE binding not found. Falling back to direct processing (not implemented yet).');
    }

    // 크레딧 차감
    await env.DB.prepare(
      `UPDATE users SET credits = credits - ? WHERE id = ?`
    )
      .bind(creditRequired, user.id)
      .run();

    return new Response(
      JSON.stringify({
        project_id: projectId,
        status: 'queued',
        message: '영상 생성이 요청되었습니다. (Queue 비활성 상태)',
        credits_used: creditRequired,
        credits_remaining: user.credits - creditRequired,
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

