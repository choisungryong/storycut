
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

// JWT 비밀키 (반드시 wrangler secret put JWT_SECRET으로 설정!)
// 주의: 하드코딩 금지! env.JWT_SECRET 사용
function getJwtSecret(env) {
  if (!env.JWT_SECRET) {
    console.error('[SECURITY] JWT_SECRET not configured! Using fallback (INSECURE!)');
    // 개발/테스트용 폴백 (프로덕션에서는 절대 사용 금지)
    return new TextEncoder().encode('dev-only-insecure-key-replace-me');
  }
  return new TextEncoder().encode(env.JWT_SECRET);
}

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

        // DB 저장 (기본 크레딧 100 제공)
        const result = await env.DB.prepare(
          `INSERT INTO users (email, password_hash, username, credits) VALUES (?, ?, ?, 100)`
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
          .sign(getJwtSecret(env));

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
        const { payload } = await jwtVerify(token, getJwtSecret(env));

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
          const { payload } = await jwtVerify(token, getJwtSecret(env));
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

    if (url.pathname === '/api/generate/video' && request.method === 'POST') {
      // Step 2: Video Generation (Existing Logic)
      // Auth & Credit check included in handleGenerate
      const authHeader = request.headers.get('Authorization');
      let userId = null;
      let userCredits = 0;

      if (authHeader && authHeader.startsWith('Bearer ')) {
        try {
          const token = authHeader.split(' ')[1];
          const { payload } = await jwtVerify(token, getJwtSecret(env));
          userId = payload.id;
          const user = await env.DB.prepare(`SELECT credits FROM users WHERE id = ?`).bind(userId).first();
          userCredits = user ? user.credits : 0;
        } catch (e) {
          console.warn('Token verification failed', e);
        }
      }
      return handleGenerate(request, env, ctx, corsHeaders, userId, userCredits);
    }

    // [New] Step 1: Story Generation (Worker Logic)
    if (url.pathname === '/api/generate/story' && request.method === 'POST') {
      return handleGenerateStory(request, env, corsHeaders);
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

    if (url.pathname.startsWith('/api/sample-voice/')) {
      return handleProxyToBackend(request, url, env, corsHeaders);
    }

    // [New] Manifest Proxy
    if (url.pathname.startsWith('/api/manifest/')) {
      return handleProxyToBackend(request, url, env, corsHeaders);
    }

    // [New] Generic Asset Download (Image/Audio/Video) from R2
    // Path: /api/asset/{projectId}/{type}/{filename}
    // Type: 'images', 'audio', 'videos'
    if (url.pathname.startsWith('/api/asset/')) {
      return handleAssetDownload(url, env, corsHeaders);
    }

    // 정적 파일 (Pages에서 서빙)
    return new Response('Not Found', { status: 404, headers: corsHeaders });
  },

  /**
   * Queue Consumer (Keep existing)
   */
  async queue(batch, env) {
    // ... existing queue logic ...
    console.log(`[Queue] Received batch of ${batch.messages.length} messages`);
    for (const message of batch.messages) {
      try {
        const { projectId, userId, input } = message.body;
        console.log(`[Queue] Processing project ${projectId} for user ${userId}`);
        message.ack();
      } catch (error) {
        console.error(`[Queue] Error processing message ${message.id}:`, error);
        message.retry();
      }
    }
  },
};

// ... existing helper functions (handleHealth, handleGenerate, handleVideoDownload, handleStatus, etc.) ...
// We need to keep them. But since replace_file_content replaces a block, I should be careful.
// I will just append the new function at the end and modify the routing block.

/**
 * [New] Story Generation Handler (JS Port of StoryAgent)
 */
async function handleGenerateStory(request, env, corsHeaders) {
  try {
    const body = await request.json();
    const { genre, mood, style, duration, topic } = body;

    // Gemini API Key Check
    const apiKey = env.GOOGLE_API_KEY;
    if (!apiKey) {
      throw new Error("Server Error: Google API Key not configured");
    }

    const total_duration_sec = duration || 90;
    const min_scenes = Math.floor(total_duration_sec / 8);
    const max_scenes = Math.floor(total_duration_sec / 4);

    // =================================================================================
    // STEP 1: Story Architecture
    // =================================================================================
    const step1_prompt = `
ROLE: Professional Storyboard Artist & Director.
TASK: Plan the structure for a ${total_duration_sec}-second YouTube Short.
GENRE: ${genre}
MOOD: ${mood}
STYLE: ${style}
SCENE COUNT: Approx ${min_scenes}-${max_scenes} scenes.
${topic ? 'USER IDEA: ' + topic : ''}

[STRUCTURE REQUIREMENT: KI-SEUNG-JEON-GYEOL]
1. **Introduction (Ki)**: Hook the audience, introduce characters/setting.
2. **Development (Seung)**: Escalate tension, build the conflict.
3. **Twist (Jeon)**: The climax, a shocking revelation or turning point.
4. **Resolution (Gyeol)**: **MANDATORY**. The aftermath. How did it end? What is the final state? 
   - DO NOT just stop at the twist. Show the consequence.

OUTPUT FORMAT (JSON):
{
  "project_title": "Creative Title",
  "logline": "One sentence summary including the ending",
  "global_style": {
    "art_style": "${style}",
    "color_palette": "e.g., Cyberpunk Neons",
    "visual_seed": ${Math.floor(Math.random() * 100000)}
  },
  "characters": {
    "Name": {
      "name": "Name",
      "appearance": "Detailed description",
      "role": "Protagonist/Antagonist"
    }
  },
  "outline": [
    { "scene_id": 1, "summary": "Brief summary of what happens", "estimated_duration": 5 }
  ]
}
`;
    const step1_response = await callGemini(apiKey, step1_prompt, SYSTEM_PROMPT);
    let structure_data = {};
    try {
      structure_data = parseGeminiResponse(step1_response);
      console.log("Step 1 Structure:", structure_data.project_title);
    } catch (e) {
      console.error("Step 1 Parse Error:", e);
      // Fallback or empty
    }

    // =================================================================================
    // STEP 2: Scene-level Details
    // =================================================================================
    const structure_context = JSON.stringify(structure_data, null, 2);

    const step2_prompt = `
ROLE: Screenwriter & Visual Director.
TASK: Generate detailed scene specs based on the approved structure.

APPROVED STRUCTURE:
${structure_context}

REQUIREMENTS:
- Follow the outline exactly.
- "narrative": The action description (Korean).
- "tts_script": The spoken line (Korean). **MUST BE NATURAL SPOKEN KOREAN (구어체).** Avoid "written style" (문어체).
    - BAD: "그는 문을 열고 들어왔다." (Narration style)
    - GOOD: "결국... 돌아왔군." (Character dialogue or Monologue)
- "image_prompt": Visual description for AI Image Generator (English). ${style} style.
- "camera_work": Specific camera movement (e.g., "Close-up", "Pan Right", "Drone Shot").

[STRICT] CHARACTER CONSISTENCY RULE:
- Refer to characters ONLY by their IDs (e.g., STORYCUT_HERO_A) in the "image_prompt".
- DO NOT describe their physical appearance (age, hair, clothes) in "image_prompt". This is already handled by the system.
- Focus ONLY on the scene's action, lighting, and composition.

[STRICT] ENDING RULE:
- The final scenes MUST clearly show the conclusion.
- The last line of "tts_script" should leave a lingering impression but NOT be open-ended.

OUTPUT FORMAT (JSON):
{
  "title": "${structure_data.project_title || 'Untitled'}",
  "genre": "${genre}",
  "total_duration_sec": ${total_duration_sec},
  "character_sheet": ${JSON.stringify(structure_data.characters || {})},
  "global_style": ${JSON.stringify(structure_data.global_style || {})},
  "scenes": [
    {
      "scene_id": 1,
      "narrative": "STORYCUT_HERO_A가 카페 문을 열고 들어온다.",
      "image_prompt": "STORYCUT_HERO_A opening a cafe door, webtoon style, high contrast, cinematic lighting.",
      "tts_script": "드디어 이곳인가...",
      "duration_sec": 5,
      "camera_work": "Close-up",
      "mood": "tense",
      "characters_in_scene": ["STORYCUT_HERO_A"]
    }
  ],
  "youtube_opt": {
    "title_candidates": ["Title 1", "Title 2"],
    "thumbnail_text": "Hook Text",
    "hashtags": ["#Tag1", "#Tag2"]
  }
}
`;
    const step2_response = await callGemini(apiKey, step2_prompt, SYSTEM_PROMPT);
    const story_data = parseGeminiResponse(step2_response);

    // Validation / Normalization
    if (story_data.scenes) {
      story_data.scenes.forEach(scene => {
        if (scene.tts_script) {
          scene.narration = scene.tts_script;
          scene.sentence = scene.tts_script;
        }
        if (scene.image_prompt) {
          scene.visual_description = scene.image_prompt;
          scene.prompt = scene.image_prompt;
        }
      });
    }

    return new Response(JSON.stringify({
      story_data: story_data,
      request_params: body,
      project_id: Math.random().toString(36).substring(2, 10)
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error("Story Generation Error:", error);
    return new Response(JSON.stringify({ detail: error.message }), {
      status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }
}

async function callGemini(apiKey, prompt, systemPrompt) {
  // [Fix] Use 'gemini-3-pro-preview' as requested (User's specific model)
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:generateContent?key=${apiKey}`;

  // Gemini 1.5 Pro supports systemInstruction
  const payload = {
    contents: [{
      role: "user",
      parts: [{ text: prompt }]
    }],
    systemInstruction: {
      parts: [{ text: systemPrompt }]
    },
    generationConfig: {
      temperature: 0.7,
      responseMimeType: "application/json"
    }
  };

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Gemini API Error ${response.status}: ${errText}`);
  }

  const data = await response.json();
  if (data.candidates && data.candidates[0].content && data.candidates[0].content.parts[0].text) {
    return data.candidates[0].content.parts[0].text;
  } else {
    throw new Error("Unexpected Gemini response structure");
  }
}

function parseGeminiResponse(text) {
  try {
    // Remove markdown
    let cleaned = text.trim();
    if (cleaned.startsWith("```json")) cleaned = cleaned.substring(7);
    if (cleaned.startsWith("```")) cleaned = cleaned.substring(3);
    if (cleaned.endsWith("```")) cleaned = cleaned.substring(0, cleaned.length - 3);
    return JSON.parse(cleaned);
  } catch (e) {
    throw new Error("Failed to parse JSON from Gemini: " + e.message);
  }
}

const SYSTEM_PROMPT = `
# 🎬 STORYCUT Master Storytelling Prompt v2.1
You are the **BEST SHORT-FORM STORYTELLER** in the world. Your job is to create **VIRAL-WORTHY stories**.

[CRITICAL RULE: COMPLETE NARRATIVE ARC]
Your story MUST have a clear **Introduction, Development, Twist, and RESOLUTION**.
- **NO CLIFFHANGERS.** The story must end definitively.
- **NO VAGUE ENDINGS.** The audience must know exactly what happened to the characters.
- **SATISFYING CONCLUSION.** Even if it's a sad ending, it must feel complete.

Generate a **complete, immersive narrative** (2-3 minutes) with a GRIPPING HOOK, RISING TENSION, SHOCKING TWIST, and a DEFINITIVE ENDING.
Output MUST be valid JSON.
`;

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
    const targetUrl = `${BACKEND_URL}${url.pathname}`;

    console.log(`Forwarding request to backend: ${targetUrl}`);

    // 백엔드에 요청 전달 (Fire-and-forget 방식)
    // 사용자는 기다리지 않고 바로 응답을 받지만, 백엔드는 작업을 수행함
    ctx.waitUntil(
      fetch(targetUrl, {
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
  // ... existing code ...
  // Keep existing logic or refactor to use generic handleAssetDownload
  return handleAssetDownload(url, env, corsHeaders, 'video');
}

/**
 * Generic Asset Download
 * /api/asset/{projectId}/{type}/{filename}
 */
async function handleAssetDownload(url, env, corsHeaders, forceType = null) {
  const parts = url.pathname.split('/');
  // Expected: ["", "api", "asset", "projectId", "type", "filename"]
  // Or for video: ["", "api", "video", "projectId"] (Handled specially)

  let projectId, type, filename;

  if (forceType === 'video') {
    projectId = parts.pop();
    type = 'videos';
    filename = 'final_video.mp4';
  } else {
    filename = parts.pop();
    type = parts.pop();
    projectId = parts.pop();
  }

  // Safety check path traversal
  if (!projectId || !type || !filename || filename.includes('..')) {
    return new Response('Invalid path', { status: 400, headers: corsHeaders });
  }

  // Map URL type to R2 folder
  // URL type: 'image' -> R2: 'images'
  // URL type: 'audio' -> R2: 'audio'
  // URL type: 'video' -> R2: 'videos'
  let folder = type;
  if (type === 'image') folder = 'images';

  const key = `${folder}/${projectId}/${filename}`;

  try {
    const object = await env.R2_BUCKET.get(key);

    if (!object) {
      return new Response(`Asset not found: ${key}`, { status: 404, headers: corsHeaders });
    }

    const headers = new Headers(object.writeHttpMetadata(corsHeaders));
    headers.set('etag', object.httpEtag);

    // Content-Type mapping
    if (filename.endsWith('.mp4')) headers.set('Content-Type', 'video/mp4');
    else if (filename.endsWith('.png')) headers.set('Content-Type', 'image/png');
    else if (filename.endsWith('.jpg')) headers.set('Content-Type', 'image/jpeg');
    else if (filename.endsWith('.mp3')) headers.set('Content-Type', 'audio/mpeg');
    else if (filename.endsWith('.wav')) headers.set('Content-Type', 'audio/wav');

    return new Response(object.body, { headers });

  } catch (error) {
    console.error('Asset Download Error:', error);
    return new Response('Error downloading asset', { status: 500, headers: corsHeaders });
  }
}

/**
 * 프로젝트 상태 조회
 */
async function handleStatus(url, env, corsHeaders) {
  const projectId = url.pathname.split('/').pop();

  try {
    const result = await env.DB.prepare(
      `SELECT id, status, video_url, created_at, completed_at, error_message, logs, progress
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
         SET status = ?, video_url = ?, completed_at = ?, progress = 100, logs = logs || ? 
         WHERE id = ?`
      )
        .bind('completed', data.output_url || data.video_url, new Date().toISOString(), '\n[완료] 영상 생성이 완료되었습니다.', projectId)
        .run();
    } else if (data.status === 'failed') {
      await env.DB.prepare(
        `UPDATE projects 
         SET status = ?, error_message = ?, logs = logs || ? 
         WHERE id = ?`
      )
        .bind('failed', data.error || 'Unknown error', `\n[오류] ${data.error || '알 수 없는 오류 발생'}`, projectId)
        .run();
    } else {
      // processing, etc.
      // 로그 및 진행률 업데이트
      const logUpdate = data.message ? `\n[${new Date().toLocaleTimeString()}] ${data.message}` : '';
      const progressUpdate = data.progress || null;

      if (progressUpdate !== null) {
        await env.DB.prepare(
          `UPDATE projects SET status = ?, progress = ?, logs = logs || ? WHERE id = ?`
        )
          .bind(data.status, progressUpdate, logUpdate, projectId)
          .run();
      } else {
        await env.DB.prepare(
          `UPDATE projects SET status = ?, logs = logs || ? WHERE id = ?`
        )
          .bind(data.status, logUpdate, projectId)
          .run();
      }
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

/**
 * Generic Backend Proxy
 * 단순히 요청을 Python 백엔드로 전달하고 결과를 반환
 */
async function handleProxyToBackend(request, url, env, corsHeaders) {
  const BACKEND_URL = env.BACKEND_URL || "https://web-production-bb6bf.up.railway.app";
  const targetUrl = `${BACKEND_URL}${url.pathname}${url.search}`;

  console.log(`Proxying request to: ${targetUrl}`);

  try {
    const response = await fetch(targetUrl, {
      method: request.method,
      headers: request.headers,
      body: request.method !== 'GET' ? await request.arrayBuffer() : undefined
    });

    // 응답 헤더 복사 (CORS 포함)
    const newHeaders = new Headers(response.headers);
    Object.keys(corsHeaders).forEach(key => {
      newHeaders.set(key, corsHeaders[key]);
    });

    return new Response(response.body, {
      status: response.status,
      headers: newHeaders
    });
  } catch (error) {
    console.error('Proxy error:', error);
    return new Response(JSON.stringify({ error: 'Backend proxy failed' }), {
      status: 502,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }
}

/**
 * [New] Story Generation Handler (JS Port of StoryAgent)
 */
