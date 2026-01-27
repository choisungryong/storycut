/**
 * STORYCUT Queue Consumer
 *
 * 역할:
 * - Queue에서 영상 생성 작업을 가져와서 처리
 * - Python 백엔드 API 호출 (실제 영상 생성)
 * - 완료된 영상을 R2에 업로드
 * - D1 상태 업데이트
 */

export default {
  async queue(batch, env) {
    for (const message of batch.messages) {
      try {
        await processVideoGeneration(message.body, env);
        message.ack();
      } catch (error) {
        console.error('Queue processing error:', error);
        message.retry();
      }
    }
  },
};

/**
 * 영상 생성 처리
 */
async function processVideoGeneration(job, env) {
  const { projectId, userId, input } = job;

  console.log(`Processing project ${projectId} for user ${userId}`);

  try {
    // 1. D1 상태 업데이트: processing
    await updateProjectStatus(projectId, 'processing', env);

    // 2. Python 백엔드 API 호출 (실제 영상 생성)
    const backendUrl = env.BACKEND_URL || 'http://your-backend-server.com';

    const response = await fetch(`${backendUrl}/api/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Project-ID': projectId,
        'Authorization': `Bearer ${env.BACKEND_API_SECRET}`,
      },
      body: JSON.stringify({
        project_id: projectId,
        ...input,
      }),
    });

    if (!response.ok) {
      throw new Error(`Backend API failed: ${response.status}`);
    }

    const result = await response.json();

    // 3. 생성된 영상을 R2에 업로드
    // (실제로는 백엔드에서 직접 R2에 업로드하는 게 효율적)
    // 여기서는 메타데이터만 업데이트

    // 4. D1 상태 업데이트: completed
    await updateProjectStatus(projectId, 'completed', env, {
      video_url: result.video_url,
      title: result.title,
      optimization: result.optimization,
    });

    // 5. 사용자에게 알림 (선택사항)
    await notifyUser(userId, projectId, env);

    console.log(`Project ${projectId} completed successfully`);
  } catch (error) {
    console.error(`Project ${projectId} failed:`, error);

    // 실패 상태로 업데이트
    await updateProjectStatus(projectId, 'failed', env, {
      error_message: error.message,
    });
  }
}

/**
 * D1 프로젝트 상태 업데이트
 */
async function updateProjectStatus(projectId, status, env, metadata = {}) {
  const updates = [`status = ?`];
  const params = [status];

  if (status === 'completed') {
    updates.push(`completed_at = ?`);
    params.push(new Date().toISOString());
  }

  if (metadata.video_url) {
    updates.push(`video_url = ?`);
    params.push(metadata.video_url);
  }

  if (metadata.title) {
    updates.push(`title = ?`);
    params.push(metadata.title);
  }

  if (metadata.optimization) {
    updates.push(`optimization_data = ?`);
    params.push(JSON.stringify(metadata.optimization));
  }

  if (metadata.error_message) {
    updates.push(`error_message = ?`);
    params.push(metadata.error_message);
  }

  params.push(projectId);

  await env.DB.prepare(
    `UPDATE projects SET ${updates.join(', ')} WHERE id = ?`
  )
    .bind(...params)
    .run();
}

/**
 * 사용자 알림
 */
async function notifyUser(userId, projectId, env) {
  // 이메일 알림, 푸시 알림 등
  // 여기서는 단순히 D1에 알림 레코드 추가
  await env.DB.prepare(
    `INSERT INTO notifications (user_id, project_id, type, message, created_at)
     VALUES (?, ?, ?, ?, ?)`
  )
    .bind(
      userId,
      projectId,
      'video_completed',
      '영상 생성이 완료되었습니다!',
      new Date().toISOString()
    )
    .run();
}
