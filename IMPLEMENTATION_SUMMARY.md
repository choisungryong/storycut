# 씬별 재생성 UI 구현 완료

## 구현 내용

### 1. HTML 수정 (`web/templates/index.html`)

#### 추가된 섹션
- **씬 관리 섹션**: 결과 페이지의 다운로드 버튼과 최적화 패키지 사이에 배치
- **씬 목록 그리드**: 개별 씬 카드를 동적으로 표시
- **재합성 버튼**: 씬 재생성 후 영상을 업데이트하는 버튼

#### CSS 스타일 (추가됨)
- `.scene-management`: 씬 관리 섹션 컨테이너
- `.result-scene-grid`: 씬 카드 그리드 레이아웃
- `.result-scene-card`: 개별 씬 카드
- `.scene-status-badge`: 씬 상태 뱃지 (완료/실패/재생성 중)
- `.btn-regenerate`: 재생성 버튼
- `.recompose-actions`: 재합성 버튼 컨테이너
- `.scene-error-message`: 에러 메시지 표시

상태별 스타일:
- `.completed`: 녹색 뱃지 (완료)
- `.failed`: 빨간색 뱃지/테두리 (실패)
- `.regenerating`: 주황색 뱃지/테두리 (재생성 중)

### 2. JavaScript 수정 (`web/static/app.js`)

#### 새로운 함수 추가

1. **`loadSceneList(projectId)`**
   - 프로젝트의 모든 씬 정보를 서버에서 가져옴
   - API: `GET /api/projects/{project_id}/scenes`
   - 씬 목록을 `renderSceneList()` 함수로 전달

2. **`renderSceneList(scenes, projectId)`**
   - 씬 목록을 UI에 렌더링
   - 각 씬 카드에 다음 정보 표시:
     - 씬 번호 (Scene ID)
     - 상태 뱃지 (완료/실패/재생성 중)
     - 내레이션 텍스트
     - 생성 방법 (generation_method)
     - 에러 메시지 (있는 경우)
     - 재생성 버튼
   - 재생성 버튼에 이벤트 리스너 연결

3. **`regenerateScene(projectId, sceneId)`**
   - 특정 씬을 재생성
   - API: `POST /api/projects/{project_id}/scenes/{scene_id}/regenerate`
   - 요청 본문:
     ```json
     {
       "regenerate_image": true,
       "regenerate_tts": true,
       "regenerate_video": true
     }
     ```
   - UI 상태 업데이트:
     - 버튼 비활성화 및 텍스트 변경
     - 카드에 `regenerating` 클래스 추가
   - 완료 후:
     - 씬 목록 새로고침
     - 재합성 버튼 표시
     - 사용자에게 알림

4. **`recomposeVideo(projectId)`**
   - 수정된 씬들을 반영하여 최종 영상 재합성
   - API: `POST /api/projects/{project_id}/recompose`
   - 완료 후:
     - 비디오 플레이어 새로고침 (캐시 우회)
     - 재합성 버튼 숨김
     - 사용자에게 알림

#### 수정된 함수

1. **`showResults(data)`**
   - `async` 함수로 변경
   - 씬 목록 로드 호출 추가: `await this.loadSceneList(data.project_id)`

2. **`setupEventListeners()`**
   - 재합성 버튼 이벤트 리스너 추가

## 사용자 플로우

1. **영상 생성 완료** → 결과 페이지로 자동 이동
2. **씬 관리 섹션 확인** → 모든 씬의 상태와 내용 확인
3. **씬 재생성** (선택적):
   - 마음에 들지 않는 씬의 "재생성" 버튼 클릭
   - 재생성 진행 중 버튼 비활성화 및 상태 표시
   - 완료 후 "영상 재합성" 버튼 표시
4. **영상 재합성** (선택적):
   - "영상 재합성" 버튼 클릭
   - 수정된 씬이 반영된 새 영상 생성
   - 비디오 플레이어 자동 갱신

## API 엔드포인트 (이미 구현됨)

- `GET /api/projects/{project_id}/scenes` - 씬 목록 조회
- `POST /api/projects/{project_id}/scenes/{scene_id}/regenerate` - 씬 재생성
- `POST /api/projects/{project_id}/recompose` - 영상 재합성

## 테스트 시나리오

### 기본 기능
- [ ] 영상 생성 완료 후 씬 관리 섹션이 표시되는가?
- [ ] 모든 씬이 올바른 정보와 함께 표시되는가?
- [ ] 씬의 상태(완료/실패)가 올바르게 표시되는가?

### 씬 재생성
- [ ] 재생성 버튼 클릭 시 API 호출이 정상적으로 되는가?
- [ ] 재생성 중 버튼이 비활성화되고 상태가 표시되는가?
- [ ] 재생성 완료 후 씬 목록이 자동으로 갱신되는가?
- [ ] 재생성 완료 후 재합성 버튼이 나타나는가?

### 영상 재합성
- [ ] 재합성 버튼이 씬 재생성 후에만 표시되는가?
- [ ] 재합성 완료 후 비디오 플레이어가 새 영상으로 갱신되는가?
- [ ] 재합성 버튼이 완료 후 다시 숨겨지는가?

### 에러 처리
- [ ] 씬 재생성 실패 시 에러 메시지가 표시되는가?
- [ ] 재합성 실패 시 적절한 피드백이 제공되는가?
- [ ] 네트워크 오류 시 UI가 복구되는가?

## 주의사항

1. **백엔드 API 의존성**: 백엔드의 씬 관리 API가 정상적으로 동작해야 함
2. **재생성 시간**: 씬 재생성은 시간이 걸리므로 사용자에게 명확한 피드백 제공
3. **재합성 타이밍**: 재합성 버튼은 씬 재생성 후에만 표시하여 혼란 방지
4. **캐시 문제**: 비디오 플레이어 갱신 시 타임스탬프 파라미터로 캐시 우회

## 파일 변경 사항

### 수정된 파일
1. `web/templates/index.html` (lines ~600-623, ~276-394)
   - 씬 관리 HTML 섹션 추가
   - CSS 스타일 추가

2. `web/static/app.js` (lines ~80, ~488, ~697-881)
   - 씬 관리 함수 4개 추가
   - showResults 함수를 async로 변경
   - setupEventListeners에 재합성 버튼 리스너 추가

### 새로 생성된 파일
- 없음 (기존 파일 수정만)

## 완료 상태

✅ HTML 구조 추가 완료
✅ CSS 스타일 추가 완료
✅ JavaScript 기능 구현 완료
✅ 이벤트 리스너 연결 완료
✅ 구문 검사 완료

## 다음 단계

1. 로컬 개발 서버에서 테스트
2. 씬 재생성 기능 동작 확인
3. 영상 재합성 기능 동작 확인
4. 에러 케이스 처리 확인
5. 프로덕션 배포
