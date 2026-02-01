# Python 3.11 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치 (FFmpeg, git 등)
# Storycut은 FFmpeg가 필수입니다.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    fonts-liberation \
    fonts-nanum \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# 폰트 캐시 갱신 (선택사항, 자막 폰트용)
RUN fc-cache -f -v

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 전체 복사
COPY . .

# 포트 노출 (Railway 등에서는 PORT 환경변수로 덮어씌워짐)
ENV PORT=8000
EXPOSE 8000

# 출력 디렉토리 생성
RUN mkdir -p outputs

# 실행 명령
CMD ["python", "api_server.py"]
