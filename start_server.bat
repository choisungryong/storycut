@echo off
echo.
echo =====================================================================
echo    STORYCUT Web UI Server Starting...
echo =====================================================================
echo.

REM 환경변수 로드 확인
if not exist .env (
    echo [WARNING] .env file not found!
    echo           Please copy .env.example to .env and configure API keys.
    echo.
    pause
)

REM Python 버전 확인
python --version
echo.

REM 필수 패키지 확인
echo Checking dependencies...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo [ERROR] fastapi not installed!
    echo         Run: pip install -r requirements.txt
    pause
    exit /b 1
)

REM 서버 시작
echo.
echo Starting server on http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

python api_server.py

pause
