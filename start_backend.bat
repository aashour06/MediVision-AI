@echo off
echo Starting MediVision-AI Backend API...

cd /d "%~dp0backend"

if not exist ".env" if not exist "..\\.env" (
    echo [INFO] No .env file found. To enable Gemini AI Chat, create backend\.env with GEMINI_API_KEY=your_key
)

uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause