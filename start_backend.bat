@echo off
echo Starting MediVision-AI Backend API...

cd /d "%~dp0backend"

uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause