```bat
@echo off
echo Starting MediVision-AI Backend API...

cd /d "%~dp0backend"

uv run --project .. --python "..\.venv\Scripts\python.exe" uvicorn main:app --reload

pause
```