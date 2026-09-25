@echo off
echo Starting MediVision-AI Backend API...
cd backend
..\.venv\Scripts\uvicorn.exe main:app --reload
