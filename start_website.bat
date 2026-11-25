@echo off
title PongRL Pro - Full Stack System
echo ========================================================
echo   PongRL Pro: Deep Reinforcement Learning System
echo ========================================================
echo.
echo [1/3] Launching FastAPI Python Inference Server on port 8000...
start "PongRL FastAPI Backend" cmd /k "cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 3 /nobreak >nul

echo [2/3] Launching Next.js Minimal Research Frontend on port 3000...
start "PongRL Next.js Frontend" cmd /k "cd frontend && npm run dev"

timeout /t 4 /nobreak >nul

echo [3/3] Opening PongRL Pro Dashboard in default browser...
start http://localhost:3000

echo.
echo ========================================================
echo   PongRL Pro is now running!
echo   Frontend : http://localhost:3000
echo   Backend  : http://localhost:8000/docs (Swagger UI)
echo ========================================================
pause
