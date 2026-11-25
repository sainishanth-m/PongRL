@echo off
title PongRL Pro - Deep Reinforcement Learning Dashboard
color 0B

echo ============================================================
echo   PongRL Pro - AI Research and Demonstration Dashboard
echo   Deep Reinforcement Learning Atari Pong
echo ============================================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in PATH. Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

echo [INFO] Python environment detected.
echo [INFO] Starting PongRL Pro Streamlit Command Center...
echo.

cd /d "%~dp0"
python -m streamlit run dashboard/app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Streamlit encountered an issue. Make sure dependencies are installed:
    echo        pip install -r requirements.txt
    pause
)

