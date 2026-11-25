@echo off
title PongRL Pro - Live Public Link
echo =====================================================================
echo  Starting Streamlit and Cloudflare Public Tunnel...
echo =====================================================================
cd /d "%~dp0"
start "" /b streamlit run DLG-AI-2021-forward\DLG-AI-2021\CAPSTONE\PongRL-Pro\dashboard\app.py --server.port=8501
timeout /t 3 /nobreak >nul
.\cloudflared.exe tunnel --protocol http2 --url http://localhost:8501
pause
