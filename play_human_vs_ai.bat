@echo off
title PongRL Pro - Human vs AI Agent
echo =====================================================================
echo  PongRL Pro: Human vs. Trained AI Agent Match
echo  Controls:
echo    W / UP Arrow   : Move Paddle UP
echo    S / DOWN Arrow : Move Paddle DOWN
echo    SPACE          : Pause / Resume
echo    R              : Reset Match
echo    ESC            : Exit
echo =====================================================================
echo.
cd /d "%~dp0"
python scripts\play_human_vs_ai.py
pause

