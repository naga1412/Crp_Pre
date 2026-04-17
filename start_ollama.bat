@echo off
title SMC Pro — Ollama DeepSeek Server

echo.
echo  ========================================================
echo   SMC Pro Analyzer — Starting DeepSeek-R1 Local AI
echo  ========================================================
echo.
echo  Model : deepseek-r1:8b
echo  Server: http://localhost:11434
echo  CORS  : Open for all origins (GitHub Pages compatible)
echo.
echo  Leave this window OPEN while using the app.
echo  Close it to stop the AI server.
echo  ========================================================
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0launch_ollama.ps1"

echo.
echo  Ollama stopped. Press any key to exit.
pause >nul
