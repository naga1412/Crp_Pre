@echo off
title SMC Pro — Ollama DeepSeek Server

echo.
echo  ========================================================
echo   SMC Pro Analyzer — Starting DeepSeek-R1 Local AI
echo  ========================================================
echo.
echo  Model : deepseek-r1:8b
echo  Server: http://localhost:11434
echo  CORS  : Open for all origins (required for GitHub Pages)
echo.
echo  Leave this window OPEN while using the app.
echo  Close it to stop the AI server.
echo  ========================================================
echo.

:: Set CORS to allow all origins (needed when app runs on GitHub Pages)
set OLLAMA_ORIGINS=*

:: Set Ollama host
set OLLAMA_HOST=0.0.0.0

:: Start Ollama server from its install location
"C:\Users\nagar\AppData\Local\Programs\Ollama\ollama.exe" serve

echo.
echo  Ollama stopped. Press any key to exit.
pause >nul
