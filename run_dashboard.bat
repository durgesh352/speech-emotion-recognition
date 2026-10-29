@echo off
title Speech Emotion Recognition Dashboard
echo ============================================================
echo   Launching Speech Emotion Recognition System (Hindi & English)
echo ============================================================
echo.
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found! Please run setup first.
    pause
    exit /b
)

call venv\Scripts\activate.bat
echo Starting Streamlit web application on http://localhost:8501...
streamlit run app.py
pause
