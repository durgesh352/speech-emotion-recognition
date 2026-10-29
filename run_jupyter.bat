@echo off
title Launch JupyterLab - Speech Emotion Recognition
echo ============================================================
echo   Launching JupyterLab for Speech Emotion Recognition
echo ============================================================
echo.
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    pause
    exit /b
)

call venv\Scripts\activate.bat
echo Starting JupyterLab in project directory...
jupyter lab notebook_demo.ipynb
pause
