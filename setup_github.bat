@echo off
title GitHub Repository Setup
echo ============================================================
echo   GitHub Setup for Speech Emotion Recognition
echo ============================================================
echo.

set /p REPO_URL="Enter your GitHub Repository URL (or press Enter to commit locally only): "

if "%REPO_URL%"=="" (
    python setup_git.py
) else (
    python setup_git.py --repo %REPO_URL%
)

echo.
pause
