@echo off
title NeuroQuant

echo ============================================
echo         NeuroQuant Launcher
echo ============================================
echo.

py -3.9 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.9 not found!
    echo Install Python 3.9 from python.org
    pause
    exit /b 1
)

echo [OK] Python 3.9 found
echo Starting GUI...
echo.

py -3.9 "%~dp0gui.py"

if errorlevel 1 (
    echo.
    echo [ERROR] GUI crashed
    pause
)
