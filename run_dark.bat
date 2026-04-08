@echo off
chcp 65001 >nul
title NeuroQuant (Dark)

py -3.9 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ПОМИЛКА] Python 3.9 не знайдено!
    pause
    exit /b 1
)

py -3.9 "%~dp0gui.py" --dark
