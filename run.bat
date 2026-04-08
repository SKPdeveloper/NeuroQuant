@echo off
chcp 65001 >nul
title NeuroQuant

echo ============================================
echo         NeuroQuant Launcher
echo ============================================
echo.

:: Перевірка Python 3.9
py -3.9 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ПОМИЛКА] Python 3.9 не знайдено!
    echo Встановіть Python 3.9 з python.org
    pause
    exit /b 1
)

echo [OK] Python 3.9 знайдено
echo.

:: Запуск GUI
echo Запуск NeuroQuant GUI...
echo.
py -3.9 "%~dp0gui.py"

if %errorlevel% neq 0 (
    echo.
    echo [ПОМИЛКА] GUI завершився з помилкою
    pause
)
