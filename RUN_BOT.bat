@echo off
echo ========================================
echo    NEXUSTRADER - TRADING BOT
echo    WARNING: Real money trading!
echo ========================================
echo.

cd /d "%~dp0"

echo [INFO] Starting bot...
echo [INFO] Press Ctrl+C to stop
echo.

C:\Users\yusif\OneDrive\Desktop\trader\.venv\Scripts\python.exe trading_bot.py

echo.
echo [INFO] Bot stopped
pause
