@echo off
:START
echo ========================================
echo    NEXUSTRADER - AUTO-RESTART BOT
echo    Will automatically restart on crashes
echo ========================================
echo.

REM Activate venv and run bot
cd /d "%~dp0"
call .venv\Scripts\activate.bat

echo [%TIME%] Starting bot...
python trading_bot.py

echo.
echo [%TIME%] Bot stopped. Waiting 5 seconds before restart...
timeout /t 5 /nobreak >nul

goto START
