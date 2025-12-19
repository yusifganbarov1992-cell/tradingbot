@echo off
:: NexusTrader AI - Auto Start Script
:: Runs at Windows startup

cd /d "C:\Users\yusif\OneDrive\Desktop\trader"

:: Activate virtual environment and start bot
call .venv\Scripts\activate.bat

:: Start bot with watchdog (restarts if crashes)
python bot_watchdog.py

pause
