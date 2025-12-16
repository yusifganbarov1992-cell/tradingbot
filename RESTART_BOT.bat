@echo off
echo Cleaning old pending signals...
python -c "import sqlite3; conn = sqlite3.connect('trading_history.db'); cursor = conn.cursor(); cursor.execute('UPDATE signals SET status=''expired'' WHERE status=''pending'''); affected = cursor.rowcount; conn.commit(); conn.close(); print(f'Marked {affected} old signals as expired')"

echo.
echo Restarting bot...
taskkill /F /IM python.exe /FI "MEMUSAGE gt 100000" 2>nul
timeout /t 2 /nobreak >nul

start /min powershell -Command "cd '%cd%'; .\.venv\Scripts\python.exe trading_bot.py"

echo.
echo Bot restarted! Check Telegram for signals in 1-2 minutes.
timeout /t 3 /nobreak >nul
