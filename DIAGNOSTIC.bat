@echo off
chcp 65001 > nul
echo === DIAGNOSTIC CHECK ===
echo.

echo 1. Emergency Stop Status:
.venv\Scripts\python.exe -c "from database import TradingDatabase; db = TradingDatabase(); print('emergency_stop =', db.load_emergency_stop())"
echo.

echo 2. Bot Process:
powershell -Command "Get-Process python -ErrorAction SilentlyContinue | Where-Object {$_.WS -gt 50MB} | Select-Object Id, @{Name='Memory(MB)';Expression={[math]::Round($_.WS/1MB,2)}}, StartTime"
echo.

echo 3. Last 3 Signals from DB:
.venv\Scripts\python.exe -c "import sqlite3; conn = sqlite3.connect('trading_history.db'); cursor = conn.cursor(); cursor.execute('SELECT symbol, signal_type, ai_confidence, status, timestamp FROM signals ORDER BY timestamp DESC LIMIT 3'); rows = cursor.fetchall(); print('Last signals:'); [print(f'  {row}') for row in rows]; conn.close()"
echo.

echo 4. Pending Signals Count:
.venv\Scripts\python.exe -c "import sqlite3; conn = sqlite3.connect('trading_history.db'); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM signals WHERE status=''pending'''); print('Pending:', cursor.fetchone()[0]); conn.close()"
echo.

pause
