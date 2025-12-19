@echo off
chcp 65001 > nul

echo ============================================================
echo   ОСТАНОВКА И ПЕРЕЗАПУСК БОТА
echo ============================================================
echo.

echo 1. Останавливаем все процессы бота...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *trading_bot*" 2>nul
taskkill /F /IM python.exe /FI "IMAGENAME eq python.exe" 2>nul

timeout /t 2 /nobreak > nul

echo.
echo 2. Очищаем старые pending signals...
.venv\Scripts\python.exe -c "from database import TradingDatabase; db = TradingDatabase(); conn = db.conn; conn.execute('UPDATE signals SET status = \"expired\" WHERE status = \"pending\" AND datetime(timestamp) < datetime(\"now\", \"-1 hour\")'); conn.commit(); print('OK')"

echo.
echo 3. Запускаем watchdog для 24/7 работы...
start "NexusTrader 24/7" .venv\Scripts\python.exe watchdog.py

echo.
echo ============================================================
echo   ✅ БОТ ПЕРЕЗАПУЩЕН В РЕЖИМЕ 24/7
echo   Окно watchdog откроется в новом окне
echo   Закройте это окно - бот будет работать в фоне
echo ============================================================
echo.

timeout /t 3
