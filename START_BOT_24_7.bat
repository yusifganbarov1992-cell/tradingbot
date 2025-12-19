@echo off
chcp 65001 > nul
echo ============================================================
echo   NEXUSTRADER - БЕСКОНЕЧНАЯ РАБОТА 24/7
echo   Автоматический перезапуск при остановке
echo ============================================================
echo.

:RESTART_LOOP

echo [%TIME%] Запуск бота...
.venv\Scripts\python.exe trading_bot.py

echo.
echo [%TIME%] Бот остановился. Перезапуск через 3 секунды...
timeout /t 3 /nobreak > nul

goto RESTART_LOOP
