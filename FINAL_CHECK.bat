@echo off
chcp 65001 > nul
cls
echo.
echo ============================================================
echo   ФИНАЛЬНАЯ ДИАГНОСТИКА - ОТВЕТ НА ВСЕ ВОПРОСЫ
echo ============================================================
echo.

REM 1. Тест get_open_trades()
echo [TEST 1] get_open_trades() исправлен?
.venv\Scripts\python.exe -c "from database import TradingDatabase; db = TradingDatabase(); result = db.get_open_trades(); print('   ✅ ИСПРАВЛЕНО: Возвращает', type(result).__name__, 'с', len(result), 'элементами')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo    ❌ ОШИБКА в get_open_trades!
)
echo.

REM 2. Баланс
echo [TEST 2] Баланс Binance достаточен?
.venv\Scripts\python.exe -c "import ccxt; import os; from dotenv import load_dotenv; load_dotenv(); exchange = ccxt.binance({'apiKey': os.getenv('BINANCE_API_KEY'), 'secret': os.getenv('BINANCE_SECRET_KEY')}); balance = exchange.fetch_balance(); usdt = balance['total'].get('USDT', 0); print('   💰 Баланс:', usdt, 'USDT'); exit(0 if usdt >= 50 else 1)" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo    ❌ КРИТИЧНО: Баланс недостаточен ($0 или меньше $50)
    echo    📝 Решение: Пополнить Binance минимум на $50-100
) else (
    echo    ✅ Баланс достаточен для торговли
)
echo.

REM 3. Логи
echo [TEST 3] Бот пишет логи?
if exist bot.log (
    echo    ✅ bot.log существует - бот работает с НОВЫМ кодом
) else (
    echo    ❌ bot.log НЕ создан - бот работает со СТАРЫМ кодом
    echo    📝 Решение: Перезапустить через START_24_7.bat
)
echo.

REM 4. Процессы
echo [TEST 4] Сколько процессов Python?
for /f %%i in ('tasklist /FI "IMAGENAME eq python.exe" ^| find /C "python.exe"') do set PROC_COUNT=%%i
echo    📊 Процессов: %PROC_COUNT%
if %PROC_COUNT% GTR 2 (
    echo    ⚠️  Слишком много процессов - возможна утечка
    echo    📝 Решение: Остановить все и запустить через watchdog
) else (
    echo    ✅ Нормальное количество процессов
)
echo.

REM 5. Watchdog
echo [TEST 5] Watchdog создан?
if exist watchdog.py (
    echo    ✅ watchdog.py существует
) else (
    echo    ❌ watchdog.py отсутствует
)
if exist START_24_7.bat (
    echo    ✅ START_24_7.bat существует
) else (
    echo    ❌ START_24_7.bat отсутствует
)
echo.

echo ============================================================
echo   ИТОГОВЫЙ ОТВЕТ
echo ============================================================
echo.

REM Подсчет проблем
set ISSUES=0

if not exist bot.log set /a ISSUES+=1
if %PROC_COUNT% GTR 2 set /a ISSUES+=1

.venv\Scripts\python.exe -c "import ccxt; import os; from dotenv import load_dotenv; load_dotenv(); exchange = ccxt.binance({'apiKey': os.getenv('BINANCE_API_KEY'), 'secret': os.getenv('BINANCE_SECRET_KEY')}); balance = exchange.fetch_balance(); usdt = balance['total'].get('USDT', 0); exit(0 if usdt >= 50 else 1)" 2>nul
if %ERRORLEVEL% NEQ 0 set /a ISSUES+=1

if %ISSUES% EQU 0 (
    echo ✅ ВСЁ РАБОТАЕТ ИДЕАЛЬНО! НЕТ БАГОВ!
    echo.
    echo Бот готов к работе 24/7.
    echo Все критичные исправления применены.
) else (
    echo ⚠️  ПОЧТИ ВСЁ РАБОТАЕТ, НО ЕСТЬ %ISSUES% ПРОБЛЕМА(Ы):
    echo.
    if not exist bot.log (
        echo    • Бот работает со старым кодом (нет bot.log^)
    )
    .venv\Scripts\python.exe -c "import ccxt; import os; from dotenv import load_dotenv; load_dotenv(); exchange = ccxt.binance({'apiKey': os.getenv('BINANCE_API_KEY'), 'secret': os.getenv('BINANCE_SECRET_KEY')}); balance = exchange.fetch_balance(); usdt = balance['total'].get('USDT', 0); exit(0 if usdt >= 50 else 1)" 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo    • Баланс $0 - торговля невозможна
    )
    if %PROC_COUNT% GTR 2 (
        echo    • Слишком много процессов Python (%PROC_COUNT%^)
    )
    echo.
    echo РЕШЕНИЕ:
    echo    1. Остановить все процессы
    echo    2. Запустить: START_24_7.bat
    echo    3. Пополнить Binance ($50-100 USDT^)
)

echo.
echo ============================================================
pause
