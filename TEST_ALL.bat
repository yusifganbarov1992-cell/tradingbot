@echo off
echo ============================================
echo   ПРОВЕРКА РАБОТОСПОСОБНОСТИ NEXUSTRADER
echo ============================================
echo.

cd /d "%~dp0"

echo [1/5] Проверка Python...
.venv\Scripts\python.exe --version
if %errorlevel% neq 0 (
    echo ОШИБКА: Python не найден!
    pause
    exit /b 1
)
echo OK
echo.

echo [2/5] Проверка зависимостей...
.venv\Scripts\python.exe -c "import ccxt, telegram, pandas, numpy, dotenv; print('OK')"
if %errorlevel% neq 0 (
    echo ОШИБКА: Не хватает зависимостей!
    pause
    exit /b 1
)
echo.

echo [3/5] Проверка баланса Binance...
.venv\Scripts\python.exe check_balance.py
echo.

echo [4/5] Проверка Telegram бота...
.venv\Scripts\python.exe -c "import requests, os; from dotenv import load_dotenv; load_dotenv(); r = requests.get(f'https://api.telegram.org/bot{os.getenv(\"TELEGRAM_BOT_TOKEN\")}/getMe'); print(r.json()['result']['username'] if r.json()['ok'] else 'ERROR')"
echo.

echo [5/5] Проверка синтаксиса trading_bot.py...
.venv\Scripts\python.exe -m py_compile trading_bot.py
if %errorlevel% neq 0 (
    echo ОШИБКА: Синтаксическая ошибка в trading_bot.py!
    pause
    exit /b 1
)
echo OK
echo.

echo ============================================
echo   ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!
echo ============================================
echo.
echo Для запуска бота выполните: RUN_BOT.bat
echo.
pause
