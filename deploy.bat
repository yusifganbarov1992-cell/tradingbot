@echo off
REM NexusTrader AI v3.0 - Windows One-Click Deployment Script

echo ========================================
echo   NexusTrader AI v3.0 Deployment
echo ========================================
echo.

REM Check for Docker
where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker not found! Please install Docker Desktop first.
    echo Visit: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

where docker-compose >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker Compose not found! Please install Docker Compose.
    pause
    exit /b 1
)

echo [OK] Docker found
echo [OK] Docker Compose found
echo.

REM Check .env file
if not exist ".env" (
    echo [WARNING] .env file not found!
    if exist ".env.template" (
        echo Creating .env from template...
        copy .env.template .env
        echo [OK] .env file created
        echo.
        echo [ACTION REQUIRED] Please edit .env file with your API keys!
        echo Press any key after editing .env file...
        pause >nul
    ) else (
        echo [ERROR] .env.template not found!
        pause
        exit /b 1
    )
) else (
    echo [OK] .env file exists
)
echo.

REM Create directories
echo Creating directories...
if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "dashboard" mkdir dashboard
echo [OK] Directories created
echo.

REM Test connections (optional)
echo Do you want to test API connections? (y/n)
set /p TEST_APIS="> "

if /i "%TEST_APIS%"=="y" (
    echo Testing Binance connection...
    python test_binance.py 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo [OK] Binance connection successful
    ) else (
        echo [WARNING] Binance test failed
        echo Continue anyway? (y/n)
        set /p CONTINUE="> "
        if /i not "%CONTINUE%"=="y" exit /b 1
    )

    echo Testing Supabase connection...
    python test_simple_supabase.py 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo [OK] Supabase connection successful
    ) else (
        echo [WARNING] Supabase test failed
        echo Continue anyway? (y/n)
        set /p CONTINUE="> "
        if /i not "%CONTINUE%"=="y" exit /b 1
    )

    echo Testing Telegram connection...
    python check_telegram.py 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo [OK] Telegram connection successful
    ) else (
        echo [WARNING] Telegram test failed
        echo Continue anyway? (y/n)
        set /p CONTINUE="> "
        if /i not "%CONTINUE%"=="y" exit /b 1
    )
)
echo.

REM Build Docker images
echo Building Docker images...
echo This may take several minutes...
docker-compose build
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker build failed!
    pause
    exit /b 1
)
echo [OK] Docker images built successfully
echo.

REM Start services
echo Starting services...
docker-compose up -d
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to start services!
    pause
    exit /b 1
)
echo [OK] Services started successfully
echo.

REM Wait for services
echo Waiting for services to be healthy...
echo This may take 30-60 seconds...
timeout /t 15 /nobreak >nul
echo.

REM Display status
echo Deployment status:
docker-compose ps
echo.

REM Show access information
echo ========================================
echo   Deployment Complete!
echo ========================================
echo.
echo Access Points:
echo   Dashboard: http://localhost:8501
echo   Bot logs: docker-compose logs -f trading-bot
echo   Dashboard logs: docker-compose logs -f dashboard
echo.
echo Useful Commands:
echo   View logs: docker-compose logs -f
echo   Stop services: docker-compose down
echo   Restart: docker-compose restart
echo   View status: docker-compose ps
echo.
echo Telegram Commands:
echo   /start - Start bot
echo   /status - Check status
echo   /balance - Check balance
echo   /help - All commands
echo.
echo [IMPORTANT REMINDERS]
echo 1. Bot is in PAPER TRADING mode by default
echo 2. Set AUTO_TRADE=true in .env to enable auto-trading
echo 3. Monitor the dashboard and Telegram closely
echo 4. Start with small amounts when going live
echo 5. Review trades regularly
echo.
echo Good luck and happy trading!
echo.

REM Open dashboard
echo Open dashboard in browser? (y/n)
set /p OPEN_BROWSER="> "

if /i "%OPEN_BROWSER%"=="y" (
    start http://localhost:8501
)

echo.
echo Deployment completed successfully!
pause
