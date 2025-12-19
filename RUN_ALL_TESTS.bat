@echo off
chcp 65001 > nul
echo ============================================================
echo   FULL AGENT TEST SUITE
echo   Running all tests sequentially
echo ============================================================
echo.

set TESTS_PASSED=0
set TESTS_FAILED=0

REM Test 1: Full Agent Test
echo [1/3] Running Full Agent Test...
echo ------------------------------------------------------------
.venv\Scripts\python.exe test_full_agent.py
if %ERRORLEVEL% EQU 0 (
    echo ✅ Full Agent Test: PASSED
    set /a TESTS_PASSED+=1
) else (
    echo ❌ Full Agent Test: FAILED
    set /a TESTS_FAILED+=1
)
echo.
echo.

REM Test 2: Live Scanning Test
echo [2/3] Running Live Scanning Test...
echo ------------------------------------------------------------
.venv\Scripts\python.exe test_live_scanning.py
if %ERRORLEVEL% EQU 0 (
    echo ✅ Live Scanning Test: PASSED
    set /a TESTS_PASSED+=1
) else (
    echo ❌ Live Scanning Test: FAILED
    set /a TESTS_FAILED+=1
)
echo.
echo.

REM Test 3: Telegram Integration Test
echo [3/3] Running Telegram Integration Test...
echo ------------------------------------------------------------
.venv\Scripts\python.exe test_telegram_integration.py
if %ERRORLEVEL% EQU 0 (
    echo ✅ Telegram Integration Test: PASSED
    set /a TESTS_PASSED+=1
) else (
    echo ❌ Telegram Integration Test: FAILED
    set /a TESTS_FAILED+=1
)
echo.
echo.

REM Summary
echo ============================================================
echo   TEST SUITE SUMMARY
echo ============================================================
echo Total tests: 3
echo Passed: %TESTS_PASSED%
echo Failed: %TESTS_FAILED%
echo.

if %TESTS_FAILED% EQU 0 (
    echo 🎉 ALL TESTS PASSED! Agent is fully functional!
    echo.
    echo Next steps:
    echo 1. Bot is ready for production use
    echo 2. Monitor first trades carefully
    echo 3. Check Telegram for signals every 5 minutes
) else (
    echo ⚠️  Some tests failed. Review errors above.
    echo.
    echo Troubleshooting:
    echo 1. Check .env file has all API keys
    echo 2. Verify internet connection
    echo 3. Check Binance/Telegram API status
)
echo ============================================================
echo.

pause
