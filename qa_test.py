"""
🔍 QA AUTOMATION TEST SUITE
Автоматическое тестирование всех компонентов проекта
"""
import os
import sys
from dotenv import load_dotenv

# Цвета для консоли
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_test(name, passed, details=""):
    status = f"{Colors.GREEN}✅ PASS{Colors.END}" if passed else f"{Colors.RED}❌ FAIL{Colors.END}"
    print(f"{status} | {name}")
    if details and not passed:
        print(f"      └─ {Colors.YELLOW}{details}{Colors.END}")

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

# Счётчики
total_tests = 0
passed_tests = 0
failed_tests = 0
warnings = []

def run_test(name, test_func):
    global total_tests, passed_tests, failed_tests
    total_tests += 1
    try:
        result, details = test_func()
        if result:
            passed_tests += 1
        else:
            failed_tests += 1
        print_test(name, result, details)
        return result
    except Exception as e:
        failed_tests += 1
        print_test(name, False, str(e))
        return False

# ============= ТЕСТЫ =============

def test_python_version():
    """Проверка версии Python"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor} (требуется 3.10+)"

def test_env_file():
    """Проверка наличия .env файла"""
    if os.path.exists('.env'):
        load_dotenv()
        return True, "Файл найден и загружен"
    return False, ".env файл не найден"

def test_binance_api_keys():
    """Проверка ключей Binance API"""
    api_key = os.getenv('BINANCE_API_KEY')
    secret = os.getenv('BINANCE_SECRET_KEY')
    if api_key and secret and len(api_key) > 20:
        return True, f"Ключи установлены (длина: {len(api_key)} символов)"
    return False, "API ключи не найдены или неполные"

def test_telegram_token():
    """Проверка Telegram Bot Token"""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if token and ':' in token:
        return True, f"Token: {token[:10]}..."
    return False, "Telegram token не найден"

def test_openai_key():
    """Проверка OpenAI API Key"""
    key = os.getenv('OPENAI_API_KEY')
    if key and key.startswith('sk-'):
        return True, f"Key: {key[:20]}..."
    return False, "OpenAI key не найден или неверный формат"

def test_import_ccxt():
    """Тест импорта CCXT"""
    try:
        import ccxt
        return True, f"CCXT v{ccxt.__version__}"
    except ImportError as e:
        return False, str(e)

def test_import_telegram():
    """Тест импорта python-telegram-bot"""
    try:
        import telegram
        return True, f"python-telegram-bot v{telegram.__version__}"
    except ImportError as e:
        return False, str(e)

def test_import_pandas():
    """Тест импорта Pandas"""
    try:
        import pandas as pd
        return True, f"Pandas v{pd.__version__}"
    except ImportError as e:
        return False, str(e)

def test_import_numpy():
    """Тест импорта NumPy"""
    try:
        import numpy as np
        return True, f"NumPy v{np.__version__}"
    except ImportError as e:
        return False, str(e)

def test_import_openai():
    """Тест импорта OpenAI"""
    try:
        import openai
        return True, f"OpenAI v{openai.__version__}"
    except ImportError as e:
        return False, str(e)

def test_binance_connection():
    """Тест подключения к Binance"""
    try:
        import ccxt
        exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET_KEY'),
        })
        balance = exchange.fetch_balance()
        usdt = balance['total'].get('USDT', 0)
        
        global warnings
        if usdt < 10:
            warnings.append(f"⚠️  Баланс ${usdt:.2f} < $10 - торговля невозможна")
        
        return True, f"Баланс USDT: ${usdt:.2f}"
    except Exception as e:
        return False, str(e)

def test_telegram_bot():
    """Тест подключения к Telegram"""
    try:
        import requests
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        response = requests.get(f"https://api.telegram.org/bot{token}/getMe")
        data = response.json()
        if data['ok']:
            bot = data['result']
            return True, f"@{bot['username']} (ID: {bot['id']})"
        return False, "Бот не отвечает"
    except Exception as e:
        return False, str(e)

def test_trading_bot_syntax():
    """Тест синтаксиса trading_bot.py"""
    try:
        import py_compile
        py_compile.compile('trading_bot.py', doraise=True)
        return True, "Синтаксис корректен"
    except Exception as e:
        return False, str(e)

def test_trading_bot_imports():
    """Тест импорта trading_bot модуля"""
    try:
        # Добавляем текущую директорию в путь
        sys.path.insert(0, os.getcwd())
        import trading_bot
        
        # Проверяем наличие ключевых классов
        assert hasattr(trading_bot, 'TradingAgent'), "TradingAgent не найден"
        assert hasattr(trading_bot, 'RiskEngine'), "RiskEngine не найден"
        assert hasattr(trading_bot, 'MetricsTracker'), "MetricsTracker не найден"
        
        return True, "Все классы найдены"
    except Exception as e:
        return False, str(e)

def test_risk_engine():
    """Тест RiskEngine"""
    try:
        sys.path.insert(0, os.getcwd())
        from trading_bot import RiskEngine
        
        risk = RiskEngine(max_position_size_pct=10, max_total_exposure_pct=30)
        balance = 100.0
        signal_strength = 5
        
        size = risk.calculate_position_size(balance, signal_strength)
        fees = risk.calculate_fees(size)
        
        assert size > 0, "Размер позиции должен быть > 0"
        assert size <= balance * 0.1, "Размер превышает 10%"
        assert fees > 0, "Комиссия должна быть > 0"
        
        return True, f"Позиция: ${size:.2f}, Комиссия: ${fees:.4f}"
    except Exception as e:
        return False, str(e)

def test_metrics_tracker():
    """Тест MetricsTracker"""
    try:
        sys.path.insert(0, os.getcwd())
        from trading_bot import MetricsTracker
        
        metrics = MetricsTracker()
        metrics.add_trade('BTC/USDT', 'BUY', 10.0, 0.5, 0.01)
        metrics.add_trade('ETH/USDT', 'BUY', 5.0, -0.2, 0.005)
        
        summary = metrics.get_summary()
        assert summary['total_trades'] == 2, "Должно быть 2 сделки"
        assert summary['win_rate'] == 50.0, "Win rate должен быть 50%"
        
        return True, f"Trades: {summary['total_trades']}, Win Rate: {summary['win_rate']}%"
    except Exception as e:
        return False, str(e)

def test_file_structure():
    """Проверка структуры проекта"""
    required_files = [
        'trading_bot.py',
        'requirements.txt',
        'package.json',
        '.env',
        'RUN_BOT.bat',
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if not missing:
        return True, f"Все {len(required_files)} файлов на месте"
    return False, f"Отсутствуют: {', '.join(missing)}"

def test_documentation():
    """Проверка наличия документации"""
    docs = [
        'README.md',
        'SAFETY_GUIDE.md',
        'QUICK_START.md',
    ]
    
    found = [d for d in docs if os.path.exists(d)]
    
    if len(found) >= 2:
        return True, f"Найдено {len(found)}/{len(docs)} файлов"
    return False, f"Недостаточно документации ({len(found)}/{len(docs)})"

# ============= ЗАПУСК ТЕСТОВ =============

if __name__ == "__main__":
    print_header("QA AUTOMATION TEST SUITE")
    print(f"{Colors.BOLD}NexusTrader AI - Quality Assurance Report{Colors.END}\n")
    
    # 1. Окружение
    print_header("1️⃣  ENVIRONMENT CHECKS")
    run_test("Python Version", test_python_version)
    run_test(".env File", test_env_file)
    
    # 2. Конфигурация
    print_header("2️⃣  CONFIGURATION")
    run_test("Binance API Keys", test_binance_api_keys)
    run_test("Telegram Bot Token", test_telegram_token)
    run_test("OpenAI API Key", test_openai_key)
    
    # 3. Зависимости
    print_header("3️⃣  DEPENDENCIES")
    run_test("CCXT Library", test_import_ccxt)
    run_test("Telegram Library", test_import_telegram)
    run_test("Pandas Library", test_import_pandas)
    run_test("NumPy Library", test_import_numpy)
    run_test("OpenAI Library", test_import_openai)
    
    # 4. Внешние подключения
    print_header("4️⃣  EXTERNAL CONNECTIONS")
    run_test("Binance API Connection", test_binance_connection)
    run_test("Telegram Bot API", test_telegram_bot)
    
    # 5. Код
    print_header("5️⃣  CODE QUALITY")
    run_test("trading_bot.py Syntax", test_trading_bot_syntax)
    run_test("Module Import", test_trading_bot_imports)
    run_test("RiskEngine Component", test_risk_engine)
    run_test("MetricsTracker Component", test_metrics_tracker)
    
    # 6. Структура
    print_header("6️⃣  PROJECT STRUCTURE")
    run_test("Required Files", test_file_structure)
    run_test("Documentation", test_documentation)
    
    # ИТОГИ
    print_header("FINAL REPORT")
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total Tests: {Colors.BOLD}{total_tests}{Colors.END}")
    print(f"Passed: {Colors.GREEN}{passed_tests}{Colors.END}")
    print(f"Failed: {Colors.RED}{failed_tests}{Colors.END}")
    print(f"Success Rate: {Colors.BOLD}{success_rate:.1f}%{Colors.END}\n")
    
    # Предупреждения
    if warnings:
        print(f"{Colors.YELLOW}⚠️  WARNINGS:{Colors.END}")
        for warning in warnings:
            print(f"   {warning}")
        print()
    
    # Вердикт
    if success_rate >= 90:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ СИСТЕМА ГОТОВА К РАБОТЕ!{Colors.END}")
    elif success_rate >= 70:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  СИСТЕМА РАБОТАЕТ С ПРЕДУПРЕЖДЕНИЯМИ{Colors.END}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ КРИТИЧЕСКИЕ ОШИБКИ - ТРЕБУЕТСЯ ИСПРАВЛЕНИЕ{Colors.END}")
    
    print()
    sys.exit(0 if failed_tests == 0 else 1)
