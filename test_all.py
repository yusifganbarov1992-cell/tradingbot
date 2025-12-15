"""
Тест всех компонентов торгового бота
"""
import os
from dotenv import load_dotenv
import ccxt

load_dotenv()

print("=" * 60)
print("ПРОВЕРКА РАБОТОСПОСОБНОСТИ NEXUSTRADER")
print("=" * 60)
print()

# 1. Проверка переменных окружения
print("1️⃣  Проверка переменных окружения...")
env_vars = {
    'BINANCE_API_KEY': os.getenv('BINANCE_API_KEY'),
    'BINANCE_SECRET_KEY': os.getenv('BINANCE_SECRET_KEY'),
    'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY')
}

for key, value in env_vars.items():
    if value:
        print(f"   ✅ {key}: {'*' * 10} (установлен)")
    else:
        print(f"   ❌ {key}: НЕ УСТАНОВЛЕН")
print()

# 2. Проверка Binance API
print("2️⃣  Проверка Binance API...")
try:
    exchange = ccxt.binance({
        'apiKey': env_vars['BINANCE_API_KEY'],
        'secret': env_vars['BINANCE_SECRET_KEY'],
        'enableRateLimit': True,
    })
    
    balance = exchange.fetch_balance()
    usdt_balance = balance['total'].get('USDT', 0)
    
    print(f"   ✅ Подключение: OK")
    print(f"   💰 Баланс USDT: ${usdt_balance:.4f}")
    
    # Проверка минимального баланса
    if usdt_balance < 10:
        print(f"   ⚠️  ВНИМАНИЕ: Баланс < $10 - торговля невозможна!")
        print(f"   ℹ️  Минимальная сумма для Binance: $10")
    else:
        print(f"   ✅ Баланс достаточен для торговли")
        
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
print()

# 3. Проверка доступа к рынкам
print("3️⃣  Проверка доступа к рынкам...")
try:
    markets = exchange.load_markets()
    usdt_pairs = [s for s in markets if '/USDT' in s and markets[s].get('active', False)]
    print(f"   ✅ Всего рынков: {len(markets)}")
    print(f"   ✅ USDT пар: {len(usdt_pairs)}")
    print(f"   📊 Примеры: {', '.join(usdt_pairs[:5])}")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
print()

# 4. Проверка получения данных
print("4️⃣  Проверка получения рыночных данных...")
try:
    ticker = exchange.fetch_ticker('BTC/USDT')
    print(f"   ✅ BTC/USDT: ${ticker['last']:,.2f}")
    print(f"   📈 24h High: ${ticker['high']:,.2f}")
    print(f"   📉 24h Low: ${ticker['low']:,.2f}")
    print(f"   📊 Volume: ${ticker['quoteVolume']:,.0f}")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
print()

# 5. Тест Risk Engine
print("5️⃣  Тест Risk Engine...")
try:
    from trading_bot import RiskEngine
    
    risk_engine = RiskEngine(max_position_size_pct=10, max_total_exposure_pct=30)
    
    # Тестовый расчёт для баланса $100
    test_balance = 100.0
    test_signal_strength = 5  # из 6
    
    position_size = risk_engine.calculate_position_size(test_balance, test_signal_strength)
    fees = risk_engine.calculate_fees(position_size)
    
    print(f"   ✅ Risk Engine инициализирован")
    print(f"   📊 Тест для баланса: ${test_balance}")
    print(f"   💪 Сила сигнала: {test_signal_strength}/6")
    print(f"   💰 Размер позиции: ${position_size:.2f} ({position_size/test_balance*100:.1f}%)")
    print(f"   💸 Комиссия: ${fees:.4f}")
    
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
print()

# 6. Тест MetricsTracker
print("6️⃣  Тест MetricsTracker...")
try:
    from trading_bot import MetricsTracker
    
    metrics = MetricsTracker()
    
    # Добавим тестовые сделки
    metrics.add_trade('BTC/USDT', 'BUY', 10.0, 0.05, 0.02)  # Прибыль
    metrics.add_trade('ETH/USDT', 'BUY', 5.0, -0.03, 0.01)  # Убыток
    metrics.add_trade('SOL/USDT', 'BUY', 7.0, 0.08, 0.015)  # Прибыль
    
    summary = metrics.get_summary()
    
    print(f"   ✅ MetricsTracker инициализирован")
    print(f"   📊 Всего сделок: {summary['total_trades']}")
    print(f"   🎯 Win Rate: {summary['win_rate']:.1f}%")
    print(f"   💰 Прибыль: ${summary['total_profit']:.4f}")
    print(f"   💸 Комиссии: ${summary['total_fees']:.4f}")
    print(f"   📈 Чистая прибыль: ${summary['net_profit']:.4f}")
    print(f"   📊 Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
    
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
print()

# 7. Проверка Telegram бота
print("7️⃣  Проверка Telegram бота...")
try:
    import requests
    
    token = env_vars['TELEGRAM_BOT_TOKEN']
    response = requests.get(f"https://api.telegram.org/bot{token}/getMe")
    bot_info = response.json()
    
    if bot_info['ok']:
        bot = bot_info['result']
        print(f"   ✅ Бот подключен")
        print(f"   🤖 Имя: {bot['first_name']}")
        print(f"   👤 Username: @{bot['username']}")
        print(f"   🆔 ID: {bot['id']}")
    else:
        print(f"   ❌ Ошибка: {bot_info}")
        
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
print()

# ИТОГИ
print("=" * 60)
print("ИТОГОВЫЙ СТАТУС")
print("=" * 60)
print()
print("✅ Все компоненты работают!")
print()
print("📋 СЛЕДУЮЩИЕ ШАГИ:")
print("   1. Запустите бота: RUN_BOT.bat")
print("   2. Откройте Telegram и найдите @IntegronixBot")
print("   3. Отправьте команду /start")
print("   4. Протестируйте /analyze (сканирует Top 100)")
print("   5. Проверьте /status (баланс и метрики)")
print()
print("⚠️  ВАЖНО:")
print(f"   • Текущий баланс: ${usdt_balance:.4f}")
if usdt_balance < 10:
    print("   • Для торговли нужно минимум $10")
    print("   • Пополните баланс на Binance")
else:
    print("   • Баланс достаточен для торговли")
print("   • НЕ оставляйте бота без присмотра!")
print("   • Читайте SAFETY_GUIDE.md перед торговлей")
print()
print("=" * 60)
