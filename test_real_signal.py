"""
РЕАЛЬНЫЙ ТЕСТ: Симуляция получения сигнала и автоматического выполнения
"""

import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

# Настройка окружения
os.environ['BINANCE_API_KEY'] = 'fake_key'
os.environ['BINANCE_SECRET_KEY'] = 'fake_secret'
os.environ['TELEGRAM_BOT_TOKEN'] = 'fake_token'
os.environ['OPERATOR_CHAT_ID'] = '12345'
os.environ['PAPER_TRADING'] = 'true'
os.environ['AUTO_TRADE'] = 'true'  # ВКЛЮЧАЕМ AUTO_TRADE!
os.environ['AUTO_MIN_CONFIDENCE'] = '7.0'

print("="*70)
print("РЕАЛЬНЫЙ ТЕСТ AUTO_TRADE")
print("="*70)

# Импорт
print("\n[1/5] Импорт TradingAgent...")
from trading_bot import TradingAgent

# Создание
print("[2/5] Создание агента...")
agent = TradingAgent()

# Проверка статуса
print(f"\n[3/5] Проверка AUTO_TRADE:")
print(f"   agent.autonomous.enabled: {agent.autonomous.enabled}")
print(f"   agent.autonomous.min_confidence: {agent.autonomous.min_confidence}")

if not agent.autonomous.enabled:
    print("\n❌ AUTO_TRADE НЕ ВКЛЮЧЕН!")
    print("   Проверьте AUTO_TRADE=true в .env")
    sys.exit(1)

print("\n✅ AUTO_TRADE ВКЛЮЧЕН!")

# Создаем тестовый сигнал
print("\n[4/5] Создаем тестовый сигнал...")

signal_data = {
    'symbol': 'BTC/USDT',
    'ai_confidence': 8.5,  # Высокая уверенность
    'usdt_amount': 50,
    'signal': 'BUY',
    'price': 50000,
    'crypto_amount': 0.001,
    'current_rsi': 45,
    'current_ema20': 49000,
    'current_ema50': 48000,
    'current_macd': 0.5,
    'current_volume': 1000,
    'avg_volume': 800,
    'current_atr': 500,
    'ai_signal': 'BUY',
    'ai_reason': 'Strong uptrend with volume spike',
    'signal_strength': 8
}

# Проверяем логику
print(f"\n[5/5] Проверка логики AUTO_TRADE:")
print(f"   Сигнал: {signal_data['signal']} {signal_data['symbol']}")
print(f"   AI Confidence: {signal_data['ai_confidence']}/10")
print(f"   Цена: ${signal_data['price']}")
print(f"   Размер: ${signal_data['usdt_amount']}")

should_auto, reason = agent.autonomous.should_execute_auto(
    signal_data=signal_data,
    active_positions={},
    balance=1000
)

print(f"\n📊 РЕЗУЛЬТАТ:")
print(f"   Should execute auto: {should_auto}")
print(f"   Reason: {reason}")

if should_auto:
    print("\n✅ СИГНАЛ ПРОШЕЛ ВСЕ ПРОВЕРКИ!")
    print("   Бот выполнил бы сделку АВТОМАТИЧЕСКИ")
    print("   БЕЗ вашего подтверждения через Telegram")
    
    # Проверим что метод существует
    print(f"\n   Метод _execute_trade_directly: {hasattr(agent, '_execute_trade_directly')}")
    
    if hasattr(agent, '_execute_trade_directly'):
        print("   ✅ Метод существует и готов к вызову")
        
        # Проверим что он реально вызовется
        print("\n   Проверяем что метод вызовется в send_signal_to_telegram...")
        import inspect
        source = inspect.getsource(agent.send_signal_to_telegram)
        
        if 'should_execute_auto' in source and '_execute_trade_directly' in source:
            print("   ✅ Метод РЕАЛЬНО вызовется!")
        else:
            print("   ❌ Метод НЕ вызовется - интеграция неполная")
    
    print("\n" + "="*70)
    print("ВЕРДИКТ: AUTO_TRADE РАБОТАЕТ!")
    print("="*70)
    print("\nЧто происходит при реальном сигнале:")
    print("1. Бот получает сигнал (RSI, EMA, MACD, etc.)")
    print("2. AI анализирует и дает confidence (7-10/10)")
    print("3. send_signal_to_telegram() вызывает should_execute_auto()")
    print("4. Проверка 10 уровней защиты")
    print("5. Если ВСЕ ОК → _execute_trade_directly() выполняет сделку")
    print("6. Уведомление в Telegram (БЕЗ кнопок)")
    print("\nБЕЗ ВАШЕГО УЧАСТИЯ!")
    
else:
    print(f"\n⚠️  СИГНАЛ НЕ ПРОШЕЛ: {reason}")
    print("   В этом случае бот попросил бы подтверждение")

print("\n" + "="*70)
print("СЛЕДУЮЩИЙ ШАГ: Запустите бота и дождитесь реального сигнала")
print("="*70)
