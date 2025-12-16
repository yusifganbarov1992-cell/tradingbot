"""
Быстрый тест сканирования рынка
"""
import sys
sys.path.insert(0, '.')
from trading_bot import TradingAgent
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

print('='*60)
print('ТЕСТ СКАНИРОВАНИЯ РЫНКА')
print('='*60)

print('\nИнициализация бота...')
agent = TradingAgent()

print('\n' + '='*60)
print('ТЕСТ 1: Сканирование ETH/USDT')
print('='*60)

signal = agent.analyze_market_symbol('ETH/USDT')

if signal:
    print(f'\n✅ СИГНАЛ НАЙДЕН!')
    print(f'Symbol: {signal.get("symbol")}')
    print(f'Signal: {signal.get("signal")}')
    print(f'AI Confidence: {signal.get("ai_confidence")}/10')
    print(f'AI Reason: {signal.get("ai_reason")}')
    print(f'Filters: BUY={signal.get("buy_filters")}, SELL={signal.get("sell_filters")}')
    print(f'Price: ${signal.get("price"):.2f}')
else:
    print('\n❌ Сигнал НЕ найден')
    print('\nВозможные причины:')
    print('1. AI confidence < 7')
    print('2. Фильтры не сработали (нужно 7+ из 9)')
    print('3. Safety блокировка (emergency_stop, daily limit, etc.)')
    print('4. AI вернул WAIT')

print('\n' + '='*60)
print('ТЕСТ 2: Полное сканирование (как в боте)')
print('='*60)

print('\nЗапуск analyze_all_markets()...')
agent.analyze_all_markets()

print('\n' + '='*60)
print('ТЕСТ ЗАВЕРШЁН')
print('='*60)
