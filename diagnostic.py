"""
Диагностика: почему бот не отправляет сигналы
"""
import os
from dotenv import load_dotenv
load_dotenv()  # ⬅️ КРИТИЧНО: загрузить .env ПЕРЕД импортом database!

import sqlite3
from database import TradingDatabase

print("="*60)
print("ДИАГНОСТИКА РАБОТЫ БОТА")
print("="*60)

# 1. Emergency Stop
print("\n1. ПРОВЕРКА EMERGENCY_STOP:")
db = TradingDatabase()
emergency_stop = db.load_emergency_stop()
print(f"   emergency_stop = {emergency_stop}")
if emergency_stop:
    print("   ❌ ПРОБЛЕМА: Emergency stop АКТИВЕН - блокирует ВСЕ сигналы!")
else:
    print("   ✅ OK: Emergency stop выключен")

# 2. Signals в БД
print("\n2. СИГНАЛЫ В БАЗЕ ДАННЫХ:")
conn = sqlite3.connect('trading_history.db')
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM signals")
total_signals = cursor.fetchone()[0]
print(f"   Всего сигналов: {total_signals}")

cursor.execute("SELECT COUNT(*) FROM signals WHERE status='pending'")
pending = cursor.fetchone()[0]
print(f"   Ожидающих: {pending}")

# Последние 5 сигналов
print("\n3. ПОСЛЕДНИЕ 5 СИГНАЛОВ:")
cursor.execute("""
    SELECT symbol, signal_type, ai_confidence, status, timestamp 
    FROM signals 
    ORDER BY id DESC 
    LIMIT 5
""")
signals = cursor.fetchall()

if signals:
    for i, s in enumerate(signals, 1):
        symbol, sig_type, confidence, status, timestamp = s
        print(f"   {i}. {symbol} {sig_type} (AI:{confidence}/10) [{status}] - {timestamp}")
else:
    print("   ❌ Сигналов НЕТ!")
    print("   Причина: Бот не находит возможностей для торговли")

# 4. Проверка .env
print("\n4. TELEGRAM НАСТРОЙКИ:")
operator_chat_id = os.getenv('OPERATOR_CHAT_ID')
telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
print(f"   OPERATOR_CHAT_ID: {operator_chat_id}")
print(f"   TELEGRAM_BOT_TOKEN: {telegram_token[:30] if telegram_token else 'MISSING'}...")
if not operator_chat_id:
    print("   ❌ OPERATOR_CHAT_ID не задан!")

# 5. Проверка trades
print("\n5. СДЕЛКИ:")
cursor.execute("SELECT COUNT(*) FROM trades")
total_trades = cursor.fetchone()[0]
print(f"   Всего сделок: {total_trades}")

cursor.execute("SELECT COUNT(*) FROM trades WHERE status='open'")
open_trades = cursor.fetchone()[0]
print(f"   Открытых сделок: {open_trades}")

conn.close()

# ИТОГ
print("\n" + "="*60)
print("ВЫВОДЫ:")
print("="*60)

if emergency_stop:
    print("❌ БОТ НЕ ТОРГУЕТ: Emergency stop активен")
    print("   Решение: Отправь /resume в Telegram @IntegronixBot")
elif total_signals == 0:
    print("❌ БОТ НЕ НАХОДИТ СИГНАЛОВ")
    print("   Возможные причины:")
    print("   1. Рынок не даёт возможностей (низкая волатильность)")
    print("   2. AI не даёт confidence ≥7")
    print("   3. Фильтры не проходят (нужно 7/9)")
    print("   Решение: Понизить порог AI confidence до 6")
elif pending > 0:
    print(f"✅ ЕСТЬ {pending} ОЖИДАЮЩИХ СИГНАЛОВ")
    print("   Они должны быть в Telegram!")
    print("   Если нет - проблема с отправкой сообщений")
else:
    print("✅ Всё в порядке, но сигналов просто нет сейчас")
    print("   Бот ищет возможности каждые 5 минут")

print("="*60)
