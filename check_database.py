"""Проверка содержимого базы данных"""
import sqlite3

conn = sqlite3.connect('trading_history.db')
cursor = conn.cursor()

# Получить список таблиц
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

print("📊 Таблицы в базе данных:")
for table in tables:
    print(f"  - {table[0]}")

print("\n📈 Статистика:")

# Подсчет сигналов
cursor.execute("SELECT COUNT(*) FROM signals")
total_signals = cursor.fetchone()[0]
print(f"  Всего сигналов: {total_signals}")

cursor.execute("SELECT COUNT(*) FROM signals WHERE status='approved'")
approved_signals = cursor.fetchone()[0]
print(f"  Одобренных сигналов: {approved_signals}")

cursor.execute("SELECT COUNT(*) FROM signals WHERE status='rejected'")
rejected_signals = cursor.fetchone()[0]
print(f"  Отклоненных сигналов: {rejected_signals}")

# Подсчет сделок
cursor.execute("SELECT COUNT(*) FROM trades")
total_trades = cursor.fetchone()[0]
print(f"\n  Всего сделок: {total_trades}")

cursor.execute("SELECT COUNT(*) FROM trades WHERE status='open'")
open_trades = cursor.fetchone()[0]
print(f"  Открытых позиций: {open_trades}")

cursor.execute("SELECT COUNT(*) FROM trades WHERE status='closed'")
closed_trades = cursor.fetchone()[0]
print(f"  Закрытых позиций: {closed_trades}")

# Последние сигналы
print("\n🔔 Последние 5 сигналов:")
cursor.execute("""
    SELECT symbol, signal_type, price, ai_confidence, status, timestamp 
    FROM signals 
    ORDER BY timestamp DESC 
    LIMIT 5
""")
for row in cursor.fetchall():
    symbol, signal_type, price, confidence, status, timestamp = row
    print(f"  {timestamp} | {symbol} {signal_type} @ ${price:.4f} | AI: {confidence}/10 | {status}")

# Последние сделки
if total_trades > 0:
    print("\n💰 Последние 5 сделок:")
    cursor.execute("""
        SELECT symbol, side, entry_price, exit_price, pnl, pnl_percent, status, entry_time 
        FROM trades 
        ORDER BY entry_time DESC 
        LIMIT 5
    """)
    for row in cursor.fetchall():
        symbol, side, entry, exit_p, pnl, pnl_pct, status, entry_time = row
        if status == 'open':
            print(f"  {entry_time} | {side} {symbol} @ ${entry:.4f} | ОТКРЫТА")
        else:
            pnl_sign = "+" if pnl > 0 else ""
            print(f"  {entry_time} | {side} {symbol} | Вход: ${entry:.4f}, Выход: ${exit_p:.4f} | P&L: {pnl_sign}${pnl:.2f} ({pnl_pct:.2f}%)")

conn.close()
print("\n✅ Все данные сохранены в файле: trading_history.db")
print("📁 Размер файла:", end=" ")
import os
size = os.path.getsize('trading_history.db')
print(f"{size:,} байт ({size/1024:.2f} KB)")
