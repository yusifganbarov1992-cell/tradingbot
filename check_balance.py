import os
from dotenv import load_dotenv
import ccxt

load_dotenv()

exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET_KEY'),
})

print("Проверка баланса Binance...")
balance = exchange.fetch_balance()
usdt = balance['total'].get('USDT', 0)
print(f"\n💰 USDT Баланс: ${usdt:.2f}")

if usdt >= 70:
    print("✅ Баланс подтверждён! Готов к торговле!")
    print(f"📊 Можно открыть позиций на: ~${usdt * 0.3:.2f} (30% от баланса)")
    print(f"💰 Размер 1 позиции (10%): ~${usdt * 0.1:.2f}")
else:
    print(f"⚠️ Баланс ${usdt:.2f} меньше ожидаемого")
