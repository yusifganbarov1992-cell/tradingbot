"""Check real Binance connection"""
import os
from dotenv import load_dotenv
load_dotenv()

print("=== BINANCE API CHECK ===")
print()

api_key = os.getenv('BINANCE_API_KEY', '')
secret = os.getenv('BINANCE_SECRET_KEY', '')

print(f"API Key length: {len(api_key)}")
print(f"Secret length: {len(secret)}")
print()

import ccxt
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': secret,
    'enableRateLimit': True
})

print("1. Public API (market data)...")
try:
    ticker = exchange.fetch_ticker('BTC/USDT')
    print(f"   BTC/USDT: ${ticker['last']:,.2f}")
    print("   ✅ Public API works")
except Exception as e:
    print(f"   ❌ Error: {e}")

print()
print("2. Private API (balance)...")
try:
    balance = exchange.fetch_balance()
    usdt = balance.get('USDT', {})
    print(f"   USDT free: {usdt.get('free', 0)}")
    print(f"   USDT used: {usdt.get('used', 0)}")
    print(f"   USDT total: {usdt.get('total', 0)}")
    
    # Show all non-zero balances
    print("   All non-zero balances:")
    for curr, data in balance['total'].items():
        if data and data > 0:
            print(f"      {curr}: {data}")
    print("   ✅ Private API works")
except Exception as e:
    print(f"   ❌ Error: {e}")

print()
print("3. API Permissions check...")
try:
    # Try to read trade history
    trades = exchange.fetch_my_trades('BTC/USDT', limit=5)
    print(f"   Trade history: {len(trades)} recent trades")
    print("   ✅ Read permission OK")
except ccxt.PermissionDenied as e:
    print(f"   ❌ Permission denied - enable Spot trading")
except Exception as e:
    print(f"   ⚠️ {type(e).__name__}: {str(e)[:100]}")

print()
print("4. Can place orders? (checking without executing)...")
try:
    # Just check if we have trading permission by looking at account info
    account = exchange.fetch_trading_fees()
    print("   ✅ Trading fees accessible")
except Exception as e:
    print(f"   ⚠️ {str(e)[:100]}")

print()
print("=== CHECK COMPLETE ===")
