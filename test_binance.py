import ccxt
import os
from dotenv import load_dotenv

load_dotenv()

print("Testing Binance API connection...")
ex = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET_KEY')
})

try:
    ticker = ex.fetch_ticker('BTC/USDT')
    print(f"✅ SUCCESS! BTC price: ${ticker['last']}")
    
    # Test balance
    balance = ex.fetch_balance()
    print(f"✅ Account access OK")
    print(f"USDT Balance: {balance.get('USDT', {}).get('free', 0)}")
except Exception as e:
    print(f"❌ ERROR: {e}")
