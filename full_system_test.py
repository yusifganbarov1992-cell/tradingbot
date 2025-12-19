"""
FULL SYSTEM CHECK - No stubs, real connections only
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("    NEXUSTRADER FULL SYSTEM CHECK")
print("=" * 70)
print()

results = []

# 1. CHECK BINANCE CONNECTION
print("1️⃣ BINANCE API CONNECTION")
try:
    from binance.client import Client
    client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_SECRET_KEY'))
    account = client.get_account()
    
    # Calculate total balance
    total_usd = 0
    for asset in account['balances']:
        free = float(asset['free'])
        locked = float(asset['locked'])
        total = free + locked
        if total > 0:
            symbol = asset['asset']
            if symbol == 'USDT':
                total_usd += total
            elif symbol.startswith('LD'):
                base = symbol[2:]
                if base == 'USDT':
                    total_usd += total
                else:
                    try:
                        ticker = client.get_symbol_ticker(symbol=f"{base}USDT")
                        total_usd += total * float(ticker['price'])
                    except:
                        pass
    
    print(f"   ✅ Connected to Binance")
    print(f"   💰 Total balance: ${total_usd:.2f}")
    print(f"   🔑 Can Trade: {account.get('canTrade', False)}")
    results.append(("Binance API", "✅ OK", f"${total_usd:.2f}"))
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    results.append(("Binance API", "❌ FAILED", str(e)))

print()

# 2. CHECK DATABASE
print("2️⃣ DATABASE CONNECTION")
try:
    from database_supabase import SupabaseDatabase
    db = SupabaseDatabase()
    trades = db.get_trade_history(limit=100)
    open_trades = [t for t in trades if t.get('status') == 'open']
    closed_trades = [t for t in trades if t.get('status') == 'closed']
    
    print(f"   ✅ Supabase connected")
    print(f"   📊 Total trades: {len(trades)}")
    print(f"   🔓 Open: {len(open_trades)}")
    print(f"   🔒 Closed: {len(closed_trades)}")
    results.append(("Database", "✅ OK", f"{len(trades)} trades"))
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    results.append(("Database", "❌ FAILED", str(e)))

print()

# 3. CHECK TELEGRAM
print("3️⃣ TELEGRAM BOT")
try:
    import requests
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    resp = requests.get(f"https://api.telegram.org/bot{token}/getMe")
    if resp.status_code == 200:
        bot_info = resp.json()['result']
        print(f"   ✅ Bot: @{bot_info['username']}")
        results.append(("Telegram", "✅ OK", f"@{bot_info['username']}"))
    else:
        print(f"   ❌ Invalid token")
        results.append(("Telegram", "❌ FAILED", "Invalid token"))
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    results.append(("Telegram", "❌ FAILED", str(e)))

print()

# 4. CHECK TRADING MODE
print("4️⃣ TRADING MODE")
paper_trading = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
auto_trade = os.getenv('AUTO_TRADE', 'false').lower() == 'true'
min_confidence = float(os.getenv('AUTO_MIN_CONFIDENCE', '7.0'))

if paper_trading:
    print(f"   📝 Mode: PAPER TRADING (simulation)")
    print(f"   ⚠️  No real orders will be placed!")
else:
    print(f"   🔥 Mode: REAL TRADING")
    print(f"   ⚠️  Real money will be used!")

print(f"   🤖 Auto Trade: {'ENABLED' if auto_trade else 'DISABLED'}")
print(f"   📈 Min Confidence: {min_confidence}/10")
results.append(("Trading Mode", "PAPER" if paper_trading else "REAL", f"Auto: {auto_trade}"))

print()

# 5. CHECK AI/ML MODULES
print("5️⃣ AI/ML MODULES")
modules_ok = 0
modules_total = 5

try:
    from modules.autonomous_trader import AutonomousTrader
    print("   ✅ AutonomousTrader")
    modules_ok += 1
except Exception as e:
    print(f"   ❌ AutonomousTrader: {e}")

try:
    from modules.performance_analyzer import PerformanceAnalyzer
    print("   ✅ PerformanceAnalyzer")
    modules_ok += 1
except Exception as e:
    print(f"   ❌ PerformanceAnalyzer: {e}")

try:
    from modules.risk_manager import AdvancedRiskManager
    print("   ✅ AdvancedRiskManager")
    modules_ok += 1
except Exception as e:
    print(f"   ❌ AdvancedRiskManager: {e}")

try:
    from modules.sentiment_analyzer import SentimentAnalyzer
    print("   ✅ SentimentAnalyzer")
    modules_ok += 1
except Exception as e:
    print(f"   ❌ SentimentAnalyzer: {e}")

try:
    from modules.market_regime import MarketRegimeManager
    print("   ✅ MarketRegimeManager")
    modules_ok += 1
except Exception as e:
    print(f"   ❌ MarketRegimeManager: {e}")

results.append(("AI Modules", f"{modules_ok}/{modules_total}", ""))

print()

# 6. CHECK BOT IS RUNNING
print("6️⃣ BOT PROCESS")
import subprocess
result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                       capture_output=True, text=True)
if 'python.exe' in result.stdout:
    lines = [l for l in result.stdout.split('\n') if 'python.exe' in l.lower()]
    print(f"   ✅ Python processes running: {len(lines)}")
    results.append(("Bot Process", "✅ Running", f"{len(lines)} processes"))
else:
    print("   ⚠️ No Python processes found")
    results.append(("Bot Process", "⚠️ Not running", ""))

print()

# SUMMARY
print("=" * 70)
print("    SUMMARY")
print("=" * 70)
print()

for name, status, detail in results:
    detail_str = f" ({detail})" if detail else ""
    print(f"   {name:20} {status:15}{detail_str}")

print()
print("=" * 70)

# Final verdict
all_ok = all("✅" in r[1] or "PAPER" in r[1] or "/" in r[1] for r in results)
if all_ok:
    print("   🎉 SYSTEM READY!")
    print()
    if paper_trading:
        print("   💡 To enable REAL trading:")
        print("      1. Edit .env: PAPER_TRADING=false")
        print("      2. Restart bot")
else:
    print("   ⚠️ ISSUES DETECTED - check above")

print("=" * 70)
