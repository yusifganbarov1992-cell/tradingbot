"""
Check if dashboard shows REAL data or MOCK data
"""
import sys

# Reset singleton
if 'dashboard.data_provider' in sys.modules:
    del sys.modules['dashboard.data_provider']

from dashboard.data_provider import DashboardDataProvider

# Force new instance
DashboardDataProvider._instance = None

provider = DashboardDataProvider()
print('=' * 60)
print('DASHBOARD DATA CHECK - Real or Mock?')
print('=' * 60)

# Portfolio
print('\n1. PORTFOLIO SUMMARY:')
portfolio = provider.get_portfolio_summary()
for k, v in portfolio.items():
    if isinstance(v, float):
        print(f'   {k}: {v:.2f}')
    else:
        print(f'   {k}: {v}')

# Check if it's mock data (mock has balance 10150.00)
if portfolio.get('balance') == 10150.00:
    print('\n   ⚠️ WARNING: This looks like MOCK data!')
else:
    print('\n   ✅ Data appears to be REAL')

# Trades
print('\n2. TRADE HISTORY:')
trades = provider.get_trade_history()
print(f'   Total rows: {len(trades)}')
if len(trades) > 0:
    print(f'   Columns: {list(trades.columns)}')
    print(trades.head(3).to_string())

# Recent activity  
print('\n3. RECENT ACTIVITY:')
activity = provider.get_recent_activity()
for a in activity[:3]:
    print(f'   {a.get("timestamp")} | {a.get("symbol")} | {a.get("action")} | PnL: {a.get("pnl", 0)}')

# Current price (LIVE from exchange)
print('\n4. LIVE PRICES FROM BINANCE:')
btc = provider.get_current_price('BTC/USDT')
eth = provider.get_current_price('ETH/USDT')
print(f'   BTC/USDT: ${btc:,.2f}')
print(f'   ETH/USDT: ${eth:,.2f}')

# Check if prices are realistic
if 50000 < btc < 200000:
    print('   ✅ BTC price is realistic')
else:
    print('   ⚠️ BTC price looks fake')

print('\n' + '=' * 60)

# Direct database check
print('\nDIRECT DATABASE CHECK:')
from database_supabase import SupabaseDatabase
db = SupabaseDatabase()
db_trades = db.get_trade_history(limit=10)
print(f'   Supabase trades: {len(db_trades)}')
for t in db_trades[:3]:
    print(f'   - {t.get("symbol")} | {t.get("status")} | Mode: {t.get("mode")}')

print('\n' + '=' * 60)
print('VERDICT:')
if len(db_trades) > 0 and portfolio.get('balance') != 10150.00:
    print('   ✅ Dashboard uses REAL DATA from Supabase')
else:
    print('   ⚠️ Dashboard may be using MOCK data')
print('=' * 60)
