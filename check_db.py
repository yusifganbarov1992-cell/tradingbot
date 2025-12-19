"""Check database trades"""
from database_supabase import SupabaseDatabase

db = SupabaseDatabase()
trades = db.get_trade_history(limit=20)

print(f"Trades in database: {len(trades)}")
print()

for t in trades[:10]:
    symbol = t.get('symbol', 'N/A')
    status = t.get('status', 'N/A')
    pnl = t.get('pnl') or 0
    mode = t.get('mode', 'N/A')
    entry_time = t.get('entry_time', 'N/A')
    print(f"{symbol:12} | {status:8} | PnL: ${pnl:>8.2f} | Mode: {mode} | {entry_time}")
