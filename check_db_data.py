from database_supabase import SupabaseDatabase
db = SupabaseDatabase()

# Check trades
trades = db.get_trade_history(limit=20)
print('=== TRADES IN DB ===')
for t in trades:
    print(f"  {t['trade_id']}: {t['side']} {t['symbol']} @ {t['entry_price']} - {t['status']}")
if not trades:
    print('  No trades')

# Check signals
result = db.client.table('signals').select('*').order('created_at', desc=True).limit(10).execute()
signals = result.data or []
print(f'\n=== SIGNALS IN DB ({len(signals)}) ===')
for s in signals:
    reason = s.get('ai_reason', 'no reason') or 'no reason'
    print(f"  {s.get('symbol')}: {s.get('signal_type')} conf={s.get('ai_confidence')} | {reason[:50]}...")
