"""Check signals table in Supabase"""
from database_supabase import SupabaseDatabase
db = SupabaseDatabase()

# Check signals table
try:
    result = db.client.table('signals').select('*').limit(10).execute()
    print(f'Signals in DB: {len(result.data)}')
    if result.data:
        for s in result.data:
            symbol = s.get('symbol', 'N/A')
            signal_type = s.get('signal_type', 'N/A')
            confidence = s.get('ai_confidence', 0)
            reason = s.get('ai_reason', 'N/A')
            if reason and len(reason) > 60:
                reason = reason[:60] + '...'
            print(f'  - {symbol} | {signal_type} | Conf: {confidence} | Reason: {reason}')
    else:
        print('No signals found')
except Exception as e:
    print(f'Error: {e}')
    print('Signals table may not exist. Creating...')
