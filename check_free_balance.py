import ccxt
import os
from dotenv import load_dotenv
load_dotenv()

exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET_KEY'),
})

balance = exchange.fetch_balance()

print('=== FREE (Available for trading) ===')
total_free = 0
for cur, amt in balance['free'].items():
    if amt > 0.001:
        if cur in ['USDT', 'USDC', 'BUSD', 'FDUSD']:
            print(f'  {cur}: {amt:.4f} USD')
            total_free += amt
        elif not cur.startswith('LD'):
            try:
                t = exchange.fetch_ticker(f'{cur}/USDT')
                usd = amt * t['last']
                if usd > 0.01:
                    print(f'  {cur}: {amt:.6f} = ${usd:.2f}')
                    total_free += usd
            except:
                pass

print(f'\nTOTAL FREE: ${total_free:.2f}')

print('\n=== LOCKED (In Earn/Staking) ===')
total_locked = 0
for cur, amt in balance['total'].items():
    if amt > 0.001 and cur.startswith('LD'):
        base = cur[2:]
        if base in ['USDT', 'USDC']:
            print(f'  {cur}: {amt:.4f} USD')
            total_locked += amt
        else:
            try:
                t = exchange.fetch_ticker(f'{base}/USDT')
                usd = amt * t['last']
                print(f'  {cur}: {amt:.6f} = ${usd:.2f}')
                total_locked += usd
            except:
                pass

print(f'\nTOTAL LOCKED: ${total_locked:.2f}')
