"""
Check REAL Binance balance - no stubs, no fakes
"""
from binance.client import Client
from dotenv import load_dotenv
import os

load_dotenv()

client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_SECRET_KEY'))
account = client.get_account()

print('=' * 60)
print('REAL BINANCE BALANCE')
print('=' * 60)

total_usd = 0
assets_with_value = []

for asset in account['balances']:
    free = float(asset['free'])
    locked = float(asset['locked'])
    total = free + locked
    if total > 0:
        symbol = asset['asset']
        usd_value = 0
        
        # Calculate USD value
        if symbol == 'USDT':
            usd_value = total
        elif symbol == 'USDC':
            usd_value = total
        elif symbol.startswith('LD'):
            # Launchpool tokens - get underlying value
            base = symbol[2:]  # Remove 'LD' prefix
            if base == 'USDT':
                usd_value = total
            else:
                try:
                    ticker = client.get_symbol_ticker(symbol=f"{base}USDT")
                    price = float(ticker['price'])
                    usd_value = total * price
                except:
                    pass
        else:
            try:
                ticker = client.get_symbol_ticker(symbol=f"{symbol}USDT")
                price = float(ticker['price'])
                usd_value = total * price
            except:
                try:
                    # Try BTC pair
                    ticker = client.get_symbol_ticker(symbol=f"{symbol}BTC")
                    btc_price = float(ticker['price'])
                    btc_usdt = float(client.get_symbol_ticker(symbol="BTCUSDT")['price'])
                    usd_value = total * btc_price * btc_usdt
                except:
                    pass
        
        total_usd += usd_value
        assets_with_value.append({
            'symbol': symbol,
            'free': free,
            'locked': locked,
            'total': total,
            'usd': usd_value
        })

# Sort by USD value
assets_with_value.sort(key=lambda x: x['usd'], reverse=True)

for a in assets_with_value:
    if a['usd'] > 0.01:
        print(f"{a['symbol']:10} Total: {a['total']:>15.4f}  = ${a['usd']:>10.2f}")
    else:
        print(f"{a['symbol']:10} Total: {a['total']:>15.4f}  (dust)")

print('=' * 60)
print(f'TOTAL PORTFOLIO VALUE: ${total_usd:.2f} USD')
print('=' * 60)

# Check trading permissions
print()
print('ACCOUNT PERMISSIONS:')
print(f"  Can Trade: {account.get('canTrade', False)}")
print(f"  Can Withdraw: {account.get('canWithdraw', False)}")
print(f"  Can Deposit: {account.get('canDeposit', False)}")
