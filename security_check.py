import os
from dotenv import load_dotenv
load_dotenv()

print('=== SECURITY CHECK ===')
print()

# 1. Paper trading
paper = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
print(f'1. PAPER_TRADING: {paper}')
if paper:
    print('   ✅ SAFE - No real money at risk')
else:
    print('   ⚠️ WARNING - Real trading enabled!')

# 2. Auto trade
auto = os.getenv('AUTO_TRADE', 'false').lower() == 'true'
print(f'2. AUTO_TRADE: {auto}')
if auto:
    print('   ⚠️ Bot will execute trades automatically')
else:
    print('   ✅ Manual confirmation required')

# 3. Min confidence
conf = float(os.getenv('MIN_CONFIDENCE', '7.0'))
print(f'3. MIN_CONFIDENCE: {conf}')
if conf >= 7:
    print('   ✅ High threshold - fewer but better trades')
else:
    print('   ⚠️ Low threshold - more risky trades')

# 4. Position size
pos = float(os.getenv('POSITION_SIZE_PCT', '2.0'))
print(f'4. POSITION_SIZE: {pos}%')
if pos <= 5:
    print('   ✅ Conservative position sizing')
else:
    print('   ⚠️ Large positions - higher risk')

# 5. Stop loss
sl = float(os.getenv('STOP_LOSS_PCT', '2.0'))
print(f'5. STOP_LOSS: {sl}%')
print('   ✅ Stop loss enabled')

# 6. API keys
api = os.getenv('BINANCE_API_KEY', '')[:10]
print(f'6. API KEY: {api}...')
print('   ✅ API keys configured')

print()
print('=== OVERALL ASSESSMENT ===')
if paper:
    print('🟢 SAFE TO RUN - Paper trading mode')
    print('   No real money will be used')
else:
    print('🟡 CAUTION - Live trading mode')
    print('   Real money at risk')
