"""
Тест живого сканирования рынка
Имитирует реальную работу бота
"""
import os
import sys
from dotenv import load_dotenv
load_dotenv()

from trading_bot import TradingAgent
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def test_live_scanning():
    """Тест живого сканирования"""
    print("="*60)
    print("  LIVE MARKET SCANNING TEST")
    print("  Simulating real bot behavior")
    print("="*60)
    
    # Инициализация
    print("\n1. Initializing agent...")
    agent = TradingAgent()
    print(f"   ✅ Agent ready")
    print(f"   - Balance: ${agent.exchange.fetch_balance()['USDT']['free']:.2f}")
    print(f"   - Emergency stop: {agent.safety.emergency_stop}")
    print(f"   - Active positions: {len(agent.active_positions)}")
    
    # Сканирование топ монет
    print("\n2. Scanning top movers...")
    agent.scan_top_movers(top_n=50, min_volume_usdt=1000000, min_price_change_pct=2.0)
    print(f"   ✅ Found {len(agent.symbols)} symbols to analyze")
    
    # Анализ каждого символа
    print("\n3. Analyzing symbols (like real bot)...")
    
    priority_symbols = ['ETH/USDT', 'BTC/USDT']
    combined_symbols = priority_symbols + [s for s in agent.symbols if s not in priority_symbols]
    combined_symbols = combined_symbols[:10]  # Ограничим для теста
    
    all_signals = []
    
    for i, symbol in enumerate(combined_symbols, 1):
        print(f"\n   [{i}/{len(combined_symbols)}] Analyzing {symbol}...")
        
        try:
            signal_data = agent.analyze_market_symbol(symbol)
            
            if signal_data:
                print(f"      ✅ SIGNAL: {signal_data['signal']} (AI: {signal_data['ai_confidence']}/10)")
                print(f"         Reason: {signal_data.get('ai_reason', 'N/A')[:60]}...")
                all_signals.append(signal_data)
            else:
                print(f"      ⊘ No signal (conditions not met)")
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            continue
    
    # Сортировка и отображение TOP-3
    print("\n" + "="*60)
    print("  RESULTS")
    print("="*60)
    
    if all_signals:
        all_signals.sort(key=lambda x: x['ai_confidence'], reverse=True)
        top_signals = all_signals[:3]
        
        print(f"\n✅ Found {len(all_signals)} signals, TOP-3:")
        
        for i, signal in enumerate(top_signals, 1):
            print(f"\n{i}. {signal['symbol']} - {signal['signal']}")
            print(f"   AI Confidence: {signal['ai_confidence']}/10")
            print(f"   Price: ${signal['price']:.2f}")
            print(f"   Amount: {signal['crypto_amount']:.6f} (~${signal['usdt_amount']:.2f})")
            print(f"   Reason: {signal.get('ai_reason', 'N/A')}")
        
        print("\n💡 In real bot, these would be sent to Telegram!")
        
    else:
        print("\n⚠️  No signals found in this scan cycle")
        print("   This is normal - bot is selective and waits for quality setups")
    
    # Статистика
    print("\n" + "="*60)
    print("  SCAN STATISTICS")
    print("="*60)
    print(f"Symbols scanned: {len(combined_symbols)}")
    print(f"Signals found: {len(all_signals)}")
    if len(combined_symbols) > 0:
        print(f"Hit rate: {(len(all_signals)/len(combined_symbols))*100:.1f}%")
    
    return len(all_signals) > 0

if __name__ == "__main__":
    try:
        success = test_live_scanning()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nScan interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
