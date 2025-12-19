"""
Тест AUTO_TRADE режима
Проверка всех 10 уровней защиты
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.autonomous_trader import AutonomousTrader
from datetime import datetime

def test_auto_trade():
    """Comprehensive AUTO_TRADE test"""
    
    print("="*60)
    print("🤖 AUTO_TRADE MODE - ТЕСТ ВСЕХ ФУНКЦИЙ")
    print("="*60)
    
    # Create AutonomousTrader
    print("\n1️⃣  Создание AutonomousTrader...")
    autonomous = AutonomousTrader(
        auto_trade_enabled=True,
        min_confidence=7.0,
        max_trades_per_hour=3,
        max_concurrent_positions=5,
        whitelist=['BTC/USDT', 'ETH/USDT'],
        blacklist=['LUNA', 'FTT']
    )
    print("✅ Created")
    
    # Test 1: Whitelist check
    print("\n2️⃣  Test WHITELIST:")
    signal_btc = {
        'symbol': 'BTC/USDT',
        'ai_confidence': 8.0,
        'usdt_amount': 50,
        'signal': 'BUY',
        'price': 50000,
        'crypto_amount': 0.001
    }
    
    should_execute, reason = autonomous.should_execute_auto(
        signal_data=signal_btc,
        active_positions={},
        balance=1000
    )
    print(f"   BTC/USDT (в whitelist): {should_execute} - {reason}")
    assert should_execute == True, "BTC должен пройти (в whitelist)"
    
    signal_sol = {
        'symbol': 'SOL/USDT',
        'ai_confidence': 9.0,  # High confidence
        'usdt_amount': 50,
        'signal': 'BUY',
        'price': 100,
        'crypto_amount': 0.5
    }
    
    should_execute, reason = autonomous.should_execute_auto(
        signal_data=signal_sol,
        active_positions={},
        balance=1000
    )
    print(f"   SOL/USDT (НЕ в whitelist): {should_execute} - {reason}")
    assert should_execute == False, "SOL должен быть отклонен (не в whitelist)"
    
    # Test 2: Blacklist check
    print("\n3️⃣  Test BLACKLIST:")
    signal_luna = {
        'symbol': 'LUNA',
        'ai_confidence': 10.0,  # Even max confidence
        'usdt_amount': 50,
        'signal': 'BUY',
        'price': 1.0,
        'crypto_amount': 50
    }
    
    # Temporarily remove whitelist for this test
    autonomous.whitelist = []
    
    should_execute, reason = autonomous.should_execute_auto(
        signal_data=signal_luna,
        active_positions={},
        balance=1000
    )
    print(f"   LUNA (в blacklist): {should_execute} - {reason}")
    assert should_execute == False, "LUNA должен быть отклонен (в blacklist)"
    
    # Restore whitelist
    autonomous.whitelist = ['BTC/USDT', 'ETH/USDT']
    
    # Test 3: Confidence check
    print("\n4️⃣  Test CONFIDENCE THRESHOLD:")
    signal_low_conf = {
        'symbol': 'BTC/USDT',
        'ai_confidence': 6.5,  # Below threshold
        'usdt_amount': 50,
        'signal': 'BUY',
        'price': 50000,
        'crypto_amount': 0.001
    }
    
    should_execute, reason = autonomous.should_execute_auto(
        signal_data=signal_low_conf,
        active_positions={},
        balance=1000
    )
    print(f"   Confidence 6.5/10 (< 7.0): {should_execute} - {reason}")
    assert should_execute == False, "Низкая confidence должна быть отклонена"
    
    # Test 4: Hourly limit
    print("\n5️⃣  Test HOURLY LIMIT:")
    
    # Simulate 3 trades already done this hour
    autonomous.trades_this_hour = [datetime.now()] * 3
    
    should_execute, reason = autonomous.should_execute_auto(
        signal_data=signal_btc,
        active_positions={},
        balance=1000
    )
    print(f"   3/3 trades done: {should_execute} - {reason}")
    assert should_execute == False, "Hourly limit должен блокировать"
    
    # Reset
    autonomous.trades_this_hour = []
    
    # Test 5: Max positions
    print("\n6️⃣  Test MAX POSITIONS:")
    
    active_positions = {
        'ADA/USDT': {},
        'DOT/USDT': {},
        'MATIC/USDT': {},
        'LINK/USDT': {},
        'XRP/USDT': {}
    }
    
    should_execute, reason = autonomous.should_execute_auto(
        signal_data=signal_btc,
        active_positions=active_positions,
        balance=1000
    )
    print(f"   5/5 positions open: {should_execute} - {reason}")
    assert should_execute == False, "Max positions должен блокировать"
    
    # Test 6: Duplicate position
    print("\n7️⃣  Test DUPLICATE POSITION:")
    
    active_positions = {'BTC/USDT': {}}
    
    should_execute, reason = autonomous.should_execute_auto(
        signal_data=signal_btc,
        active_positions=active_positions,
        balance=1000
    )
    print(f"   BTC/USDT уже открыт: {should_execute} - {reason}")
    assert should_execute == False, "Duplicate position должен блокировать"
    
    # Test 7: Balance check
    print("\n8️⃣  Test BALANCE:")
    
    should_execute, reason = autonomous.should_execute_auto(
        signal_data=signal_btc,
        active_positions={},
        balance=10  # Too low
    )
    print(f"   Balance $10 vs required $55: {should_execute} - {reason}")
    assert should_execute == False, "Недостаточный баланс должен блокировать"
    
    # Test 8: Emergency pause
    print("\n9️⃣  Test EMERGENCY PAUSE:")
    
    autonomous.emergency_stop("Test")
    
    should_execute, reason = autonomous.should_execute_auto(
        signal_data=signal_btc,
        active_positions={},
        balance=1000
    )
    print(f"   Emergency paused: {should_execute} - {reason}")
    assert should_execute == False, "Emergency pause должен блокировать"
    
    autonomous.resume_trading()
    
    # Test 9: Smart logic (high confidence)
    print("\n🔟 Test SMART LOGIC:")
    
    signal_high_conf = {
        'symbol': 'BTC/USDT',
        'ai_confidence': 9.5,  # Very high
        'usdt_amount': 50,
        'signal': 'BUY',
        'price': 50000,
        'crypto_amount': 0.001
    }
    
    should_execute, reason = autonomous.should_execute_auto(
        signal_data=signal_high_conf,
        active_positions={},
        balance=1000
    )
    print(f"   Confidence 9.5/10: {should_execute} - {reason}")
    assert should_execute == True, "High confidence должен пройти"
    
    # Test 10: Record trade
    print("\n1️⃣1️⃣  Test RECORD TRADE:")
    
    before_count = len(autonomous.trades_this_hour)
    autonomous.record_trade()
    after_count = len(autonomous.trades_this_hour)
    
    print(f"   Trades: {before_count} → {after_count}")
    assert after_count == before_count + 1, "Trade должен быть записан"
    
    # Test 11: Status
    print("\n1️⃣2️⃣  Test STATUS:")
    
    status = autonomous.get_status()
    print(f"   Enabled: {status['enabled']}")
    print(f"   Emergency paused: {status['emergency_paused']}")
    print(f"   Trades this hour: {status['trades_this_hour']}")
    print(f"   Min confidence: {status['min_confidence']}")
    
    # Test 12: Aggressive mode
    print("\n1️⃣3️⃣  Test AGGRESSIVE MODE:")
    
    signal_medium_conf = {
        'symbol': 'BTC/USDT',
        'ai_confidence': 8.2,
        'usdt_amount': 50,
        'signal': 'BUY',
        'price': 50000,
        'crypto_amount': 0.001
    }
    
    # Conservative mode (default)
    autonomous.set_aggressive(False)
    should_execute_cons, reason_cons = autonomous.should_execute_auto(
        signal_data=signal_medium_conf,
        active_positions={},
        balance=1000
    )
    print(f"   Conservative mode (8.2/10): {should_execute_cons} - {reason_cons}")
    
    # Aggressive mode
    autonomous.set_aggressive(True)
    should_execute_agg, reason_agg = autonomous.should_execute_auto(
        signal_data=signal_medium_conf,
        active_positions={},
        balance=1000
    )
    print(f"   Aggressive mode (8.2/10): {should_execute_agg} - {reason_agg}")
    assert should_execute_agg == True, "Aggressive mode должен принять 8+/10"
    
    # FINAL SUMMARY
    print("\n"+"="*60)
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
    print("="*60)
    print("\n📊 Проверено:")
    print("   1. ✅ Whitelist")
    print("   2. ✅ Blacklist")
    print("   3. ✅ Confidence threshold")
    print("   4. ✅ Hourly limit")
    print("   5. ✅ Max positions")
    print("   6. ✅ Duplicate position")
    print("   7. ✅ Balance check")
    print("   8. ✅ Emergency pause")
    print("   9. ✅ Smart logic")
    print("   10. ✅ Record trade")
    print("   11. ✅ Status")
    print("   12. ✅ Aggressive mode")
    print("\n🎉 AUTO_TRADE готов к работе!")
    
    return True

if __name__ == "__main__":
    try:
        test_auto_trade()
        print("\n✅ EXIT CODE: 0 (SUCCESS)")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("\n❌ EXIT CODE: 1 (FAILURE)")
        sys.exit(1)
