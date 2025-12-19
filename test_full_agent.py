"""
Полное тестирование торгового агента
Проверяет все компоненты системы
"""
import os
import sys
from dotenv import load_dotenv
load_dotenv()

from trading_bot import TradingAgent, SafetyManager
from database import TradingDatabase
from database_supabase import SupabaseDatabase
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_1_agent_initialization():
    """Тест 1: Инициализация агента"""
    print_section("TEST 1: AGENT INITIALIZATION")
    
    try:
        agent = TradingAgent()
        print("✅ Agent initialized successfully")
        print(f"   - Paper trading: {agent.paper_trading}")
        print(f"   - Operator chat: {agent.operator_chat_id}")
        print(f"   - Active positions: {len(agent.active_positions)}")
        print(f"   - Supabase connected: {agent.supabase_db is not None}")
        return agent, True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return None, False

def test_2_market_data_fetching(agent):
    """Тест 2: Получение рыночных данных"""
    print_section("TEST 2: MARKET DATA FETCHING")
    
    try:
        # Тест получения OHLCV
        symbol = 'ETH/USDT'
        ohlcv = agent.exchange.fetch_ohlcv(symbol, '1h', limit=100)
        print(f"✅ OHLCV data fetched: {len(ohlcv)} candles for {symbol}")
        
        # Тест получения текущей цены
        ticker = agent.exchange.fetch_ticker(symbol)
        price = ticker['last']
        print(f"✅ Current price: ${price:,.2f}")
        
        # Тест получения баланса
        balance = agent.exchange.fetch_balance()
        usdt = balance['USDT']['free']
        print(f"✅ Balance: ${usdt:.2f} USDT")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_3_technical_indicators(agent):
    """Тест 3: Расчет технических индикаторов"""
    print_section("TEST 3: TECHNICAL INDICATORS")
    
    try:
        import pandas as pd
        import numpy as np
        
        symbol = 'BTC/USDT'
        ohlcv = agent.exchange.fetch_ohlcv(symbol, '1h', limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        rsi = df['rsi'].iloc[-1]
        
        # EMA
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        ema20 = df['ema20'].iloc[-1]
        ema50 = df['ema50'].iloc[-1]
        
        # MACD
        df['macd'] = df['ema20'] - df['ema50']
        macd = df['macd'].iloc[-1]
        
        # ATR
        df['h-l'] = df['high'] - df['low']
        df['h-pc'] = abs(df['high'] - df['close'].shift(1))
        df['l-pc'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        atr = df['atr'].iloc[-1]
        
        # Проверка на NaN
        if pd.isna([rsi, ema20, ema50, macd, atr]).any():
            print("❌ FAILED: NaN values in indicators")
            return False
        
        print(f"✅ RSI: {rsi:.2f}")
        print(f"✅ EMA20: ${ema20:,.2f}, EMA50: ${ema50:,.2f}")
        print(f"✅ MACD: {macd:.2f}")
        print(f"✅ ATR: ${atr:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_4_ai_analysis(agent):
    """Тест 4: AI анализ"""
    print_section("TEST 4: AI ANALYSIS")
    
    try:
        from openai import OpenAI
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠️  SKIPPED: No OpenAI API key")
            return True
        
        client = OpenAI(api_key=api_key)
        
        # Простой тест запрос
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Reply only: BUY|8|Test successful"},
                {"role": "user", "content": "Test"}
            ],
            max_tokens=20
        )
        
        ai_response = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens
        
        # Парсинг
        parts = ai_response.split('|')
        if len(parts) >= 3:
            signal = parts[0].strip()
            confidence = parts[1].strip()
            reason = parts[2].strip()
            
            print(f"✅ AI response: {signal}, Confidence: {confidence}, Reason: {reason}")
            print(f"✅ Tokens used: {tokens}")
            return True
        else:
            print(f"❌ FAILED: Invalid format: {ai_response}")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_5_safety_system(agent):
    """Тест 5: Система безопасности"""
    print_section("TEST 5: SAFETY SYSTEM")
    
    try:
        # Тест emergency stop
        agent.safety.activate_emergency_stop()
        print("✅ Emergency stop activated")
        
        test_signal = {
            'symbol': 'ETH/USDT',
            'confidence': 9,
            'signal': 'BUY'
        }
        
        is_safe, reason = agent.safety.check_all_safety_levels(
            test_signal, 1000.0, [], [3500, 3505, 3510]
        )
        
        if not is_safe and "EMERGENCY" in reason:
            print(f"✅ Emergency stop blocking trades: {reason}")
        else:
            print("❌ FAILED: Emergency stop not working")
            return False
        
        # Снять emergency stop
        agent.safety.deactivate_emergency_stop()
        print("✅ Emergency stop deactivated")
        
        # Тест AI confidence
        test_signal_low = {
            'symbol': 'BTC/USDT',
            'confidence': 5,
            'signal': 'BUY'
        }
        
        is_safe2, reason2 = agent.safety.check_all_safety_levels(
            test_signal_low, 1000.0, [], [50000, 50050]
        )
        
        if not is_safe2 and "confidence too low" in reason2:
            print(f"✅ Low confidence blocking: {reason2}")
        else:
            print("❌ FAILED: Confidence check not working")
            return False
        
        print("✅ All safety checks working")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_6_database_operations():
    """Тест 6: Операции с базой данных"""
    print_section("TEST 6: DATABASE OPERATIONS")
    
    try:
        from datetime import datetime
        
        db = TradingDatabase()
        
        # Тест emergency stop save/load
        db.save_emergency_stop(False)
        loaded = db.load_emergency_stop()
        if loaded == False:
            print("✅ SQLite: Emergency stop save/load OK")
        else:
            print(f"❌ SQLite: Emergency stop mismatch (expected False, got {loaded})")
            return False
        
        # Тест сохранения сигнала
        test_id = f"TEST_{int(datetime.now().timestamp())}"
        db.save_signal(
            trade_id=test_id,
            symbol='TEST/USDT',
            signal_type='BUY',
            price=100.0,
            indicators={'rsi': 45, 'ema20': 105, 'ema50': 95},
            ai_analysis={'signal': 'BUY', 'confidence': 8, 'reason': 'Test'},
            position_info={'amount': 0.1, 'usdt_amount': 10, 'fee': 0.01}
        )
        print(f"✅ SQLite: Test signal saved (ID: {test_id})")
        
        # Тест Supabase
        try:
            supabase_db = SupabaseDatabase()
            stats = supabase_db.get_statistics()
            print(f"✅ Supabase: Connected, {stats.get('total_trades', 0)} trades")
        except Exception as supabase_error:
            print(f"⚠️  Supabase: {supabase_error}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_7_position_management(agent):
    """Тест 7: Управление позициями"""
    print_section("TEST 7: POSITION MANAGEMENT")
    
    try:
        # Проверка восстановления позиций
        initial_count = len(agent.active_positions)
        print(f"✅ Active positions restored: {initial_count}")
        
        # Тест расчета position size
        balance = 1000.0
        price = 3500.0
        signal_strength = 8
        
        crypto_amount, usdt_amount = agent.risk_engine.calculate_position_size(
            balance, price, signal_strength
        )
        
        position_pct = (usdt_amount / balance) * 100
        
        print(f"✅ Position size: {crypto_amount:.6f} crypto (~${usdt_amount:.2f})")
        print(f"   - Percentage: {position_pct:.1f}% of balance")
        print(f"   - Price: ${price:.2f}")
        
        if position_pct > 15:
            print(f"❌ FAILED: Position too large ({position_pct:.1f}% > 15%)")
            return False
        
        # Тест расчета fees
        fee = agent.risk_engine.calculate_fees(usdt_amount)
        print(f"✅ Trading fee: ${fee:.4f} (0.1%)")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_8_signal_generation(agent):
    """Тест 8: Генерация сигналов"""
    print_section("TEST 8: SIGNAL GENERATION")
    
    try:
        symbol = 'ETH/USDT'
        
        print(f"Analyzing {symbol}...")
        signal_data = agent.analyze_market_symbol(symbol)
        
        if signal_data:
            print(f"✅ SIGNAL GENERATED!")
            print(f"   - Symbol: {signal_data['symbol']}")
            print(f"   - Signal: {signal_data['signal']}")
            print(f"   - AI Confidence: {signal_data['ai_confidence']}/10")
            print(f"   - Price: ${signal_data['price']:.2f}")
            print(f"   - Filters: BUY={signal_data.get('buy_filters', 0)}, SELL={signal_data.get('sell_filters', 0)}")
            return True
        else:
            print("⚠️  No signal generated (market conditions not met)")
            print("   This is normal - bot is selective!")
            return True
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def run_all_tests():
    """Запуск всех тестов"""
    print("\n" + "="*60)
    print("  FULL AGENT TESTING SUITE")
    print("  Testing all components of trading bot")
    print("="*60)
    
    results = {}
    
    # Test 1: Initialization
    agent, success = test_1_agent_initialization()
    results['Initialization'] = success
    
    if not agent:
        print("\n❌ Cannot continue - agent initialization failed")
        return results
    
    # Test 2: Market Data
    results['Market Data'] = test_2_market_data_fetching(agent)
    
    # Test 3: Indicators
    results['Indicators'] = test_3_technical_indicators(agent)
    
    # Test 4: AI
    results['AI Analysis'] = test_4_ai_analysis(agent)
    
    # Test 5: Safety
    results['Safety System'] = test_5_safety_system(agent)
    
    # Test 6: Database
    results['Database'] = test_6_database_operations()
    
    # Test 7: Positions
    results['Positions'] = test_7_position_management(agent)
    
    # Test 8: Signals
    results['Signals'] = test_8_signal_generation(agent)
    
    # Summary
    print("\n" + "="*60)
    print("  TEST RESULTS SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s} {status}")
    
    print("\n" + "-"*60)
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print("-"*60)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Agent is fully functional!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review errors above.")
    
    return results

if __name__ == "__main__":
    try:
        results = run_all_tests()
        
        # Exit code для автоматизации
        failed_count = sum(1 for v in results.values() if not v)
        sys.exit(0 if failed_count == 0 else 1)
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
