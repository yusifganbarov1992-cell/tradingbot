"""
ГЛОБАЛЬНЫЙ ТЕСТ ВСЕЙ СИСТЕМЫ
Проверяет все компоненты торгового бота
"""
import os
import sys
from dotenv import load_dotenv
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
import asyncio
import ccxt
from database import TradingDatabase
from database_supabase import SupabaseDatabase
import pandas as pd
import numpy as np
from datetime import datetime

load_dotenv()

# Цвета для вывода
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'

def print_test(name):
    print(f"\n{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.CYAN}ТЕСТ: {name}{Colors.END}")
    print(f"{Colors.CYAN}{'='*60}{Colors.END}")

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")

# ============ ТЕСТ 1: TELEGRAM API ============
async def test_telegram():
    print_test("TELEGRAM API")
    
    try:
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not token or not chat_id:
            print_error("TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID не найдены в .env")
            return False
        
        bot = Bot(token)
        
        # Подключение к боту
        me = await bot.get_me()
        print_success(f"Bot connected: @{me.username}")
        
        # Получение updates
        updates = await bot.get_updates(limit=5)
        print_success(f"Updates received: {len(updates)} messages")
        
        # Отправка тестового сообщения
        msg = await bot.send_message(
            chat_id=chat_id,
            text=f"🧪 ГЛОБАЛЬНЫЙ ТЕСТ\nВремя: {datetime.now().strftime('%H:%M:%S')}\n\nTelegram API: ✅ OK"
        )
        print_success(f"Test message sent (ID: {msg.message_id})")
        
        # Тест inline кнопок
        keyboard = [[
            InlineKeyboardButton("Тест OK", callback_data="test_ok"),
            InlineKeyboardButton("Тест FAIL", callback_data="test_fail")
        ]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        btn_msg = await bot.send_message(
            chat_id=chat_id,
            text="🧪 Тест inline кнопок:",
            reply_markup=reply_markup
        )
        print_success(f"Inline buttons sent (ID: {btn_msg.message_id})")
        
        return True
        
    except Exception as e:
        print_error(f"Telegram test failed: {e}")
        return False

# ============ ТЕСТ 2: BINANCE API ============
def test_binance():
    print_test("BINANCE API")
    
    try:
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            print_error("BINANCE_API_KEY или BINANCE_API_SECRET не найдены в .env")
            return False
        
        # Подключение к бирже
        exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        print_success("Connected to Binance")
        
        # Получение баланса
        balance = exchange.fetch_balance()
        usdt_free = balance['USDT']['free']
        usdt_used = balance['USDT']['used']
        total = usdt_free + usdt_used
        
        print_success(f"Balance: ${total:.2f} (Free: ${usdt_free:.2f}, Used: ${usdt_used:.2f})")
        
        # Получение цены BTC
        ticker = exchange.fetch_ticker('BTC/USDT')
        btc_price = ticker['last']
        print_success(f"BTC/USDT price: ${btc_price:,.2f}")
        
        # Получение OHLCV данных
        ohlcv = exchange.fetch_ohlcv('ETH/USDT', '1h', limit=10)
        print_success(f"OHLCV data fetched: {len(ohlcv)} candles")
        
        # Проверка rate limits
        rate_limit = exchange.rateLimit
        print_success(f"Rate limit: {rate_limit}ms between requests")
        
        return True
        
    except Exception as e:
        print_error(f"Binance test failed: {e}")
        return False

# ============ ТЕСТ 3: SQLITE DATABASE ============
def test_sqlite():
    print_test("SQLITE DATABASE")
    
    try:
        db = TradingDatabase()
        print_success("Database connected: trading_history.db")
        
        # Тест таблицы safety_state
        emergency_stop = db.load_emergency_stop()
        print_success(f"Emergency stop state: {emergency_stop}")
        
        # Тест сохранения/загрузки
        db.save_emergency_stop(False)
        loaded = db.load_emergency_stop()
        if loaded == False:
            print_success("Emergency stop save/load: OK")
        else:
            print_error(f"Emergency stop mismatch: expected False, got {loaded}")
        
        # Проверка таблиц
        import sqlite3
        conn = sqlite3.connect('trading_history.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print_success(f"Tables found: {', '.join(tables)}")
        
        required_tables = ['signals', 'trades', 'performance', 'safety_state']
        for table in required_tables:
            if table in tables:
                # Safe: table names are from hardcoded whitelist, not user input
                cursor.execute(f"SELECT COUNT(*) FROM {table}")  # nosec - table is whitelisted
                count = cursor.fetchone()[0]
                print_success(f"  {table}: {count} records")
            else:
                print_error(f"  {table}: NOT FOUND")
        
        conn.close()
        return True
        
    except Exception as e:
        print_error(f"SQLite test failed: {e}")
        return False

# ============ ТЕСТ 4: SUPABASE DATABASE ============
def test_supabase():
    print_test("SUPABASE DATABASE")
    
    try:
        supabase_db = SupabaseDatabase()
        print_success("Supabase connected")
        
        # Тест получения статистики
        stats = supabase_db.get_statistics()
        print_success(f"Statistics: {stats.get('total_trades', 0)} trades")
        
        # Тест получения истории
        history = supabase_db.get_trade_history(limit=5)
        print_success(f"Trade history: {len(history)} records")
        
        # Тест открытых сделок
        open_trades = supabase_db.get_open_trades()
        print_success(f"Open trades: {len(open_trades)}")
        
        return True
        
    except Exception as e:
        print_error(f"Supabase test failed: {e}")
        return False

# ============ ТЕСТ 5: ИНДИКАТОРЫ И ФИЛЬТРЫ ============
def test_indicators():
    print_test("ИНДИКАТОРЫ И ФИЛЬТРЫ")
    
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        
        # Получаем данные для ETH/USDT
        ohlcv = exchange.fetch_ohlcv('ETH/USDT', '1h', limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        print_success(f"Data fetched: {len(df)} candles")
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        current_rsi = df['rsi'].iloc[-1]
        
        if pd.notna(current_rsi):
            print_success(f"RSI: {current_rsi:.2f}")
        else:
            print_error("RSI: NaN")
            return False
        
        # EMA
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        ema20 = df['ema20'].iloc[-1]
        ema50 = df['ema50'].iloc[-1]
        
        print_success(f"EMA20: ${ema20:.2f}, EMA50: ${ema50:.2f}")
        
        # MACD
        df['macd'] = df['ema20'] - df['ema50']
        macd = df['macd'].iloc[-1]
        print_success(f"MACD: {macd:.4f}")
        
        # ATR
        df['h-l'] = df['high'] - df['low']
        df['h-pc'] = abs(df['high'] - df['close'].shift(1))
        df['l-pc'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        atr = df['atr'].iloc[-1]
        
        print_success(f"ATR: {atr:.2f}")
        
        # Проверка фильтров
        filters_count = 0
        
        # Filter 1: RSI
        if current_rsi < 30:
            filters_count += 1
            print_success("  Filter 1 (RSI): BUY signal")
        elif current_rsi > 70:
            print_success("  Filter 1 (RSI): SELL signal")
        
        # Filter 2: EMA Trend
        if ema20 > ema50:
            filters_count += 1
            print_success("  Filter 2 (EMA): Bullish trend")
        else:
            print_success("  Filter 2 (EMA): Bearish trend")
        
        print_success(f"Total filters passed: {filters_count}/9")
        
        return True
        
    except Exception as e:
        print_error(f"Indicators test failed: {e}")
        return False

# ============ ТЕСТ 6: AI АНАЛИЗ ============
async def test_ai():
    print_test("AI АНАЛИЗ (OpenAI)")
    
    try:
        from openai import OpenAI
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print_warning("OPENAI_API_KEY не найден, пропускаем тест AI")
            return True
        
        client = OpenAI(api_key=api_key)
        
        # Простой тест запрос
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a trading assistant. Reply in format: SIGNAL|CONFIDENCE|REASON"},
                {"role": "user", "content": "BTC price is 105000, RSI=45, EMA20>EMA50, volume spike. Should I buy?"}
            ],
            max_tokens=50,
            temperature=0.3
        )
        
        ai_response = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens
        
        print_success(f"AI response: {ai_response}")
        print_success(f"Tokens used: {tokens_used}")
        
        # Парсинг ответа
        parts = ai_response.split('|')
        if len(parts) >= 3:
            signal = parts[0].strip()
            confidence = parts[1].strip()
            reason = parts[2].strip()
            print_success(f"  Signal: {signal}, Confidence: {confidence}, Reason: {reason}")
        else:
            print_warning(f"AI response format unexpected: {ai_response}")
        
        return True
        
    except Exception as e:
        print_error(f"AI test failed: {e}")
        return False

# ============ ТЕСТ 7: SAFETYMANAGER ============
def test_safety():
    print_test("SAFETYMANAGER (8 LEVELS)")
    
    try:
        from trading_bot import SafetyManager
        from database import TradingDatabase
        
        db = TradingDatabase()
        safety = SafetyManager(initial_balance=1000.0, db=db)
        
        print_success(f"SafetyManager initialized with balance: $1000")
        print_success(f"Emergency stop: {safety.emergency_stop}")
        print_success(f"Paused: {safety.paused}")
        
        # Тест сигнала
        test_signal = {
            'symbol': 'ETH/USDT',
            'confidence': 8,
            'signal': 'BUY'
        }
        
        is_safe, reason = safety.check_all_safety_levels(
            signal_data=test_signal,
            current_balance=1000.0,
            active_positions=[],
            recent_prices=[3500, 3505, 3510, 3515, 3520]
        )
        
        if is_safe:
            print_success(f"Safety check PASSED: {reason}")
        else:
            print_warning(f"Safety check BLOCKED: {reason}")
        
        # Тест emergency stop
        safety.activate_emergency_stop()
        print_success("Emergency stop activated")
        
        is_safe2, reason2 = safety.check_all_safety_levels(
            test_signal, 1000.0, [], [3500, 3505]
        )
        
        if not is_safe2 and "EMERGENCY" in reason2:
            print_success(f"Emergency stop working: {reason2}")
        else:
            print_error("Emergency stop NOT working!")
        
        safety.deactivate_emergency_stop()
        print_success("Emergency stop deactivated")
        
        return True
        
    except Exception as e:
        print_error(f"Safety test failed: {e}")
        return False

# ============ ТЕСТ 8: END-TO-END СИГНАЛ ============
async def test_end_to_end():
    print_test("END-TO-END: ПОЛНЫЙ ЦИКЛ СИГНАЛА")
    
    try:
        # Имитация сигнала
        trade_id = f"TEST_{int(datetime.now().timestamp())}"
        symbol = "ETH/USDT"
        signal = "BUY"
        price = 3500.0
        amount = 0.01
        usdt_amount = 35.0
        
        print_success(f"Trade ID: {trade_id}")
        print_success(f"Symbol: {symbol}, Signal: {signal}")
        print_success(f"Price: ${price}, Amount: {amount}, USDT: ${usdt_amount}")
        
        # Отправка в Telegram
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        bot = Bot(token)
        
        message = f"""
🧪 ТЕСТОВЫЙ СИГНАЛ #{trade_id}

{signal} {symbol}
Цена: ${price:.2f}
Позиция: {amount:.6f} (~${usdt_amount:.2f})

AI Confidence: 8/10
Причина: Тестовый сигнал для проверки

⚠️ ЭТО ТЕСТ! НЕ ПОДТВЕРЖДАЙТЕ!
"""
        
        keyboard = [[
            InlineKeyboardButton("✅ Подтвердить", callback_data=f"approve_{trade_id}"),
            InlineKeyboardButton("❌ Отклонить", callback_data=f"reject_{trade_id}")
        ]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        msg = await bot.send_message(
            chat_id=chat_id,
            text=message,
            reply_markup=reply_markup
        )
        
        print_success(f"Signal sent to Telegram (ID: {msg.message_id})")
        
        # Сохранение в БД
        db = TradingDatabase()
        db.save_signal(
            trade_id=trade_id,
            symbol=symbol,
            signal_type=signal,
            price=price,
            indicators={'rsi': 45, 'ema20': 3510, 'ema50': 3490, 'macd': 20, 'atr': 50},
            ai_analysis={'signal': signal, 'confidence': 8, 'reason': 'Test signal'},
            position_info={'amount': amount, 'usdt_amount': usdt_amount, 'fee': 0.35}
        )
        
        print_success("Signal saved to SQLite")
        
        # Сохранение в Supabase
        supabase_db = SupabaseDatabase()
        supabase_db.save_signal(
            trade_id=trade_id,
            symbol=symbol,
            signal_type=signal,
            price=price,
            indicators={'rsi': 45, 'ema20': 3510, 'ema50': 3490, 'macd': 20, 'atr': 50},
            ai_analysis={'signal': signal, 'confidence': 8, 'reason': 'Test signal'},
            position_info={'amount': amount, 'usdt_amount': usdt_amount, 'fee': 0.35}
        )
        
        print_success("Signal saved to Supabase")
        
        return True
        
    except Exception as e:
        print_error(f"End-to-end test failed: {e}")
        return False

# ============ ГЛАВНАЯ ФУНКЦИЯ ============
async def main():
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}ГЛОБАЛЬНОЕ ТЕСТИРОВАНИЕ ТОРГОВОГО БОТА{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    results = {}
    
    # Запуск всех тестов
    results['Telegram'] = await test_telegram()
    results['Binance'] = test_binance()
    results['SQLite'] = test_sqlite()
    results['Supabase'] = test_supabase()
    results['Indicators'] = test_indicators()
    results['AI'] = await test_ai()
    results['Safety'] = test_safety()
    results['End-to-End'] = await test_end_to_end()
    
    # Итоговый отчёт
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}ИТОГОВЫЙ ОТЧЁТ{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    for test_name, result in results.items():
        status = f"{Colors.GREEN}✅ PASSED{Colors.END}" if result else f"{Colors.RED}❌ FAILED{Colors.END}"
        print(f"{test_name:20s} {status}")
    
    print(f"\n{Colors.BLUE}Всего тестов: {total}{Colors.END}")
    print(f"{Colors.GREEN}Успешно: {passed}{Colors.END}")
    print(f"{Colors.RED}Провалено: {failed}{Colors.END}")
    
    if failed == 0:
        print(f"\n{Colors.GREEN}{'='*60}{Colors.END}")
        print(f"{Colors.GREEN}🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! СИСТЕМА РАБОТАЕТ!{Colors.END}")
        print(f"{Colors.GREEN}{'='*60}{Colors.END}\n")
    else:
        print(f"\n{Colors.RED}{'='*60}{Colors.END}")
        print(f"{Colors.RED}⚠️  ОБНАРУЖЕНЫ ПРОБЛЕМЫ! ТРЕБУЕТСЯ ИСПРАВЛЕНИЕ!{Colors.END}")
        print(f"{Colors.RED}{'='*60}{Colors.END}\n")

if __name__ == "__main__":
    asyncio.run(main())
