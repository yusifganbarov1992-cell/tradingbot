"""
Тестирование подключения к Supabase
"""
import os
from dotenv import load_dotenv
from database_supabase import SupabaseDatabase
from datetime import datetime

load_dotenv()

def test_supabase():
    print("🔍 Тестирование Supabase...")
    
    try:
        # Инициализация
        db = SupabaseDatabase()
        print("✅ Подключение успешно")
        
        # Тестовый сигнал
        trade_id = f"TEST_{int(datetime.now().timestamp())}"
        success = db.save_signal(
            trade_id=trade_id,
            symbol='BTC/USDT',
            signal_type='BUY',
            price=42000.0,
            indicators={
                'rsi': 35.5,
                'ema20': 41800.0,
                'ema50': 41500.0,
                'macd': 150.0,
                'volume': 1500000.0,
                'avg_volume': 1200000.0,
                'atr': 500.0,
                'filters_passed': 7
            },
            ai_analysis={
                'signal': 'BUY',
                'confidence': 8,
                'reason': 'Strong bullish momentum with oversold RSI'
            },
            position_info={
                'amount': 0.001,
                'usdt_amount': 42.0,
                'fee': 0.042
            }
        )
        
        if success:
            print("✅ Сигнал сохранён")
        else:
            print("❌ Ошибка сохранения сигнала")
            return
        
        # Тестовая сделка
        success = db.save_trade(
            trade_id=trade_id,
            symbol='BTC/USDT',
            side='BUY',
            entry_price=42000.0,
            amount=0.001,
            usdt_amount=42.0,
            mode='test',
            stop_loss=41000.0,
            take_profit=43000.0,
            fee=0.042
        )
        
        if success:
            print("✅ Сделка сохранена")
        else:
            print("❌ Ошибка сохранения сделки")
            return
        
        # Получить открытые сделки
        open_trades = db.get_open_trades()
        print(f"📊 Открытых сделок: {len(open_trades)}")
        
        # Закрыть тестовую сделку
        success = db.update_trade(
            trade_id=trade_id,
            exit_price=42500.0,
            pnl=0.5,
            pnl_percent=1.19,
            fee=0.0425
        )
        
        if success:
            print("✅ Сделка закрыта")
        else:
            print("❌ Ошибка закрытия сделки")
            return
        
        # Статистика
        stats = db.get_statistics()
        print("\n📈 Статистика:")
        print(f"   Всего сделок: {stats.get('total_trades', 0)}")
        print(f"   Прибыльных: {stats.get('winning_trades', 0)}")
        print(f"   Убыточных: {stats.get('losing_trades', 0)}")
        print(f"   Win Rate: {stats.get('win_rate', 0):.2f}%")
        print(f"   Общий PNL: ${stats.get('total_pnl', 0):.2f}")
        
        # История
        history = db.get_trade_history(limit=10)
        print(f"\n📜 Последних сделок: {len(history)}")
        
        print("\n✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print("🎉 Supabase полностью подключена и работает!")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_supabase()
