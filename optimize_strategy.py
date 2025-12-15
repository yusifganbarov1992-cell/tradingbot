"""
ОПТИМИЗАЦИЯ СТРАТЕГИИ ЧЕРЕЗ МАССОВЫЙ БЭКТЕСТИНГ

Тестирует разные комбинации параметров на разных монетах
и находит оптимальные настройки.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from backtest import BacktestEngine
import ccxt

# Параметры для тестирования
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'LINK/USDT']
DAYS = 30

# Варианты параметров стратегии
PARAM_GRID = {
    'ai_threshold': [2, 3, 4],  # Минимум фильтров для AI анализа
    'confidence_threshold': [7, 8, 9],  # Минимальная уверенность AI
    'position_size': [0.05, 0.10, 0.15],  # % от баланса
    'sl_multiplier': [2.0, 2.5, 3.0],  # Stop Loss в ATR
    'tp_multiplier': [3.0, 4.0, 5.0],  # Take Profit в ATR
}

def run_optimization():
    """Запуск оптимизации на всех комбинациях"""
    
    print("=" * 80)
    print("ОПТИМИЗАЦИЯ СТРАТЕГИИ ТРЕЙДИНГ-БОТА")
    print("=" * 80)
    print(f"Монеты: {', '.join(SYMBOLS)}")
    print(f"Период: {DAYS} дней")
    print(f"Комбинаций: {len(PARAM_GRID['ai_threshold']) * len(PARAM_GRID['confidence_threshold']) * len(PARAM_GRID['position_size']) * len(PARAM_GRID['sl_multiplier']) * len(PARAM_GRID['tp_multiplier'])}")
    print("=" * 80)
    
    # Инициализация биржи
    from dotenv import load_dotenv
    load_dotenv()
    
    exchange = ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET_KEY'),
        'enableRateLimit': True,
    })
    
    results = []
    total_tests = 0
    
    # Перебор всех комбинаций
    for symbol in SYMBOLS:
        print(f"\n{'='*80}")
        print(f"ТЕСТИРОВАНИЕ: {symbol}")
        print(f"{'='*80}")
        
        # Загрузка данных один раз для каждой монеты
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=DAYS)
            
            ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=DAYS*24)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            print(f"✅ Загружено {len(df)} свечей")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки {symbol}: {e}")
            continue
        
        # Тестируем разные параметры
        for ai_thresh in PARAM_GRID['ai_threshold']:
            for conf_thresh in PARAM_GRID['confidence_threshold']:
                for pos_size in PARAM_GRID['position_size']:
                    for sl_mult in PARAM_GRID['sl_multiplier']:
                        for tp_mult in PARAM_GRID['tp_multiplier']:
                            
                            total_tests += 1
                            
                            # Создание движка с параметрами
                            engine = BacktestEngine(
                                initial_balance=1000,
                                position_size=pos_size,
                                sl_multiplier=sl_mult,
                                tp_multiplier=tp_mult,
                                ai_threshold=ai_thresh,
                                confidence_threshold=conf_thresh
                            )
                            
                            # Запуск бэктеста
                            trades = engine.run_backtest(df)
                            metrics = engine.calculate_metrics()
                            
                            # Сохранение результатов
                            result = {
                                'symbol': symbol,
                                'ai_threshold': ai_thresh,
                                'confidence_threshold': conf_thresh,
                                'position_size': pos_size,
                                'sl_multiplier': sl_mult,
                                'tp_multiplier': tp_mult,
                                'total_trades': metrics['total_trades'],
                                'win_rate': metrics['win_rate'],
                                'roi': metrics['roi'],
                                'sharpe_ratio': metrics['sharpe_ratio'],
                                'profit_factor': metrics['profit_factor'],
                                'max_drawdown': metrics['max_drawdown'],
                                'final_balance': metrics['final_balance']
                            }
                            results.append(result)
                            
                            # Прогресс
                            if total_tests % 10 == 0:
                                print(f"Прогресс: {total_tests} тестов завершено...")
    
    # Преобразование в DataFrame
    results_df = pd.DataFrame(results)
    
    # Сортировка по ROI (лучшие сверху)
    results_df = results_df.sort_values('roi', ascending=False)
    
    # Сохранение в CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'optimization_results_{timestamp}.csv'
    results_df.to_csv(csv_filename, index=False)
    
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print("=" * 80)
    print(f"Всего тестов: {total_tests}")
    print(f"Результаты сохранены: {csv_filename}")
    
    # Топ-10 лучших комбинаций
    print("\n📊 ТОП-10 ЛУЧШИХ КОМБИНАЦИЙ:")
    print("=" * 80)
    
    top_10 = results_df.head(10)
    for idx, row in top_10.iterrows():
        print(f"\n#{top_10.index.get_loc(idx) + 1}")
        print(f"  Монета: {row['symbol']}")
        print(f"  ROI: {row['roi']:.2f}% | Win Rate: {row['win_rate']:.1f}% | Sharpe: {row['sharpe_ratio']:.2f}")
        print(f"  Сделок: {row['total_trades']} | Profit Factor: {row['profit_factor']:.2f}")
        print(f"  Параметры:")
        print(f"    - AI Threshold: {row['ai_threshold']} фильтров")
        print(f"    - Confidence: {row['confidence_threshold']}/10")
        print(f"    - Position Size: {row['position_size']*100:.0f}%")
        print(f"    - Stop Loss: {row['sl_multiplier']} ATR")
        print(f"    - Take Profit: {row['tp_multiplier']} ATR")
    
    # Статистика по монетам
    print("\n" + "=" * 80)
    print("СТАТИСТИКА ПО МОНЕТАМ (средний ROI):")
    print("=" * 80)
    
    symbol_stats = results_df.groupby('symbol').agg({
        'roi': 'mean',
        'win_rate': 'mean',
        'total_trades': 'mean',
        'sharpe_ratio': 'mean'
    }).sort_values('roi', ascending=False)
    
    for symbol, stats in symbol_stats.iterrows():
        print(f"{symbol}: ROI={stats['roi']:.2f}% | Win={stats['win_rate']:.1f}% | Trades={stats['total_trades']:.0f} | Sharpe={stats['sharpe_ratio']:.2f}")
    
    # Лучшая конфигурация
    print("\n" + "=" * 80)
    print("🏆 РЕКОМЕНДУЕМАЯ КОНФИГУРАЦИЯ:")
    print("=" * 80)
    
    best = results_df.iloc[0]
    print(f"Монета: {best['symbol']}")
    print(f"ROI: {best['roi']:.2f}%")
    print(f"Win Rate: {best['win_rate']:.1f}%")
    print(f"Sharpe Ratio: {best['sharpe_ratio']:.2f}")
    print(f"Profit Factor: {best['profit_factor']:.2f}")
    print(f"\nПараметры для trading_bot.py:")
    print(f"  AI_FILTERS_THRESHOLD = {int(best['ai_threshold'])}")
    print(f"  AI_CONFIDENCE_MIN = {int(best['confidence_threshold'])}")
    print(f"  POSITION_SIZE = {best['position_size']}")
    print(f"  STOP_LOSS_ATR = {best['sl_multiplier']}")
    print(f"  TAKE_PROFIT_ATR = {best['tp_multiplier']}")
    
    return results_df, best

if __name__ == '__main__':
    results, best_config = run_optimization()
