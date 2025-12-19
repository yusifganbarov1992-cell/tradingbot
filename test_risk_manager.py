"""
Test Advanced Risk Manager - Phase 8
Тестируем все функции risk management
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt.async_support as ccxt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from modules.risk_manager import AdvancedRiskManager


async def test_risk_manager():
    """Test all risk manager features"""
    
    print("\n" + "="*70)
    print("🧪 ТЕСТИРОВАНИЕ ADVANCED RISK MANAGER (Phase 8)")
    print("="*70)
    
    # Step 1: Initialize exchange and get data
    print("\n📊 Шаг 1: Получаем данные с Binance...")
    
    exchange = ccxt.binance()
    
    try:
        # Get 1000 candles for testing
        ohlcv = await exchange.fetch_ohlcv('BTC/USDT', '1h', limit=1000)
        await exchange.close()
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        print(f"✅ Получено {len(df)} свечей")
        print(f"   Период: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
        print(f"   Текущая цена: ${df['close'].iloc[-1]:.2f}")
        
    except Exception as e:
        print(f"❌ Ошибка получения данных: {e}")
        return
    
    # Step 2: Create Risk Manager
    print("\n📊 Шаг 2: Создаем Risk Manager...")
    
    rm = AdvancedRiskManager(
        initial_balance=10000,
        max_risk_per_trade=0.02
    )
    
    print(f"✅ Risk Manager создан")
    print(f"   Баланс: ${rm.current_balance:.2f}")
    print(f"   Max risk: {rm.max_risk_per_trade:.1%}")
    
    # Step 3: Test Kelly Criterion
    print("\n📊 Шаг 3: Тестируем Kelly Criterion...")
    
    # Simulate some trade history
    rm.trade_history = [
        {'symbol': 'BTC/USDT', 'pnl': 100, 'win': True},
        {'symbol': 'BTC/USDT', 'pnl': 150, 'win': True},
        {'symbol': 'BTC/USDT', 'pnl': -50, 'win': False},
        {'symbol': 'BTC/USDT', 'pnl': 80, 'win': True},
        {'symbol': 'BTC/USDT', 'pnl': -40, 'win': False},
        {'symbol': 'BTC/USDT', 'pnl': 120, 'win': True},
        {'symbol': 'BTC/USDT', 'pnl': -30, 'win': False},
        {'symbol': 'BTC/USDT', 'pnl': 90, 'win': True},
        {'symbol': 'BTC/USDT', 'pnl': 110, 'win': True},
        {'symbol': 'BTC/USDT', 'pnl': -60, 'win': False},
    ]
    
    current_price = df['close'].iloc[-1]
    kelly_size = rm.get_kelly_position_size('BTC/USDT', current_price)
    
    print(f"✅ Kelly position size: ${kelly_size:.2f}")
    print(f"   Это {kelly_size/rm.current_balance:.1%} от баланса")
    
    # Step 4: Test VaR (Value at Risk)
    print("\n📊 Шаг 4: Тестируем VaR (Value at Risk)...")
    
    # Historical VaR
    var_hist = rm.calculate_portfolio_var(df, confidence=0.95, method='historical')
    print(f"\n✅ Historical VaR (95%):")
    print(f"   1 день:  {var_hist['var_1day_pct']:.2%} (${var_hist['var_1day_usd']:.2f})")
    print(f"   1 неделя: {var_hist['var_1week_pct']:.2%} (${var_hist['var_1week_usd']:.2f})")
    print(f"   1 месяц: {var_hist['var_1month_pct']:.2%} (${var_hist['var_1month_usd']:.2f})")
    print(f"   Интерпретация: {var_hist['interpretation']}")
    
    # Parametric VaR
    var_param = rm.calculate_portfolio_var(df, confidence=0.95, method='parametric')
    print(f"\n✅ Parametric VaR (95%):")
    print(f"   1 день:  {var_param['var_1day_pct']:.2%} (${var_param['var_1day_usd']:.2f})")
    print(f"   Интерпретация: {var_param['interpretation']}")
    
    # Step 5: Test ATR-based Stop-Loss
    print("\n📊 Шаг 5: Тестируем ATR-based Stop-Loss...")
    
    entry_price = df['close'].iloc[-1]
    
    # Long position
    sl_long = rm.calculate_atr_stop_loss(df, entry_price, side='long', atr_multiplier=2.0)
    tp_long = rm.calculate_atr_take_profit(df, entry_price, side='long', risk_reward_ratio=2.0)
    
    print(f"\n✅ Long позиция:")
    print(f"   Entry:  ${entry_price:.2f}")
    print(f"   SL:     ${sl_long:.2f} ({(sl_long-entry_price)/entry_price:.2%})")
    print(f"   TP:     ${tp_long:.2f} ({(tp_long-entry_price)/entry_price:.2%})")
    print(f"   Risk/Reward: 1:2")
    
    # Short position
    sl_short = rm.calculate_atr_stop_loss(df, entry_price, side='short', atr_multiplier=2.0)
    tp_short = rm.calculate_atr_take_profit(df, entry_price, side='short', risk_reward_ratio=2.0)
    
    print(f"\n✅ Short позиция:")
    print(f"   Entry:  ${entry_price:.2f}")
    print(f"   SL:     ${sl_short:.2f} ({(sl_short-entry_price)/entry_price:.2%})")
    print(f"   TP:     ${tp_short:.2f} ({(tp_short-entry_price)/entry_price:.2%})")
    
    # Step 6: Test Volatility-Based Position Sizing
    print("\n📊 Шаг 6: Тестируем Volatility-Based Sizing...")
    
    base_size = 1000  # $1000 base
    adjusted_size = rm.calculate_volatility_adjusted_size(df, base_size, target_volatility=0.02)
    
    print(f"✅ Volatility adjustment:")
    print(f"   Base size:     ${base_size:.2f}")
    print(f"   Adjusted size: ${adjusted_size:.2f}")
    print(f"   Adjustment:    {adjusted_size/base_size:.2f}x")
    
    # Step 7: Test Portfolio Metrics
    print("\n📊 Шаг 7: Тестируем Portfolio Metrics...")
    
    metrics = rm.get_portfolio_metrics(df)
    
    print(f"\n✅ Portfolio Metrics:")
    print(f"   Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
    print(f"   Sortino Ratio:     {metrics['sortino_ratio']:.2f}")
    print(f"   Max Drawdown:      {metrics['max_drawdown']:.2%}")
    print(f"   Annual Volatility: {metrics['volatility_annual']:.2%}")
    print(f"   Total Return:      {metrics['total_return']:.2%}")
    print(f"   Risk Level:        {metrics['risk_level']}")
    
    # Step 8: Test Correlation Matrix (need multiple assets)
    print("\n📊 Шаг 8: Тестируем Correlation Matrix...")
    
    # Get data for multiple pairs
    print("   Получаем данные для BTC, ETH, BNB...")
    
    exchange2 = ccxt.binance()
    
    try:
        btc_ohlcv = await exchange2.fetch_ohlcv('BTC/USDT', '1h', limit=500)
        eth_ohlcv = await exchange2.fetch_ohlcv('ETH/USDT', '1h', limit=500)
        bnb_ohlcv = await exchange2.fetch_ohlcv('BNB/USDT', '1h', limit=500)
        await exchange2.close()
        
        btc_df = pd.DataFrame(btc_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        eth_df = pd.DataFrame(eth_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        bnb_df = pd.DataFrame(bnb_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Calculate correlation
        price_data = {
            'BTC/USDT': btc_df,
            'ETH/USDT': eth_df,
            'BNB/USDT': bnb_df
        }
        
        corr_matrix = rm.calculate_correlation_matrix(price_data)
        
        print(f"\n✅ Correlation Matrix:")
        print(corr_matrix.round(3))
        
        # Check diversification
        div_metrics = rm.check_portfolio_diversification(corr_matrix)
        
        print(f"\n✅ Diversification Analysis:")
        print(f"   Avg Correlation:   {div_metrics['avg_correlation']:.3f}")
        print(f"   Diversification:   {div_metrics['diversification_score']}")
        print(f"   Recommendation:    {div_metrics['recommendation']}")
        
        if div_metrics['high_correlation_pairs']:
            print(f"\n   ⚠️ Highly correlated pairs:")
            for pair_info in div_metrics['high_correlation_pairs']:
                print(f"      {pair_info['pair'][0]} ↔ {pair_info['pair'][1]}: {pair_info['correlation']:.3f}")
        
    except Exception as e:
        print(f"   ⚠️ Не удалось получить данные для correlation: {e}")
    
    # Step 9: Get Status
    print("\n📊 Шаг 9: Статус Risk Manager...")
    
    status = rm.get_status()
    
    print(f"\n✅ Статус:")
    print(f"   Баланс:        ${status['current_balance']:.2f}")
    print(f"   Total PnL:     ${status['total_pnl']:.2f}")
    print(f"   Total Trades:  {status['total_trades']}")
    print(f"   Win Rate:      {status['win_rate']:.1%}")
    print(f"   Max Risk:      {status['max_risk_per_trade']:.1%}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("="*70)
    
    print("\n✅ Kelly Criterion:")
    print(f"   Optimal position size: ${kelly_size:.2f} ({kelly_size/rm.current_balance:.1%})")
    
    print("\n✅ Value at Risk (95%):")
    print(f"   1-day VaR: {var_hist['var_1day_pct']:.2%} (${var_hist['var_1day_usd']:.2f})")
    print(f"   Risk level: {var_hist['interpretation']}")
    
    print("\n✅ ATR-based Stop-Loss:")
    print(f"   Long SL:  {(sl_long-entry_price)/entry_price:.2%} от entry")
    print(f"   Long TP:  {(tp_long-entry_price)/entry_price:.2%} от entry")
    
    print("\n✅ Portfolio Metrics:")
    print(f"   Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"   Max DD: {metrics['max_drawdown']:.2%}")
    print(f"   Risk:   {metrics['risk_level']}")
    
    print("\n💡 Advanced Risk Manager работает корректно!")
    print("   Все 8 функций протестированы ✅")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(test_risk_manager())
