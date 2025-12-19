"""
Quick test script for Market Regime Detection (Phase 5)
Tests HMM model without running full bot
"""

import sys
import os
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.market_regime import MarketRegimeManager, MarketRegime
import ccxt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("MARKET REGIME DETECTION TEST")
    logger.info("=" * 60)
    
    # Create exchange
    logger.info("\n[1/5] Initializing Binance exchange...")
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        logger.info("✅ Exchange initialized")
    except Exception as e:
        logger.error(f"❌ Exchange initialization failed: {e}")
        return
    
    # Create regime manager
    logger.info("\n[2/5] Initializing MarketRegimeManager...")
    try:
        manager = MarketRegimeManager(db_path="trading_history.db")
        logger.info("✅ MarketRegimeManager initialized")
    except Exception as e:
        logger.error(f"❌ MarketRegimeManager initialization failed: {e}")
        return
    
    # Fit HMM model
    logger.info("\n[3/5] Fitting HMM model on BTC/USDT data...")
    logger.info("   Fetching 30 days of 1h candles...")
    logger.info("   This may take ~20 seconds...")
    
    try:
        success = manager.fit_model(exchange, "BTC/USDT")
        
        if success:
            logger.info("✅ HMM model fitted successfully!")
            
            # Show regime mapping
            status = manager.detector.get_status()
            logger.info(f"   Regime mapping:")
            for state, regime in status['regime_mapping'].items():
                logger.info(f"     State {state} → {regime}")
        else:
            logger.error("❌ HMM fitting failed")
            return
    except Exception as e:
        logger.error(f"❌ HMM fitting error: {e}")
        return
    
    # Detect current regime
    logger.info("\n[4/5] Detecting current market regime...")
    
    try:
        regime = manager.detect_regime(exchange, "BTC/USDT")
        logger.info(f"✅ Current regime detected: {regime.value}")
        
        # Get strategy for this regime
        strategy = manager.get_current_strategy()
        logger.info(f"   Strategy: {strategy['description']}")
        logger.info(f"   Confidence threshold: {strategy['confidence_threshold']}")
        logger.info(f"   Position size multiplier: {strategy['position_size_multiplier']}x")
        logger.info(f"   Aggressive mode: {strategy['aggressive_mode']}")
        
        # Should trade?
        should_trade = manager.should_trade_now()
        if should_trade:
            logger.info("   ✅ Trading RECOMMENDED")
        else:
            logger.info("   🚨 Trading NOT RECOMMENDED")
    
    except Exception as e:
        logger.error(f"❌ Regime detection error: {e}")
        return
    
    # Get statistics
    logger.info("\n[5/5] Getting regime statistics...")
    
    try:
        stats = manager.detector.get_regime_statistics()
        
        if 'message' in stats:
            logger.info(f"   {stats['message']}")
        else:
            logger.info(f"   Total detections: {stats['total_detections']}")
            logger.info(f"   Current: {stats['current_regime']} (prob={stats['current_probability']:.2f})")
            
            if stats['regime_percentages']:
                logger.info("   Distribution:")
                for regime, pct in sorted(stats['regime_percentages'].items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"     {regime}: {pct:.1f}%")
    
    except Exception as e:
        logger.error(f"❌ Statistics error: {e}")
        return
    
    logger.info("\n" + "=" * 60)
    logger.info("ТЕСТ ЗАВЕРШЕН!")
    logger.info("=" * 60)
    logger.info("\nИспользуйте в Telegram боте:")
    logger.info("  /regime_fit - Обучить HMM модель")
    logger.info("  /regime - Определить текущий режим")
    logger.info("  /regime_history - История режимов")
    logger.info("  /regime_stats - Статистика из БД")
    
    logger.info("\n📊 Режимы рынка:")
    logger.info("  📈 TREND_UP - Восходящий тренд (агрессивная покупка)")
    logger.info("  📉 TREND_DOWN - Нисходящий тренд (осторожная торговля)")
    logger.info("  ↔️ RANGE - Боковое движение (скальпинг)")
    logger.info("  ⚡ HIGH_VOLATILITY - Высокая волатильность (уменьшенные позиции)")
    logger.info("  🚨 CRASH - Обвал рынка (только выход из позиций!)")

if __name__ == "__main__":
    main()
