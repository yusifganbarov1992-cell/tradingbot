"""
Market Regime Detection Module - Phase 5
Использует Hidden Markov Models (HMM) для определения текущего режима рынка:
- TREND_UP: восходящий тренд
- TREND_DOWN: нисходящий тренд
- RANGE: боковое движение (консолидация)
- HIGH_VOLATILITY: высокая волатильность
- CRASH: обвал рынка
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from enum import Enum
import sqlite3
from hmmlearn import hmm

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Типы рыночных режимов"""
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    CRASH = "CRASH"
    UNKNOWN = "UNKNOWN"


class RegimeDetector:
    """
    Детектор рыночных режимов на основе HMM
    
    Использует Gaussian HMM с 5 скрытыми состояниями для классификации
    рыночных условий на основе исторических данных цен.
    """
    
    def __init__(self, n_regimes: int = 5, lookback_days: int = 30):
        """
        Args:
            n_regimes: Количество скрытых состояний (режимов)
            lookback_days: Период анализа в днях
        """
        self.n_regimes = n_regimes
        self.lookback_days = lookback_days
        
        # HMM model (Gaussian)
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        self.is_fitted = False
        self.regime_mapping = {}  # Mapping from HMM states to MarketRegime
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_probability = 0.0
        self.regime_history = []  # History of detected regimes
        
        logger.info(f"📊 RegimeDetector initialized (n_regimes={n_regimes})")
    
    def _fetch_market_data(self, exchange, symbol: str = "BTC/USDT", days: int = None) -> pd.DataFrame:
        """
        Получить исторические данные с биржи
        
        Args:
            exchange: CCXT exchange instance
            symbol: Trading pair
            days: Number of days to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        if days is None:
            days = self.lookback_days
        
        try:
            # Fetch OHLCV data (1h timeframe)
            since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
            ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=since, limit=1000)
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return pd.DataFrame()
    
    def _calculate_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Рассчитать признаки для HMM
        
        Features:
        - Returns (log returns)
        - Volatility (rolling std of returns)
        - Volume ratio (current vs average)
        - Price momentum (RSI-like)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Array of features (n_samples, n_features)
        """
        if len(df) < 20:
            return np.array([])
        
        # Log returns
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility (rolling std of returns)
        df['volatility'] = df['returns'].rolling(window=10).std()
        
        # Volume ratio
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price momentum (rate of change)
        df['momentum'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        # Select features
        features = df[['returns', 'volatility', 'volume_ratio', 'momentum']].values
        
        return features
    
    def fit(self, exchange, symbol: str = "BTC/USDT") -> bool:
        """
        Обучить HMM модель на исторических данных
        
        Args:
            exchange: CCXT exchange instance
            symbol: Trading pair
            
        Returns:
            True if successful
        """
        logger.info(f"🔄 Fitting HMM model on {symbol} data...")
        
        # Fetch market data
        df = self._fetch_market_data(exchange, symbol, days=self.lookback_days)
        
        if len(df) < 50:
            logger.warning("Insufficient data for HMM fitting")
            return False
        
        # Calculate features
        features = self._calculate_features(df)
        
        if len(features) < 50:
            logger.warning("Insufficient features for HMM fitting")
            return False
        
        try:
            # Fit HMM model
            self.model.fit(features)
            self.is_fitted = True
            
            # Map HMM states to regime types
            self._map_regimes(features)
            
            logger.info(f"✅ HMM model fitted successfully ({len(features)} samples)")
            return True
        
        except Exception as e:
            logger.error(f"Error fitting HMM model: {e}")
            return False
    
    def _map_regimes(self, features: np.ndarray):
        """
        Сопоставить скрытые состояния HMM с типами режимов
        
        Анализирует средние значения признаков в каждом состоянии
        и присваивает соответствующий режим.
        """
        # Predict states for all samples
        states = self.model.predict(features)
        
        # Calculate mean features for each state
        regime_stats = {}
        for state in range(self.n_regimes):
            mask = states == state
            if np.sum(mask) > 0:
                state_features = features[mask]
                regime_stats[state] = {
                    'returns': np.mean(state_features[:, 0]),
                    'volatility': np.mean(state_features[:, 1]),
                    'volume_ratio': np.mean(state_features[:, 2]),
                    'momentum': np.mean(state_features[:, 3]),
                    'count': np.sum(mask)
                }
        
        # Map states to regimes based on characteristics
        for state, stats in regime_stats.items():
            returns = stats['returns']
            volatility = stats['volatility']
            momentum = stats['momentum']
            
            # Classification logic
            if volatility > 0.03:  # High volatility
                if returns < -0.02:  # Negative returns
                    regime = MarketRegime.CRASH
                else:
                    regime = MarketRegime.HIGH_VOLATILITY
            
            elif abs(momentum) < 0.02:  # Low momentum
                regime = MarketRegime.RANGE
            
            elif momentum > 0.02:  # Positive momentum
                regime = MarketRegime.TREND_UP
            
            elif momentum < -0.02:  # Negative momentum
                regime = MarketRegime.TREND_DOWN
            
            else:
                regime = MarketRegime.RANGE
            
            self.regime_mapping[state] = regime
            logger.info(f"   State {state} → {regime.value} (returns={returns:.4f}, vol={volatility:.4f}, mom={momentum:.4f})")
    
    def detect_current_regime(self, exchange, symbol: str = "BTC/USDT") -> Tuple[MarketRegime, float]:
        """
        Определить текущий режим рынка
        
        Args:
            exchange: CCXT exchange instance
            symbol: Trading pair
            
        Returns:
            (MarketRegime, probability)
        """
        if not self.is_fitted:
            logger.warning("HMM model not fitted. Call fit() first.")
            return MarketRegime.UNKNOWN, 0.0
        
        # Fetch recent data (last 24 hours)
        df = self._fetch_market_data(exchange, symbol, days=2)
        
        if len(df) < 20:
            logger.warning("Insufficient recent data")
            return MarketRegime.UNKNOWN, 0.0
        
        # Calculate features
        features = self._calculate_features(df)
        
        if len(features) == 0:
            logger.warning("No features calculated")
            return MarketRegime.UNKNOWN, 0.0
        
        try:
            # Get last observation
            last_features = features[-1:, :]
            
            # Predict state
            state = self.model.predict(last_features)[0]
            
            # Get state probabilities
            log_probs = self.model.score_samples(last_features)
            probability = np.exp(log_probs[0])
            
            # Map to regime
            regime = self.regime_mapping.get(state, MarketRegime.UNKNOWN)
            
            # Update current regime
            self.current_regime = regime
            self.regime_probability = float(probability)
            
            # Add to history
            self.regime_history.append({
                'timestamp': datetime.now(),
                'regime': regime.value,
                'probability': probability,
                'state': int(state)
            })
            
            # Keep only last 100 records
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            logger.info(f"📊 Current regime: {regime.value} (prob={probability:.2f})")
            return regime, probability
        
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return MarketRegime.UNKNOWN, 0.0
    
    def get_regime_statistics(self) -> Dict:
        """
        Получить статистику по режимам из истории
        
        Returns:
            Dictionary with regime statistics
        """
        if not self.regime_history:
            return {'message': 'No regime history available'}
        
        # Convert to DataFrame
        df = pd.DataFrame(self.regime_history)
        
        # Count regimes
        regime_counts = df['regime'].value_counts().to_dict()
        
        # Calculate percentages
        total = len(df)
        regime_percentages = {
            regime: (count / total * 100) for regime, count in regime_counts.items()
        }
        
        # Get current regime
        current = df.iloc[-1]
        
        # Recent regimes (last 10)
        recent = df.tail(10)['regime'].tolist()
        
        return {
            'current_regime': current['regime'],
            'current_probability': float(current['probability']),
            'regime_counts': regime_counts,
            'regime_percentages': regime_percentages,
            'recent_regimes': recent,
            'total_detections': total
        }
    
    def get_trading_strategy_for_regime(self, regime: MarketRegime) -> Dict[str, any]:
        """
        Получить рекомендуемую стратегию для режима
        
        Args:
            regime: MarketRegime
            
        Returns:
            Dictionary with strategy parameters
        """
        strategies = {
            MarketRegime.TREND_UP: {
                'description': '📈 Восходящий тренд - агрессивная покупка',
                'confidence_threshold': 7.0,
                'position_size_multiplier': 1.2,
                'stop_loss_multiplier': 0.8,  # Tighter stop loss
                'take_profit_multiplier': 1.3,  # Higher take profit
                'max_positions': 4,
                'trade_on_pullbacks': True,
                'aggressive_mode': True
            },
            MarketRegime.TREND_DOWN: {
                'description': '📉 Нисходящий тренд - осторожная торговля',
                'confidence_threshold': 8.5,
                'position_size_multiplier': 0.6,
                'stop_loss_multiplier': 1.2,  # Wider stop loss
                'take_profit_multiplier': 0.8,  # Lower take profit
                'max_positions': 2,
                'trade_on_pullbacks': False,
                'aggressive_mode': False
            },
            MarketRegime.RANGE: {
                'description': '↔️ Боковое движение - скальпинг',
                'confidence_threshold': 7.5,
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.0,
                'max_positions': 3,
                'trade_on_pullbacks': True,
                'aggressive_mode': False
            },
            MarketRegime.HIGH_VOLATILITY: {
                'description': '⚡ Высокая волатильность - уменьшенные позиции',
                'confidence_threshold': 8.0,
                'position_size_multiplier': 0.7,
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 1.5,
                'max_positions': 2,
                'trade_on_pullbacks': False,
                'aggressive_mode': False
            },
            MarketRegime.CRASH: {
                'description': '🚨 ОБВАЛ - только выход из позиций!',
                'confidence_threshold': 9.5,
                'position_size_multiplier': 0.0,  # No new positions
                'stop_loss_multiplier': 2.0,
                'take_profit_multiplier': 0.5,
                'max_positions': 0,
                'trade_on_pullbacks': False,
                'aggressive_mode': False
            },
            MarketRegime.UNKNOWN: {
                'description': '❓ Режим неизвестен - консервативная торговля',
                'confidence_threshold': 8.0,
                'position_size_multiplier': 0.8,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.0,
                'max_positions': 2,
                'trade_on_pullbacks': False,
                'aggressive_mode': False
            }
        }
        
        return strategies.get(regime, strategies[MarketRegime.UNKNOWN])
    
    def should_trade_in_regime(self, regime: MarketRegime) -> bool:
        """
        Определить, стоит ли торговать в текущем режиме
        
        Args:
            regime: MarketRegime
            
        Returns:
            True if trading is recommended
        """
        # Don't trade during crash
        if regime == MarketRegime.CRASH:
            return False
        
        # Trade in all other regimes
        return True
    
    def get_status(self) -> Dict:
        """Получить статус детектора режимов"""
        return {
            'is_fitted': self.is_fitted,
            'n_regimes': self.n_regimes,
            'current_regime': self.current_regime.value,
            'regime_probability': self.regime_probability,
            'regime_mapping': {k: v.value for k, v in self.regime_mapping.items()},
            'history_size': len(self.regime_history)
        }


class MarketRegimeManager:
    """
    Менеджер рыночных режимов с интеграцией в торговую систему
    """
    
    def __init__(self, db_path: str = "trading_history.db"):
        self.db_path = db_path
        self.detector = RegimeDetector()
        self.current_regime = MarketRegime.UNKNOWN
        self.last_detection_time = None
        
        # Create regime_history table
        self._init_database()
        
        logger.info("📊 MarketRegimeManager initialized")
    
    def _init_database(self):
        """Создать таблицу для хранения истории режимов"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS regime_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    regime TEXT NOT NULL,
                    probability REAL,
                    state INTEGER,
                    symbol TEXT DEFAULT 'BTC/USDT'
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Regime history table created")
        
        except Exception as e:
            logger.error(f"Error creating regime_history table: {e}")
    
    def fit_model(self, exchange, symbol: str = "BTC/USDT") -> bool:
        """Обучить HMM модель"""
        return self.detector.fit(exchange, symbol)
    
    def detect_regime(self, exchange, symbol: str = "BTC/USDT", save_to_db: bool = True) -> MarketRegime:
        """
        Определить текущий режим
        
        Args:
            exchange: CCXT exchange
            symbol: Trading pair
            save_to_db: Save to database
            
        Returns:
            MarketRegime
        """
        regime, probability = self.detector.detect_current_regime(exchange, symbol)
        
        self.current_regime = regime
        self.last_detection_time = datetime.now()
        
        # Save to database
        if save_to_db and regime != MarketRegime.UNKNOWN:
            self._save_to_database(regime, probability, symbol)
        
        return regime
    
    def _save_to_database(self, regime: MarketRegime, probability: float, symbol: str):
        """Сохранить обнаруженный режим в БД"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO regime_history (regime, probability, symbol)
                VALUES (?, ?, ?)
            ''', (regime.value, probability, symbol))
            
            conn.commit()
            conn.close()
        
        except Exception as e:
            logger.error(f"Error saving regime to database: {e}")
    
    def get_regime_from_db(self, days: int = 7) -> pd.DataFrame:
        """Получить историю режимов из БД"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"""
                SELECT * FROM regime_history 
                WHERE timestamp >= datetime('now', '-{days} days')
                ORDER BY timestamp DESC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        
        except Exception as e:
            logger.error(f"Error fetching regime history: {e}")
            return pd.DataFrame()
    
    def get_current_strategy(self) -> Dict:
        """Получить стратегию для текущего режима"""
        return self.detector.get_trading_strategy_for_regime(self.current_regime)
    
    def should_trade_now(self) -> bool:
        """Стоит ли торговать в текущем режиме"""
        return self.detector.should_trade_in_regime(self.current_regime)


# Example usage
if __name__ == "__main__":
    import ccxt
    logging.basicConfig(level=logging.INFO)
    
    # Create exchange
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # Create manager
    manager = MarketRegimeManager()
    
    # Fit model
    print("Fitting HMM model...")
    manager.fit_model(exchange, "BTC/USDT")
    
    # Detect current regime
    print("\nDetecting current regime...")
    regime = manager.detect_regime(exchange, "BTC/USDT")
    
    # Get strategy
    strategy = manager.get_current_strategy()
    print(f"\nCurrent regime: {regime.value}")
    print(f"Strategy: {strategy['description']}")
    print(f"Should trade: {manager.should_trade_now()}")
    
    # Get statistics
    stats = manager.detector.get_regime_statistics()
    print(f"\nStatistics: {stats}")
