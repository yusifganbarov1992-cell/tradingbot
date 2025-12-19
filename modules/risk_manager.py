"""
Advanced Risk Manager - Phase 8
Управление рисками с использованием:
1. Kelly Criterion - оптимальный размер позиции
2. VaR (Value at Risk) - максимальный возможный убыток
3. Correlation Matrix - диверсификация портфеля
4. ATR-based Stop-Loss - динамический стоп-лосс
5. Position Sizing based on Volatility
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Уровни риска"""
    VERY_LOW = "VERY_LOW"      # <5% риск
    LOW = "LOW"                # 5-10% риск
    MEDIUM = "MEDIUM"          # 10-20% риск
    HIGH = "HIGH"              # 20-30% риск
    VERY_HIGH = "VERY_HIGH"    # >30% риск


class AdvancedRiskManager:
    """
    Продвинутый Risk Manager
    
    Функции:
    - Kelly Criterion для оптимального sizing
    - VaR calculation (Historical, Parametric)
    - Correlation analysis для портфеля
    - ATR-based dynamic stop-loss
    - Volatility-based position sizing
    - Portfolio risk metrics
    """
    
    def __init__(self, initial_balance: float = 10000, max_risk_per_trade: float = 0.02):
        """
        Args:
            initial_balance: Начальный баланс
            max_risk_per_trade: Максимальный риск на сделку (2% по умолчанию)
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_risk_per_trade = max_risk_per_trade
        
        # Portfolio tracking
        self.positions = {}  # {symbol: {'size': float, 'entry_price': float, 'current_price': float}}
        self.trade_history = []  # List of completed trades
        
        # Risk metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        
        # Kelly Criterion parameters
        self.kelly_fraction = 0.25  # Use 25% of Kelly (fractional Kelly)
        
        # VaR parameters
        self.var_confidence = 0.95  # 95% confidence level
        self.var_horizon = 1  # 1 day
        
        logger.info(f"💼 AdvancedRiskManager initialized (balance=${initial_balance:.2f}, max_risk={max_risk_per_trade:.1%})")
    
    # ========================================
    # KELLY CRITERION
    # ========================================
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Рассчитать Kelly Criterion для оптимального размера позиции
        
        Formula: f* = (p * b - q) / b
        где:
        - f* = оптимальная доля капитала для ставки
        - p = вероятность выигрыша (win rate)
        - q = вероятность проигрыша (1 - p)
        - b = отношение выигрыша к проигрышу (avg_win / avg_loss)
        
        Args:
            win_rate: Процент выигрышных сделок (0-1)
            avg_win: Средний выигрыш
            avg_loss: Средний проигрыш (положительное число)
            
        Returns:
            Оптимальная доля капитала (0-1)
        """
        if avg_loss == 0:
            logger.warning("avg_loss is zero, cannot calculate Kelly")
            return 0
        
        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss
        
        # Kelly formula
        kelly = (p * b - q) / b
        
        # Apply fractional Kelly (more conservative)
        fractional_kelly = kelly * self.kelly_fraction
        
        # Clamp to reasonable range [0, max_risk_per_trade * 2]
        optimal_fraction = max(0, min(fractional_kelly, self.max_risk_per_trade * 2))
        
        logger.info(f"📊 Kelly Criterion: {kelly:.2%} → Fractional: {fractional_kelly:.2%} → Final: {optimal_fraction:.2%}")
        
        return optimal_fraction
    
    def get_kelly_position_size(self, symbol: str, current_price: float) -> float:
        """
        Получить размер позиции на основе Kelly Criterion
        
        Args:
            symbol: Торговая пара
            current_price: Текущая цена
            
        Returns:
            Размер позиции в USD
        """
        # Calculate win rate and avg win/loss from history
        if len(self.trade_history) < 10:
            # Not enough history, use default
            logger.info("Not enough trade history for Kelly, using default 2%")
            return self.current_balance * self.max_risk_per_trade
        
        # Filter trades for this symbol (or use all trades)
        symbol_trades = [t for t in self.trade_history if t.get('symbol') == symbol]
        if len(symbol_trades) < 5:
            # Use all trades if not enough for this symbol
            symbol_trades = self.trade_history
        
        # Calculate metrics
        wins = [t['pnl'] for t in symbol_trades if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in symbol_trades if t['pnl'] < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            logger.info("No wins or losses yet, using default")
            return self.current_balance * self.max_risk_per_trade
        
        win_rate = len(wins) / len(symbol_trades)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        
        # Get Kelly fraction
        kelly_fraction = self.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
        
        # Calculate position size
        position_size_usd = self.current_balance * kelly_fraction
        
        logger.info(f"📊 Kelly position size for {symbol}: ${position_size_usd:.2f} ({kelly_fraction:.1%} of balance)")
        
        return position_size_usd
    
    # ========================================
    # VALUE AT RISK (VAR)
    # ========================================
    
    def calculate_var_historical(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Historical VaR - максимальный убыток с заданной вероятностью
        
        Args:
            returns: Массив исторических returns
            confidence: Уровень доверия (0.95 = 95%)
            
        Returns:
            VaR в процентах (положительное число = убыток)
        """
        if len(returns) == 0:
            return 0
        
        # Sort returns
        sorted_returns = np.sort(returns)
        
        # Find percentile
        percentile_idx = int((1 - confidence) * len(sorted_returns))
        var = -sorted_returns[percentile_idx]  # Negative because we want loss
        
        return var
    
    def calculate_var_parametric(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Parametric VaR - предполагает нормальное распределение
        
        Args:
            returns: Массив исторических returns
            confidence: Уровень доверия (0.95 = 95%)
            
        Returns:
            VaR в процентах
        """
        if len(returns) == 0:
            return 0
        
        # Calculate mean and std
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Z-score for confidence level
        # 95% = 1.645, 99% = 2.326
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence)
        
        # VaR formula
        var = -(mean_return + z_score * std_return)
        
        return var
    
    def calculate_portfolio_var(self, df: pd.DataFrame, confidence: float = 0.95, method: str = 'historical') -> Dict:
        """
        Рассчитать VaR для портфеля
        
        Args:
            df: DataFrame с ценами
            confidence: Уровень доверия
            method: 'historical' или 'parametric'
            
        Returns:
            Dictionary с VaR метриками
        """
        # Calculate returns
        returns = df['close'].pct_change().dropna().values
        
        if len(returns) < 30:
            logger.warning("Not enough data for VaR calculation")
            return {'error': 'Not enough data'}
        
        # Calculate VaR
        if method == 'historical':
            var_1day = self.calculate_var_historical(returns, confidence)
        else:
            var_1day = self.calculate_var_parametric(returns, confidence)
        
        # Scale to different horizons
        var_1week = var_1day * np.sqrt(7)
        var_1month = var_1day * np.sqrt(30)
        
        # Calculate in USD
        current_value = self.current_balance
        var_1day_usd = current_value * var_1day
        var_1week_usd = current_value * var_1week
        var_1month_usd = current_value * var_1month
        
        result = {
            'method': method,
            'confidence': confidence,
            'var_1day_pct': var_1day,
            'var_1week_pct': var_1week,
            'var_1month_pct': var_1month,
            'var_1day_usd': var_1day_usd,
            'var_1week_usd': var_1week_usd,
            'var_1month_usd': var_1month_usd,
            'interpretation': self._interpret_var(var_1day)
        }
        
        logger.info(f"📊 VaR ({method}, {confidence:.0%}): 1d={var_1day:.2%}, 1w={var_1week:.2%}, 1m={var_1month:.2%}")
        
        return result
    
    def _interpret_var(self, var: float) -> str:
        """Интерпретация VaR"""
        if var < 0.02:
            return "Very Low Risk"
        elif var < 0.05:
            return "Low Risk"
        elif var < 0.10:
            return "Medium Risk"
        elif var < 0.20:
            return "High Risk"
        else:
            return "Very High Risk"
    
    # ========================================
    # CORRELATION MATRIX
    # ========================================
    
    def calculate_correlation_matrix(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Рассчитать correlation matrix для портфеля
        
        Args:
            price_data: {symbol: DataFrame с ценами}
            
        Returns:
            Correlation matrix (DataFrame)
        """
        if len(price_data) < 2:
            logger.warning("Need at least 2 assets for correlation")
            return pd.DataFrame()
        
        # Extract returns for each symbol
        returns_dict = {}
        
        for symbol, df in price_data.items():
            returns = df['close'].pct_change().dropna()
            returns_dict[symbol] = returns
        
        # Create DataFrame
        returns_df = pd.DataFrame(returns_dict)
        
        # Calculate correlation
        corr_matrix = returns_df.corr()
        
        logger.info(f"📊 Correlation matrix calculated for {len(price_data)} assets")
        
        return corr_matrix
    
    def check_portfolio_diversification(self, corr_matrix: pd.DataFrame) -> Dict:
        """
        Проверить diversification портфеля
        
        Args:
            corr_matrix: Correlation matrix
            
        Returns:
            Diversification metrics
        """
        if corr_matrix.empty:
            return {'error': 'Empty correlation matrix'}
        
        # Get average correlation (excluding diagonal)
        n = len(corr_matrix)
        total_corr = 0
        count = 0
        
        for i in range(n):
            for j in range(i+1, n):
                total_corr += abs(corr_matrix.iloc[i, j])
                count += 1
        
        avg_correlation = total_corr / count if count > 0 else 0
        
        # Find highly correlated pairs (>0.7)
        high_corr_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append({
                        'pair': (corr_matrix.index[i], corr_matrix.columns[j]),
                        'correlation': corr_val
                    })
        
        # Diversification score (lower correlation = better)
        if avg_correlation < 0.3:
            diversification = "Excellent"
        elif avg_correlation < 0.5:
            diversification = "Good"
        elif avg_correlation < 0.7:
            diversification = "Fair"
        else:
            diversification = "Poor"
        
        result = {
            'avg_correlation': avg_correlation,
            'diversification_score': diversification,
            'num_assets': n,
            'high_correlation_pairs': high_corr_pairs,
            'recommendation': self._get_diversification_recommendation(avg_correlation)
        }
        
        logger.info(f"📊 Portfolio diversification: {diversification} (avg corr={avg_correlation:.2f})")
        
        return result
    
    def _get_diversification_recommendation(self, avg_corr: float) -> str:
        """Рекомендации по diversification"""
        if avg_corr < 0.3:
            return "Portfolio is well diversified. Good job!"
        elif avg_corr < 0.5:
            return "Portfolio has decent diversification. Consider adding uncorrelated assets."
        elif avg_corr < 0.7:
            return "⚠️ Portfolio is moderately correlated. Diversify more to reduce risk."
        else:
            return "🚨 Portfolio is highly correlated! Assets will move together. High risk!"
    
    # ========================================
    # ATR-BASED STOP-LOSS
    # ========================================
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR)
        
        Args:
            df: DataFrame с OHLC данными
            period: Период для ATR (14 по умолчанию)
            
        Returns:
            ATR value
        """
        if len(df) < period:
            logger.warning(f"Not enough data for ATR (need {period}, got {len(df)})")
            return 0
        
        # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # ATR = moving average of TR
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def calculate_atr_stop_loss(self, df: pd.DataFrame, entry_price: float, 
                                side: str = 'long', atr_multiplier: float = 2.0) -> float:
        """
        Рассчитать dynamic stop-loss на основе ATR
        
        Args:
            df: DataFrame с OHLC
            entry_price: Цена входа
            side: 'long' или 'short'
            atr_multiplier: Множитель ATR (2.0 по умолчанию)
            
        Returns:
            Stop-loss price
        """
        atr = self.calculate_atr(df)
        
        if atr == 0:
            # Fallback to fixed % if ATR unavailable
            logger.warning("ATR=0, using fixed 2% stop-loss")
            if side == 'long':
                return entry_price * 0.98
            else:
                return entry_price * 1.02
        
        # Calculate stop-loss
        if side == 'long':
            stop_loss = entry_price - (atr * atr_multiplier)
        else:
            stop_loss = entry_price + (atr * atr_multiplier)
        
        # Calculate SL distance in %
        sl_pct = abs(stop_loss - entry_price) / entry_price
        
        logger.info(f"📊 ATR-based SL: ${stop_loss:.2f} ({sl_pct:.2%} from entry, ATR={atr:.2f})")
        
        return stop_loss
    
    def calculate_atr_take_profit(self, df: pd.DataFrame, entry_price: float,
                                  side: str = 'long', risk_reward_ratio: float = 2.0) -> float:
        """
        Рассчитать take-profit на основе ATR и risk/reward ratio
        
        Args:
            df: DataFrame с OHLC
            entry_price: Цена входа
            side: 'long' или 'short'
            risk_reward_ratio: Отношение риск/награда (2.0 = берем profit в 2x от риска)
            
        Returns:
            Take-profit price
        """
        stop_loss = self.calculate_atr_stop_loss(df, entry_price, side)
        
        # Risk = distance from entry to stop-loss
        risk = abs(entry_price - stop_loss)
        
        # Reward = risk * ratio
        reward = risk * risk_reward_ratio
        
        # Calculate take-profit
        if side == 'long':
            take_profit = entry_price + reward
        else:
            take_profit = entry_price - reward
        
        tp_pct = abs(take_profit - entry_price) / entry_price
        
        logger.info(f"📊 ATR-based TP: ${take_profit:.2f} ({tp_pct:.2%} from entry, R/R={risk_reward_ratio})")
        
        return take_profit
    
    # ========================================
    # VOLATILITY-BASED POSITION SIZING
    # ========================================
    
    def calculate_volatility_adjusted_size(self, df: pd.DataFrame, base_size: float,
                                           target_volatility: float = 0.02) -> float:
        """
        Корректировать размер позиции на основе volatility
        
        Логика: Если volatility высокая, уменьшаем размер позиции
        
        Args:
            df: DataFrame с ценами
            base_size: Базовый размер позиции (USD)
            target_volatility: Целевая volatility (2% по умолчанию)
            
        Returns:
            Adjusted position size
        """
        # Calculate historical volatility (std of returns)
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < 20:
            logger.warning("Not enough data for volatility calculation")
            return base_size
        
        current_volatility = returns.std()
        
        # Adjustment factor = target / current
        # If current vol = 4%, target = 2%, factor = 0.5 (reduce size by half)
        adjustment_factor = target_volatility / current_volatility if current_volatility > 0 else 1.0
        
        # Clamp factor to [0.25, 2.0]
        adjustment_factor = max(0.25, min(adjustment_factor, 2.0))
        
        adjusted_size = base_size * adjustment_factor
        
        logger.info(f"📊 Volatility adjustment: {current_volatility:.2%} → factor={adjustment_factor:.2f} → ${adjusted_size:.2f}")
        
        return adjusted_size
    
    # ========================================
    # PORTFOLIO RISK METRICS
    # ========================================
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Sharpe Ratio = (return - risk_free_rate) / std_dev
        
        Measures risk-adjusted return
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annual, 2% default)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0
        
        # Annualize returns (assuming daily)
        avg_return = np.mean(returns) * 365
        std_return = np.std(returns) * np.sqrt(365)
        
        if std_return == 0:
            return 0
        
        sharpe = (avg_return - risk_free_rate) / std_return
        
        return sharpe
    
    def calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Sortino Ratio - like Sharpe but only considers downside volatility
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0
        
        # Only negative returns (downside)
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')  # No downside = perfect
        
        # Annualize
        avg_return = np.mean(returns) * 365
        downside_std = np.std(negative_returns) * np.sqrt(365)
        
        if downside_std == 0:
            return float('inf')
        
        sortino = (avg_return - risk_free_rate) / downside_std
        
        return sortino
    
    def calculate_max_drawdown(self, equity_curve: np.ndarray) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown from peak
        
        Args:
            equity_curve: Array of portfolio values over time
            
        Returns:
            (max_drawdown_pct, start_idx, end_idx)
        """
        if len(equity_curve) == 0:
            return 0, 0, 0
        
        # Calculate cumulative max
        cumulative_max = np.maximum.accumulate(equity_curve)
        
        # Calculate drawdown
        drawdown = (equity_curve - cumulative_max) / cumulative_max
        
        # Find max drawdown
        max_dd_idx = np.argmin(drawdown)
        max_dd = drawdown[max_dd_idx]
        
        # Find peak before max drawdown
        peak_idx = np.argmax(cumulative_max[:max_dd_idx+1])
        
        return abs(max_dd), peak_idx, max_dd_idx
    
    def get_portfolio_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Получить все метрики риска портфеля
        
        Args:
            df: DataFrame с ценами
            
        Returns:
            Dictionary с метриками
        """
        returns = df['close'].pct_change().dropna().values
        equity_curve = (1 + pd.Series(returns)).cumprod().values * self.initial_balance
        
        # Calculate metrics
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        max_dd, dd_start, dd_end = self.calculate_max_drawdown(equity_curve)
        
        # VaR
        var_metrics = self.calculate_portfolio_var(df)
        
        # Volatility
        volatility_daily = np.std(returns)
        volatility_annual = volatility_daily * np.sqrt(365)
        
        # Total return
        total_return = (equity_curve[-1] - self.initial_balance) / self.initial_balance
        
        metrics = {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'volatility_annual': volatility_annual,
            'total_return': total_return,
            'var_1day': var_metrics.get('var_1day_pct', 0),
            'var_1day_usd': var_metrics.get('var_1day_usd', 0),
            'current_balance': self.current_balance,
            'risk_level': self._assess_risk_level(max_dd, volatility_annual, var_metrics.get('var_1day_pct', 0))
        }
        
        logger.info(f"📊 Portfolio metrics: Sharpe={sharpe:.2f}, Sortino={sortino:.2f}, MaxDD={max_dd:.2%}")
        
        return metrics
    
    def _assess_risk_level(self, max_dd: float, volatility: float, var: float) -> str:
        """Оценить общий уровень риска"""
        # Weighted score
        dd_score = max_dd * 0.4
        vol_score = volatility * 0.3
        var_score = var * 0.3
        
        total_score = dd_score + vol_score + var_score
        
        if total_score < 0.05:
            return RiskLevel.VERY_LOW.value
        elif total_score < 0.10:
            return RiskLevel.LOW.value
        elif total_score < 0.20:
            return RiskLevel.MEDIUM.value
        elif total_score < 0.30:
            return RiskLevel.HIGH.value
        else:
            return RiskLevel.VERY_HIGH.value
    
    def get_status(self) -> Dict:
        """Получить статус risk manager"""
        return {
            'current_balance': self.current_balance,
            'initial_balance': self.initial_balance,
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            'max_risk_per_trade': self.max_risk_per_trade,
            'kelly_fraction': self.kelly_fraction,
            'num_positions': len(self.positions)
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Advanced Risk Manager - Phase 8")
    print("=" * 70)
    
    # Create risk manager
    rm = AdvancedRiskManager(initial_balance=10000, max_risk_per_trade=0.02)
    
    print(f"\n✅ Risk Manager created")
    print(f"   Balance: ${rm.current_balance:.2f}")
    print(f"   Max risk per trade: {rm.max_risk_per_trade:.1%}")
    print(f"   Kelly fraction: {rm.kelly_fraction:.1%}")
