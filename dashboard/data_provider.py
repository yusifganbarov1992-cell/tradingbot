"""
Dashboard Data Provider - SYNC version for Streamlit
Provides live data for dashboard without async complications
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import trading bot modules
try:
    from database_supabase import SupabaseDatabase
    import ccxt
except ImportError as e:
    logging.warning(f"Import warning: {e}")

logger = logging.getLogger(__name__)


class DashboardDataProvider:
    """Provides real data for dashboard from trading bot"""
    
    _instance = None  # Singleton pattern
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize data provider"""
        if hasattr(self, 'initialized'):
            return
        
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        try:
            # Database
            self.db = SupabaseDatabase()
            
            # Exchange (sync version for Streamlit) - WITH API KEYS for balance
            self.exchange = ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_SECRET_KEY'),
                'enableRateLimit': True,
            })
            
            self.initialized = True
            logger.info("✅ Dashboard data provider initialized (sync mode)")
            
        except Exception as e:
            logger.error(f"Failed to initialize data provider: {e}")
            self.db = None
            self.exchange = None
    
    # ============================================
    # PORTFOLIO DATA
    # ============================================
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary metrics"""
        try:
            if not self.db:
                return self._get_mock_portfolio()
            
            # Get all trades from database
            trades = self.db.get_trade_history(limit=1000)
            
            if not trades:
                return self._get_mock_portfolio()
            
            # Calculate metrics
            total_trades = len(trades)
            closed_trades = [t for t in trades if t.get('status') == 'closed']
            
            # Calculate win rate even with few trades
            winning_trades = [t for t in closed_trades if (t.get('pnl') or 0) > 0]
            losing_trades = [t for t in closed_trades if (t.get('pnl') or 0) < 0]
            
            win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
            
            total_pnl = sum((t.get('pnl') or 0) for t in closed_trades)
            
            # Get REAL balance from Binance
            current_balance = 10000  # Default
            try:
                if self.exchange:
                    balance = self.exchange.fetch_balance()
                    # Calculate total USD value
                    total_usd = 0
                    for currency, amounts in balance['total'].items():
                        if amounts > 0:
                            if currency in ['USDT', 'USDC', 'BUSD']:
                                total_usd += amounts
                            elif currency.startswith('LD'):
                                # Binance Earn tokens
                                base = currency[2:]
                                if base in ['USDT', 'USDC']:
                                    total_usd += amounts
                                else:
                                    try:
                                        ticker = self.exchange.fetch_ticker(f'{base}/USDT')
                                        total_usd += amounts * ticker['last']
                                    except:
                                        pass
                            else:
                                try:
                                    ticker = self.exchange.fetch_ticker(f'{currency}/USDT')
                                    total_usd += amounts * ticker['last']
                                except:
                                    pass
                    current_balance = total_usd if total_usd > 0 else 10000
            except Exception as e:
                logger.warning(f"Could not fetch real balance: {e}")
            
            # Sharpe ratio calculation
            returns = [(t.get('pnl_percent') or 0) / 100 for t in closed_trades]
            sharpe = self._calculate_sharpe(returns) if closed_trades else 0
            
            # Max drawdown
            initial_balance = 10000
            equity_curve = self._build_equity_curve(closed_trades, initial_balance) if closed_trades else [initial_balance]
            max_dd = self._calculate_max_drawdown(equity_curve) if len(equity_curve) > 1 else 0
            
            return {
                'balance': current_balance,
                'balance_change': total_pnl,
                'balance_change_pct': (total_pnl / initial_balance) * 100 if initial_balance else 0,
                'total_trades': total_trades,
                'trades_today': self._count_trades_today(trades),
                'win_rate': win_rate * 100,
                'win_rate_change': 0.0,  # TODO: calculate from historical data
                'sharpe_ratio': sharpe,
                'sharpe_change': 0.0,  # TODO: calculate from historical data
                'max_drawdown': abs(max_dd),
                'max_dd_change': 0.0  # TODO: calculate from historical data
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return self._get_mock_portfolio()
    
    def get_equity_curve(self, days: int = 45) -> pd.DataFrame:
        """Get equity curve data"""
        try:
            if not self.db:
                return self._get_mock_equity_curve(days)
            
            # Get trades from database
            trades = self.db.get_trade_history(limit=1000)
            closed_trades = [t for t in trades if t.get('status') == 'closed']
            
            if not closed_trades:
                return self._get_mock_equity_curve(days)
            
            # Build equity curve
            initial_balance = 10000
            equity_curve = self._build_equity_curve(closed_trades, initial_balance)
            
            # Create DataFrame
            dates = pd.date_range(
                end=datetime.now(),
                periods=len(equity_curve),
                freq='D'
            )
            
            df = pd.DataFrame({
                'date': dates[-days:],
                'equity': equity_curve[-days:]
            })
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting equity curve: {e}")
            return self._get_mock_equity_curve(days)
    
    def get_pnl_distribution(self) -> List[float]:
        """Get PnL distribution for histogram"""
        try:
            if not self.db:
                return self._get_mock_pnl_distribution()
            
            trades = self.db.get_trade_history(limit=100)
            closed_trades = [t for t in trades if t.get('status') == 'closed']
            
            if not closed_trades:
                return self._get_mock_pnl_distribution()
            
            pnl_values = [t.get('pnl', 0) for t in closed_trades]
            return pnl_values
            
        except Exception as e:
            logger.error(f"Error getting PnL distribution: {e}")
            return self._get_mock_pnl_distribution()
    
    def get_recent_activity(self) -> List[Dict]:
        """Get recent activity events"""
        try:
            if not self.db:
                return self._get_mock_activity()
            
            trades = self.db.get_trade_history(limit=10)
            
            activity = []
            for trade in trades:
                time_str = self._format_time(trade.get('entry_time'))
                status = trade.get('status', 'unknown')
                pnl = trade.get('pnl', 0)
                
                activity.append({
                    'timestamp': time_str,
                    'type': 'trade',
                    'action': 'CLOSED' if status == 'closed' else 'OPENED',
                    'symbol': trade.get('symbol', 'N/A'),
                    'price': trade.get('entry_price', 0),
                    'pnl': pnl,
                    'message': f"{'Closed with profit' if pnl > 0 else 'Closed with loss'}" if status == 'closed' else 'Trade opened'
                })
            
            return activity if activity else self._get_mock_activity()
            
        except Exception as e:
            logger.error(f"Error getting recent activity: {e}")
            return self._get_mock_activity()
    
    def get_ai_prediction(self, symbol: str = 'BTC/USDT') -> Dict:
        """Get AI prediction for symbol - mock for now"""
        # TODO: Integrate with actual AI model
        return {
            'predicted_price': 87450,
            'confidence': 78,
            'final_signal': 'STRONG BUY',
            'lstm_signal': 'BUY',
            'lstm_confidence': 78,
            'pattern_signal': 'BUY',
            'pattern_confidence': 85,
            'technical_signal': 'BUY',
            'technical_confidence': 65,
            'patterns': [
                {'Pattern': 'Double Bottom', 'Signal': 'BUY', 'Confidence': 85},
                {'Pattern': 'Ascending Triangle', 'Signal': 'BUY', 'Confidence': 72},
                {'Pattern': 'Support Breakout', 'Signal': 'HOLD', 'Confidence': 55},
                {'Pattern': 'MACD Cross', 'Signal': 'BUY', 'Confidence': 68}
            ]
        }
    
    def get_price_prediction_chart(self, symbol: str = 'BTC/USDT') -> Dict:
        """Get price prediction data for chart"""
        # Mock data for now
        historical_prices = 85000 + np.cumsum(np.random.randn(100) * 500)
        future_prices = historical_prices[-1] + np.cumsum(np.random.randn(20) * 300)
        
        return {
            'historical_x': list(range(100)),
            'historical_y': historical_prices.tolist(),
            'prediction_x': list(range(99, 120)),
            'prediction_y': np.concatenate([[historical_prices[-1]], future_prices]).tolist()
        }
    
    def get_risk_metrics(self, symbol: str = 'BTC/USDT') -> Dict:
        """Get comprehensive risk metrics"""
        return {
            'var': {
                '1d_95': 0.84,
                '1w_95': 2.22,
                '1m_95': 4.59,
                '1d_95_param': 0.89,
                '1w_95_param': 2.35,
                '1m_95_param': 4.86,
                '1d_99': 1.25,
                '1w_99': 3.31,
                '1m_99': 6.85
            },
            'kelly_size': 400,
            'kelly_pct': 4.0,
            'sortino_ratio': 2.31,
            'recovery_factor': 1.79,
            'volatility': 12.4,
            'calmar_ratio': 2.18
        }
    
    def get_correlation_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """Get correlation matrix for symbols"""
        # Mock data for now
        corr_matrix = np.array([
            [1.00, 0.90, 0.87, 0.82],
            [0.90, 1.00, 0.86, 0.79],
            [0.87, 0.86, 1.00, 0.75],
            [0.82, 0.79, 0.75, 1.00]
        ])
        
        return pd.DataFrame(corr_matrix, columns=symbols, index=symbols)
    
    def get_trade_history(self, filters: Dict = None) -> pd.DataFrame:
        """Get filtered trade history"""
        try:
            if not self.db:
                return self._get_mock_trades()
            
            trades = self.db.get_trade_history(limit=100)
            
            if not trades:
                return self._get_mock_trades()
            
            df = pd.DataFrame(trades)
            
            # Apply filters
            if filters:
                if filters.get('status'):
                    df = df[df['status'].isin(filters['status'])]
                if filters.get('symbol'):
                    df = df[df['symbol'].isin(filters['symbol'])]
                if filters.get('side'):
                    df = df[df['side'].isin(filters['side'])]
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return self._get_mock_trades()
    
    def get_current_price(self, symbol: str = 'BTC/USDT') -> float:
        """Get current price for symbol"""
        try:
            if not self.exchange:
                return 87226.0
            
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 87226.0
    
    def get_sentiment_data(self) -> Dict:
        """Get sentiment analysis data"""
        return {
            'fear_greed_index': 65,
            'sentiment': 'Neutral',
            'trend': 'Improving'
        }
    
    # ============================================
    # HELPER METHODS
    # ============================================
    
    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe
        sharpe = (mean_return / std_return) * np.sqrt(252)
        return sharpe
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (value - peak) / peak
            if dd < max_dd:
                max_dd = dd
        
        return max_dd * 100  # Return as percentage
    
    def _build_equity_curve(self, trades: List[Dict], initial_balance: float) -> List[float]:
        """Build equity curve from trades"""
        equity = [initial_balance]
        current_balance = initial_balance
        
        for trade in sorted(trades, key=lambda x: x.get('entry_time', '')):
            pnl = trade.get('pnl', 0)
            current_balance += pnl
            equity.append(current_balance)
        
        return equity
    
    def _count_trades_today(self, trades: List[Dict]) -> int:
        """Count trades executed today"""
        today = datetime.now().date()
        count = 0
        
        for trade in trades:
            entry_time_str = trade.get('entry_time', '')
            if entry_time_str:
                try:
                    entry_date = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00')).date()
                    if entry_date == today:
                        count += 1
                except:
                    pass
        
        return count
    
    def _format_time(self, timestamp_str: str) -> str:
        """Format timestamp to readable string"""
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt.strftime('%H:%M')
        except:
            return 'N/A'
    
    # ============================================
    # MOCK DATA METHODS
    # ============================================
    
    def _get_mock_portfolio(self) -> Dict:
        """Mock portfolio data"""
        return {
            'balance': 10150.00,
            'balance_change': 150.00,
            'balance_change_pct': 1.5,
            'total_trades': 45,
            'trades_today': 3,
            'win_rate': 62.2,
            'win_rate_change': 1.2,
            'sharpe_ratio': 1.85,
            'sharpe_change': 0.15,
            'max_drawdown': 8.5,
            'max_dd_change': -0.5
        }
    
    def _get_mock_equity_curve(self, days: int) -> pd.DataFrame:
        """Mock equity curve"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        equity = 10000 * (1 + np.cumsum(np.random.randn(days) * 0.02))
        
        return pd.DataFrame({'date': dates, 'equity': equity})
    
    def _get_mock_pnl_distribution(self) -> List[float]:
        """Mock PnL distribution"""
        return list(np.random.normal(50, 100, 45))
    
    def _get_mock_activity(self) -> List[Dict]:
        """Mock recent activity"""
        return [
            {'timestamp': '10:30', 'type': 'trade', 'action': 'CLOSED', 'symbol': 'BTC/USDT', 'price': 85000, 'pnl': 85, 'message': 'Trade closed with profit'},
            {'timestamp': '09:45', 'type': 'signal', 'action': 'BUY', 'symbol': 'ETH/USDT', 'price': 3200, 'pnl': 0, 'message': 'Buy signal generated'},
            {'timestamp': '08:15', 'type': 'trade', 'action': 'OPENED', 'symbol': 'BNB/USDT', 'price': 610, 'pnl': 0, 'message': 'Trade opened'},
            {'timestamp': '07:30', 'type': 'alert', 'action': 'RISK', 'symbol': 'SOL/USDT', 'price': 210, 'pnl': 0, 'message': 'Risk alert: High volatility'},
            {'timestamp': '06:00', 'type': 'system', 'action': 'SCAN', 'symbol': 'Multiple', 'price': 0, 'pnl': 0, 'message': 'Market scan completed'}
        ]
    
    def _get_mock_trades(self) -> pd.DataFrame:
        """Mock trade history"""
        return pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(hours=i*3) for i in range(20)],
            'symbol': np.random.choice(['BTC/USDT', 'ETH/USDT'], 20),
            'side': np.random.choice(['LONG', 'SHORT'], 20),
            'entry_price': np.random.uniform(85000, 88000, 20),
            'exit_price': np.random.uniform(85000, 88000, 20),
            'amount': np.random.uniform(100, 1000, 20),
            'pnl': np.random.uniform(-100, 200, 20),
            'pnl_pct': np.random.uniform(-2, 4, 20),
            'status': ['closed'] * 20
        })


# Singleton accessor
_provider_instance = None

def get_data_provider() -> DashboardDataProvider:
    """Get singleton instance of data provider"""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = DashboardDataProvider()
    return _provider_instance
