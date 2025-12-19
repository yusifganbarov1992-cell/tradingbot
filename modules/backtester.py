"""
Backtester Module - Test strategy on historical data
FREE - uses only Binance historical data
"""

import os
import sys
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Single backtest trade"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    amount: float
    pnl: float
    pnl_percent: float
    reason: str


@dataclass
class BacktestResult:
    """Backtest results summary"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_percent: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_duration: float
    best_trade: float
    worst_trade: float
    trades: List[BacktestTrade]


class Backtester:
    """
    Strategy Backtester
    Tests trading strategy on historical data
    """
    
    def __init__(self, initial_balance: float = 1000.0):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = None
        self.trades: List[BacktestTrade] = []
        
        # Strategy parameters
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.sma_fast = 10
        self.sma_slow = 20
        self.stop_loss_pct = 2.0
        self.take_profit_pct = 4.0
        self.position_size_pct = 10.0
    
    def fetch_historical_data(self, symbol: str, timeframe: str = '1h', 
                              days: int = 30) -> pd.DataFrame:
        """Fetch historical OHLCV data"""
        try:
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # SMAs
        df['sma_fast'] = df['close'].rolling(window=self.sma_fast).mean()
        df['sma_slow'] = df['close'].rolling(window=self.sma_slow).mean()
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        
        # Volume SMA
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        return df
    
    def generate_signal(self, row: pd.Series, prev_row: pd.Series) -> Tuple[str, str]:
        """Generate trading signal based on indicators"""
        signal = None
        reason = ""
        
        # Skip if indicators not ready
        if pd.isna(row['rsi']) or pd.isna(row['sma_fast']):
            return None, ""
        
        # === BUY CONDITIONS ===
        buy_conditions = []
        
        # RSI oversold
        if row['rsi'] < self.rsi_oversold:
            buy_conditions.append(f"RSI oversold ({row['rsi']:.0f})")
        
        # SMA crossover (fast crosses above slow)
        if prev_row['sma_fast'] < prev_row['sma_slow'] and row['sma_fast'] > row['sma_slow']:
            buy_conditions.append("SMA bullish crossover")
        
        # MACD crossover
        if prev_row['macd'] < prev_row['macd_signal'] and row['macd'] > row['macd_signal']:
            buy_conditions.append("MACD bullish crossover")
        
        # Price at lower Bollinger Band
        if row['close'] < row['bb_lower']:
            buy_conditions.append("Price below BB lower")
        
        # Volume spike
        if row['volume'] > row['volume_sma'] * 1.5:
            buy_conditions.append("High volume")
        
        # === SELL CONDITIONS ===
        sell_conditions = []
        
        # RSI overbought
        if row['rsi'] > self.rsi_overbought:
            sell_conditions.append(f"RSI overbought ({row['rsi']:.0f})")
        
        # SMA crossover (fast crosses below slow)
        if prev_row['sma_fast'] > prev_row['sma_slow'] and row['sma_fast'] < row['sma_slow']:
            sell_conditions.append("SMA bearish crossover")
        
        # MACD crossover
        if prev_row['macd'] > prev_row['macd_signal'] and row['macd'] < row['macd_signal']:
            sell_conditions.append("MACD bearish crossover")
        
        # Price at upper Bollinger Band
        if row['close'] > row['bb_upper']:
            sell_conditions.append("Price above BB upper")
        
        # Generate signal
        if len(buy_conditions) >= 2:
            signal = 'BUY'
            reason = " + ".join(buy_conditions)
        elif len(sell_conditions) >= 2:
            signal = 'SELL'
            reason = " + ".join(sell_conditions)
        
        return signal, reason
    
    def run_backtest(self, symbol: str = 'BTC/USDT', timeframe: str = '1h',
                     days: int = 30) -> BacktestResult:
        """Run backtest on historical data"""
        print(f"\n{'='*60}")
        print(f"BACKTESTING: {symbol} | {timeframe} | {days} days")
        print(f"{'='*60}")
        
        # Fetch data
        df = self.fetch_historical_data(symbol, timeframe, days)
        if df.empty:
            raise ValueError("No data fetched")
        
        print(f"Loaded {len(df)} candles")
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Reset state
        self.balance = self.initial_balance
        self.position = None
        self.trades = []
        balance_history = [self.initial_balance]
        
        # Iterate through data
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            current_time = df.index[i]
            current_price = row['close']
            
            # Check position stop loss / take profit
            if self.position:
                if self.position['side'] == 'BUY':
                    pnl_pct = (current_price - self.position['entry_price']) / self.position['entry_price'] * 100
                else:
                    pnl_pct = (self.position['entry_price'] - current_price) / self.position['entry_price'] * 100
                
                # Stop loss
                if pnl_pct <= -self.stop_loss_pct:
                    self._close_position(current_time, current_price, "Stop loss hit")
                # Take profit
                elif pnl_pct >= self.take_profit_pct:
                    self._close_position(current_time, current_price, "Take profit hit")
            
            # Generate signal
            signal, reason = self.generate_signal(row, prev_row)
            
            # Execute signal
            if signal and not self.position:
                self._open_position(current_time, current_price, signal, symbol, reason)
            elif signal and self.position and signal != self.position['side']:
                self._close_position(current_time, current_price, f"Signal reversed: {reason}")
                self._open_position(current_time, current_price, signal, symbol, reason)
            
            balance_history.append(self.balance + (self._position_value(current_price) if self.position else 0))
        
        # Close any remaining position
        if self.position:
            self._close_position(df.index[-1], df.iloc[-1]['close'], "End of backtest")
        
        # Calculate results
        return self._calculate_results(balance_history)
    
    def _open_position(self, time: datetime, price: float, side: str, 
                       symbol: str, reason: str):
        """Open a new position"""
        position_value = self.balance * (self.position_size_pct / 100)
        amount = position_value / price
        
        self.position = {
            'entry_time': time,
            'entry_price': price,
            'side': side,
            'symbol': symbol,
            'amount': amount,
            'reason': reason,
        }
        
        self.balance -= position_value
    
    def _close_position(self, time: datetime, price: float, reason: str):
        """Close current position"""
        if not self.position:
            return
        
        if self.position['side'] == 'BUY':
            pnl = (price - self.position['entry_price']) * self.position['amount']
            pnl_pct = (price - self.position['entry_price']) / self.position['entry_price'] * 100
        else:
            pnl = (self.position['entry_price'] - price) * self.position['amount']
            pnl_pct = (self.position['entry_price'] - price) / self.position['entry_price'] * 100
        
        position_value = self.position['amount'] * price
        self.balance += position_value + pnl
        
        trade = BacktestTrade(
            entry_time=self.position['entry_time'],
            exit_time=time,
            symbol=self.position['symbol'],
            side=self.position['side'],
            entry_price=self.position['entry_price'],
            exit_price=price,
            amount=self.position['amount'],
            pnl=pnl,
            pnl_percent=pnl_pct,
            reason=f"{self.position['reason']} → {reason}"
        )
        self.trades.append(trade)
        self.position = None
    
    def _position_value(self, current_price: float) -> float:
        """Calculate current position value"""
        if not self.position:
            return 0
        return self.position['amount'] * current_price
    
    def _calculate_results(self, balance_history: List[float]) -> BacktestResult:
        """Calculate backtest metrics"""
        if not self.trades:
            return BacktestResult(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_pnl=0, total_pnl_percent=0,
                max_drawdown=0, sharpe_ratio=0, profit_factor=0,
                avg_trade_duration=0, best_trade=0, worst_trade=0,
                trades=[]
            )
        
        # Basic stats
        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_pnl_percent = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        # Win rate
        win_rate = len(winning) / len(self.trades) * 100 if self.trades else 0
        
        # Max drawdown
        peak = balance_history[0]
        max_dd = 0
        for balance in balance_history:
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        # Sharpe ratio (simplified)
        returns = pd.Series(balance_history).pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade duration
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in self.trades]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Best/worst trades
        pnls = [t.pnl_percent for t in self.trades]
        best_trade = max(pnls) if pnls else 0
        worst_trade = min(pnls) if pnls else 0
        
        return BacktestResult(
            total_trades=len(self.trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_trade_duration=avg_duration,
            best_trade=best_trade,
            worst_trade=worst_trade,
            trades=self.trades
        )
    
    def print_results(self, result: BacktestResult):
        """Print backtest results"""
        print(f"\n{'='*60}")
        print("BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance:   ${self.balance:,.2f}")
        print(f"{'='*60}")
        print(f"Total Trades:    {result.total_trades}")
        print(f"Winning Trades:  {result.winning_trades}")
        print(f"Losing Trades:   {result.losing_trades}")
        print(f"Win Rate:        {result.win_rate:.1f}%")
        print(f"{'='*60}")
        print(f"Total P&L:       ${result.total_pnl:,.2f} ({result.total_pnl_percent:+.2f}%)")
        print(f"Max Drawdown:    {result.max_drawdown:.2f}%")
        print(f"Sharpe Ratio:    {result.sharpe_ratio:.2f}")
        print(f"Profit Factor:   {result.profit_factor:.2f}")
        print(f"{'='*60}")
        print(f"Best Trade:      {result.best_trade:+.2f}%")
        print(f"Worst Trade:     {result.worst_trade:+.2f}%")
        print(f"Avg Duration:    {result.avg_trade_duration:.1f} hours")
        print(f"{'='*60}")
        
        # Rating
        score = 0
        if result.win_rate > 50: score += 2
        if result.total_pnl_percent > 0: score += 2
        if result.max_drawdown < 10: score += 2
        if result.sharpe_ratio > 1: score += 2
        if result.profit_factor > 1.5: score += 2
        
        rating = "⭐" * score
        print(f"\nSTRATEGY RATING: {rating} ({score}/10)")
        
        if score >= 8:
            print("✅ EXCELLENT - Ready for paper trading")
        elif score >= 6:
            print("🟡 GOOD - Consider optimizing parameters")
        elif score >= 4:
            print("🟠 FAIR - Needs improvement")
        else:
            print("🔴 POOR - Do not use in production")
        
        # Show recent trades
        print(f"\n{'='*60}")
        print("RECENT TRADES:")
        print(f"{'='*60}")
        for trade in result.trades[-5:]:
            emoji = "✅" if trade.pnl > 0 else "❌"
            print(f"{emoji} {trade.side} {trade.symbol}")
            print(f"   Entry: ${trade.entry_price:.2f} → Exit: ${trade.exit_price:.2f}")
            print(f"   P&L: ${trade.pnl:.2f} ({trade.pnl_percent:+.2f}%)")
            print(f"   Reason: {trade.reason[:60]}...")
            print()


def main():
    """Run backtest"""
    from dotenv import load_dotenv
    load_dotenv()
    
    bt = Backtester(initial_balance=1000)
    
    # Test on multiple symbols
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    for symbol in symbols:
        try:
            result = bt.run_backtest(symbol=symbol, timeframe='1h', days=30)
            bt.print_results(result)
        except Exception as e:
            print(f"Error backtesting {symbol}: {e}")
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
