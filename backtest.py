"""
Backtesting Engine для NexusTrader AI
Тестирует стратегию на исторических данных
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Движок для backtesting торговой стратегии
    """
    
    def __init__(self, exchange: ccxt.Exchange = None, initial_balance: float = 1000.0,
                 position_size: float = 0.10, sl_multiplier: float = 2.0,
                 tp_multiplier: float = 3.0, ai_threshold: int = 2, 
                 confidence_threshold: int = 7):
        self.exchange = exchange
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []  # Открытые позиции
        self.closed_trades = []  # Закрытые сделки
        self.trades_log = []  # Подробный лог
        
        # Параметры стратегии
        self.position_size = position_size  # % от баланса
        self.sl_multiplier = sl_multiplier  # Stop Loss в ATR
        self.tp_multiplier = tp_multiplier  # Take Profit в ATR
        self.ai_threshold = ai_threshold  # Минимум фильтров для AI
        self.confidence_threshold = confidence_threshold  # Минимальная уверенность AI
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Рассчитать технические индикаторы (как в основном боте)"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # EMA
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Volume average
        df['volume_avg'] = df['volume'].rolling(20).mean()
        
        return df
    
    def apply_filters(self, row: pd.Series) -> Tuple[int, int]:
        """Применить 8 фильтров как в основном боте"""
        buy_filters = 0
        sell_filters = 0
        
        # Filter 1: RSI
        if pd.notna(row['rsi']):
            if row['rsi'] < 30:
                buy_filters += 1
            elif row['rsi'] > 70:
                sell_filters += 1
        
        # Filter 2: EMA crossover
        if pd.notna(row['ema20']) and pd.notna(row['ema50']):
            if row['ema20'] > row['ema50']:
                buy_filters += 1
            else:
                sell_filters += 1
        
        # Filter 3: MACD
        if pd.notna(row['macd']):
            if row['macd'] > 0:
                buy_filters += 1
            else:
                sell_filters += 1
        
        # Filter 4: Volume
        if pd.notna(row['volume_avg']) and row['volume'] > row['volume_avg'] * 1.5:
            buy_filters += 1
        
        # Filters 5-8 упрощены для backtesting
        # (можно добавить позже для точности)
        
        return buy_filters, sell_filters
    
    def simulate_ai_decision(self, buy_filters: int, sell_filters: int, row: pd.Series) -> Tuple[str, int]:
        """
        Симуляция AI решения на основе фильтров
        Упрощенная версия без реального AI (экономия токенов)
        """
        # Если достаточно фильтров на покупку
        if buy_filters >= self.ai_threshold:
            confidence = min(10, buy_filters + 4)
            # Проверка минимальной уверенности
            if confidence >= self.confidence_threshold:
                return "BUY", confidence
        # Если достаточно фильтров на продажу
        elif sell_filters >= self.ai_threshold:
            confidence = min(10, sell_filters + 4)
            if confidence >= self.confidence_threshold:
                return "SELL", confidence
        
        return "WAIT", 0
    
    def open_position(self, timestamp: datetime, price: float, signal: str, atr: float):
        """Открыть позицию"""
        position_size = self.balance * self.position_size
        
        if signal == "BUY":
            stop_loss = price - (self.sl_multiplier * atr)
            take_profit = price + (self.tp_multiplier * atr)
        else:  # SELL
            stop_loss = price + (self.sl_multiplier * atr)
            take_profit = price - (self.tp_multiplier * atr)
        
        position = {
            'timestamp': timestamp,
            'signal': signal,
            'entry_price': price,
            'size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop': stop_loss,
            'atr': atr
        }
        
        self.positions.append(position)
        logger.debug(f"Opened {signal} at ${price:.2f}, SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}")
    
    def update_position(self, position: dict, current_price: float, timestamp: datetime) -> bool:
        """
        Обновить позицию и проверить условия закрытия
        Returns: True если позиция закрыта
        """
        signal = position['signal']
        
        # Обновить trailing stop
        if signal == "BUY":
            new_stop = current_price - (2 * position['atr'])
            if new_stop > position['trailing_stop']:
                position['trailing_stop'] = new_stop
            
            # Проверить условия закрытия
            if current_price <= position['trailing_stop']:
                self.close_position(position, current_price, timestamp, "Trailing Stop")
                return True
            elif current_price >= position['take_profit']:
                self.close_position(position, current_price, timestamp, "Take Profit")
                return True
        else:  # SELL
            new_stop = current_price + (2 * position['atr'])
            if new_stop < position['trailing_stop']:
                position['trailing_stop'] = new_stop
            
            if current_price >= position['trailing_stop']:
                self.close_position(position, current_price, timestamp, "Trailing Stop")
                return True
            elif current_price <= position['take_profit']:
                self.close_position(position, current_price, timestamp, "Take Profit")
                return True
        
        return False
    
    def close_position(self, position: dict, exit_price: float, timestamp: datetime, reason: str):
        """Закрыть позицию"""
        signal = position['signal']
        entry_price = position['entry_price']
        size = position['size']
        
        # Рассчитать P&L
        if signal == "BUY":
            pnl = ((exit_price - entry_price) / entry_price) * size
        else:  # SELL
            pnl = ((entry_price - exit_price) / entry_price) * size
        
        # Комиссия 0.1%
        fee = size * 0.001
        pnl -= fee
        
        self.balance += pnl
        
        trade = {
            'entry_time': position['timestamp'],
            'exit_time': timestamp,
            'signal': signal,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'reason': reason,
            'duration': (timestamp - position['timestamp']).total_seconds() / 3600  # hours
        }
        
        self.closed_trades.append(trade)
        logger.debug(f"Closed {signal} at ${exit_price:.2f}, P&L: ${pnl:.2f} ({reason})")
    
    def run_backtest(self, data_or_symbol, timeframe: str = '1h', days: int = 30) -> Dict:
        """
        Запустить backtesting
        
        Args:
            data_or_symbol: DataFrame с данными ИЛИ строка с символом (например, "BTC/USDT")
            timeframe: Таймфрейм ('1h', '4h', '1d') - используется только если передан symbol
            days: Количество дней для теста - используется только если передан symbol
        
        Returns:
            Dict с результатами
        """
        # Если передан DataFrame, используем его напрямую
        if isinstance(data_or_symbol, pd.DataFrame):
            df = data_or_symbol.copy()
            logger.info(f"=== Backtesting on provided data ({len(df)} candles) ===")
        else:
            # Иначе загружаем данные с биржи
            symbol = data_or_symbol
            logger.info(f"=== Backtesting {symbol} on {timeframe} for {days} days ===")
            
            # Загрузить исторические данные
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            logger.info(f"Loaded {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        
        # Рассчитать индикаторы
        df = self.calculate_indicators(df)
        
        # Прогнать каждую свечу
        for i in range(50, len(df)):  # Пропустить первые 50 для индикаторов
            row = df.iloc[i]
            timestamp = row['timestamp']
            price = row['close']
            
            # Обновить открытые позиции
            positions_to_remove = []
            for pos in self.positions:
                if self.update_position(pos, price, timestamp):
                    positions_to_remove.append(pos)
            
            for pos in positions_to_remove:
                self.positions.remove(pos)
            
            # Проверить сигналы на открытие новых позиций
            if len(self.positions) < 5:  # MAX 5 позиций
                buy_filters, sell_filters = self.apply_filters(row)
                signal, confidence = self.simulate_ai_decision(buy_filters, sell_filters, row)
                
                if signal in ["BUY", "SELL"] and confidence >= 7:
                    self.open_position(timestamp, price, signal, row['atr'])
        
        # Закрыть все открытые позиции
        final_price = df.iloc[-1]['close']
        final_timestamp = df.iloc[-1]['timestamp']
        for pos in self.positions[:]:
            self.close_position(pos, final_price, final_timestamp, "End of backtest")
        
        # Рассчитать метрики
        results = self.calculate_metrics()
        
        return results
    
    def calculate_metrics(self) -> Dict:
        """Рассчитать метрики производительности"""
        if not self.closed_trades:
            logger.info("=== Backtest Results ===")
            logger.info("No trades executed")
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'roi': 0,
                'final_balance': self.balance,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'avg_duration': 0
            }
        
        trades_df = pd.DataFrame(self.closed_trades)
        
        # Основные метрики
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0] if total_trades > 0 else pd.DataFrame()
        losing_trades = trades_df[trades_df['pnl'] < 0] if total_trades > 0 else pd.DataFrame()
        
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        total_pnl = trades_df['pnl'].sum() if total_trades > 0 else 0
        roi = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        
        # Average win/loss
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
        
        # Profit Factor
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Sharpe Ratio (упрощенный)
        returns = trades_df['pnl']
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Maximum Drawdown
        cumulative = trades_df['pnl'].cumsum()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max)
        max_drawdown = drawdown.min()
        
        results = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'roi': roi,
            'final_balance': self.balance,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_duration': trades_df['duration'].mean()
        }
        
        logger.info(f"=== Backtest Results ===")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.info(f"Total P&L: ${total_pnl:.2f}")
        logger.info(f"ROI: {roi:.2f}%")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: ${max_drawdown:.2f}")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        
        return results
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Получить DataFrame со всеми сделками"""
        return pd.DataFrame(self.closed_trades)


def run_backtest_report(symbol: str = "BTC/USDT", days: int = 30):
    """
    Запустить backtesting и сохранить отчет
    """
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Инициализировать биржу
    exchange = ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET_KEY'),
        'enableRateLimit': True
    })
    
    # Создать движок
    engine = BacktestEngine(exchange, initial_balance=1000.0)
    
    # Запустить тест
    results = engine.run_backtest(symbol, '1h', days=days)
    
    # Сохранить результаты
    print("\n" + "=" * 60)
    print(f"BACKTEST REPORT: {symbol} ({days} days)")
    print("=" * 60)
    print(f"\n📊 RESULTS:")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Winning: {results['winning_trades']} | Losing: {results['losing_trades']}")
    print(f"Win Rate: {results['win_rate']:.1f}%")
    print(f"\n💰 PROFITABILITY:")
    print(f"Initial Balance: $1000.00")
    print(f"Final Balance: ${results['final_balance']:.2f}")
    print(f"Total P&L: ${results['total_pnl']:.2f}")
    print(f"ROI: {results['roi']:.2f}%")
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Max Drawdown: ${results['max_drawdown']:.2f}")
    print(f"Avg Win: ${results['avg_win']:.2f}")
    print(f"Avg Loss: ${results['avg_loss']:.2f}")
    print(f"Avg Trade Duration: {results['avg_duration']:.1f} hours")
    print("=" * 60 + "\n")
    
    # Сохранить сделки в CSV
    trades_df = engine.get_trades_dataframe()
    if not trades_df.empty:
        trades_df.to_csv(f'backtest_{symbol.replace("/", "_")}_{days}d.csv', index=False)
        print(f"✅ Trades saved to: backtest_{symbol.replace('/', '_')}_{days}d.csv")
    
    return results


if __name__ == '__main__':
    # Пример использования
    logging.basicConfig(level=logging.INFO)
    
    print("🔬 NexusTrader AI - Backtesting Engine\n")
    
    # Тест на BTC за 30 дней
    results = run_backtest_report("BTC/USDT", days=30)
