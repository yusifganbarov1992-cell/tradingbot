import { MarketData, TradeSignal, SignalType, LogEntry } from './types';

export const MOCK_MARKET_DATA: MarketData[] = Array.from({ length: 50 }, (_, i) => {
  const basePrice = 64000;
  const randomFluctuation = Math.random() * 1000 - 500;
  const trend = i * 50;
  const price = basePrice + trend + randomFluctuation;
  
  return {
    time: `${10 + Math.floor(i / 6)}:${(i % 6) * 10}0`,
    price: price,
    volume: Math.floor(Math.random() * 500) + 100,
    // LSTM prediction is slightly offset future version of price with noise
    predicted: price + (Math.random() * 200 - 50) + 100 
  };
});

export const INITIAL_SIGNALS: TradeSignal[] = [
  {
    id: 'sig-1',
    symbol: 'BTC/USDT',
    type: SignalType.BUY,
    price: 64230.50,
    timestamp: Date.now() - 1000 * 60 * 5, // 5 mins ago
    confidence: 87,
    reasoning: [
      'RSI (32) indicates oversold condition',
      'Price bounced off EMA-200 support',
      'Positive divergence on MACD',
      'Sentiment Analysis: Bullish news cycle'
    ],
    indicators: {
      rsi: 32,
      macd: 12.5,
      ema: 64100,
      atr: 450
    },
    status: 'PENDING'
  },
  {
    id: 'sig-2',
    symbol: 'ETH/USDT',
    type: SignalType.SELL,
    price: 3450.00,
    timestamp: Date.now() - 1000 * 60 * 30,
    confidence: 92,
    reasoning: [
      'Bollinger Band upper breakout failed',
      'Volume declining on price rise',
      'LSTM predicts trend reversal in 4h'
    ],
    indicators: {
      rsi: 76,
      macd: -5.2,
      ema: 3420,
      atr: 45
    },
    status: 'EXECUTED'
  }
];

export const MOCK_LOGS: LogEntry[] = [
  { id: '1', timestamp: Date.now() - 10000, level: 'INFO', message: 'Connected to Binance API WebSocket' },
  { id: '2', timestamp: Date.now() - 8000, level: 'SUCCESS', message: 'LSTM Model weights loaded successfully' },
  { id: '3', timestamp: Date.now() - 5000, level: 'INFO', message: 'Scanning top 20 pairs for volatility...' },
];