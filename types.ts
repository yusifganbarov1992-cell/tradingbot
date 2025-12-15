export enum SignalType {
  BUY = 'BUY',
  SELL = 'SELL',
  HOLD = 'HOLD'
}

export enum BotStatus {
  ACTIVE = 'ACTIVE',
  PAUSED = 'PAUSED',
  TRAINING = 'TRAINING'
}

export interface MarketData {
  time: string;
  price: number;
  volume: number;
  predicted: number; // LSTM Prediction
  ema?: number;      // Exponential Moving Average
  rsi?: number;      // Relative Strength Index
}

export interface IndicatorValues {
  rsi: number;
  macd: number;
  ema: number;
  atr: number;
}

export interface TradeSignal {
  id: string;
  symbol: string;
  type: SignalType;
  price: number;
  timestamp: number;
  confidence: number; // 0-100%
  reasoning: string[]; // e.g. "RSI Oversold", "LSTM Uptrend"
  indicators: IndicatorValues;
  status: 'PENDING' | 'APPROVED' | 'REJECTED' | 'EXECUTED';
}

export interface BotConfig {
  riskPerTrade: number; // Percentage
  stopLoss: number; // Percentage
  takeProfit: number; // Percentage
  strategy: 'CONSERVATIVE' | 'MODERATE' | 'AGGRESSIVE';
  useAiSentiment: boolean;
  binanceApiKey: string;
  telegramToken: string;
  telegramChatId: string;
}

export interface LogEntry {
  id: string;
  timestamp: number;
  level: 'INFO' | 'WARNING' | 'ERROR' | 'SUCCESS';
  message: string;
}