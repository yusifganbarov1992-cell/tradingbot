import { MarketData } from '../types';

export const connectBinanceStream = (
  symbol: string, 
  onData: (data: MarketData) => void
) => {
  // Convert BTC/USDT to btcusdt for WebSocket stream
  const wsSymbol = symbol.replace('/', '').toLowerCase();
  const ws = new WebSocket(`wss://stream.binance.com:9443/ws/${wsSymbol}@kline_1m`);

  ws.onopen = () => {
    console.log('Connected to Binance WebSocket');
  };

  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    const kline = message.k;
    
    // Only process if candle is closed or updating current price
    // We map the Binance kline format to our MarketData type
    const newData: MarketData = {
      time: new Date(kline.t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      price: parseFloat(kline.c),
      volume: parseFloat(kline.v),
      // Mock LSTM prediction as "current price + trend" for visualization
      // In a real app, this would come from your Python backend API
      predicted: parseFloat(kline.c) * (1 + (Math.random() * 0.002 - 0.001)) 
    };

    onData(newData);
  };

  ws.onerror = (error) => {
    console.error('Binance WebSocket Error:', error);
  };

  return ws;
};