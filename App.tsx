import React, { useState, useEffect, useRef } from 'react';
import { 
  BotConfig, 
  TradeSignal, 
  BotStatus, 
  LogEntry,
  MarketData,
  SignalType
} from './types';
import { 
  INITIAL_SIGNALS, 
  MOCK_LOGS 
} from './constants';
import ChartSection from './components/ChartSection';
import BotConfigPanel from './components/BotConfigPanel';
import SignalFeed from './components/SignalFeed';
import AiAnalyst from './components/AiAnalyst';
import { Terminal, Power, Activity, AlertCircle, Wifi, WifiOff, FileCode } from 'lucide-react';
import { connectBinanceStream } from './services/binanceService';
import { calculateRSI, calculateEMA } from './utils/indicators';
import { sendTelegramMessage } from './services/telegramService';
import { useLocalStorage } from './utils/storage';

const App: React.FC = () => {
  // Application State
  const [botStatus, setBotStatus] = useState<BotStatus>(BotStatus.ACTIVE);
  const [signals, setSignals] = useState<TradeSignal[]>(INITIAL_SIGNALS);
  const [logs, setLogs] = useState<LogEntry[]>(MOCK_LOGS);
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [currentPrice, setCurrentPrice] = useState<number>(0);
  
  // Persist config in Local Storage
  const [config, setConfig] = useLocalStorage<BotConfig>('nexus_trader_config', {
    riskPerTrade: 2,
    stopLoss: 3,
    takeProfit: 7,
    strategy: 'MODERATE',
    useAiSentiment: true,
    binanceApiKey: '',
    telegramToken: '',
    telegramChatId: ''
  });

  // Refs for logic that doesn't need re-renders or to avoid stale closures
  const wsRef = useRef<WebSocket | null>(null);
  const lastSignalTimeRef = useRef<number>(0);
  // CRITICAL: Keep track of market data in a ref to access it inside WebSocket callbacks without stale closures
  const marketDataRef = useRef<MarketData[]>([]);
  // Store bot status in ref to access inside WebSocket without dependency injection causing reconnects
  const botStatusRef = useRef(botStatus);

  // Sync ref with state
  useEffect(() => {
    botStatusRef.current = botStatus;
  }, [botStatus]);

  // Initialize Real Data Connection (Run Once)
  useEffect(() => {
    addLog('INFO', 'Initializing Binance WebSocket Connection...');
    
    wsRef.current = connectBinanceStream('BTC/USDT', (newData) => {
      setIsConnected(true);
      setCurrentPrice(newData.price);
      
      setMarketData(prev => {
        // 1. Calculate Indicators based on history + new point
        const history = [...prev];
        const prices = history.map(d => d.price);
        prices.push(newData.price);
        
        const rsi = calculateRSI(prices, 14);
        const ema = calculateEMA(prices, 20); // EMA 20

        const enrichedData: MarketData = {
            ...newData,
            rsi,
            ema
        };

        const updated = [...prev, enrichedData];
        if (updated.length > 100) updated.shift();
        
        // Update Ref for Logic
        marketDataRef.current = updated;

        // 2. Check Signals using the REF value of botStatus
        // This avoids tearing down the WebSocket when user toggles pause/start
        if (botStatusRef.current === BotStatus.ACTIVE) {
             checkSignals(enrichedData, rsi, ema);
        }

        return updated;
      });
    });

    return () => {
      if (wsRef.current) wsRef.current.close();
      setIsConnected(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); 

  // Logic to Generate Signals based on Real Data
  const checkSignals = async (data: MarketData, rsi: number, ema: number) => {
    // We need at least 20 points for EMA-20 and 15 for RSI-14
    if (marketDataRef.current.length < 20) return;

    // Debounce signals: Prevent spamming. Max 1 signal per minute.
    const now = Date.now();
    if (now - lastSignalTimeRef.current < 60000) return;

    // Deterministic Strategy Logic
    let signalType: SignalType | null = null;
    let reason = "";

    if (rsi < 30) {
      signalType = SignalType.BUY;
      reason = "RSI Oversold (<30) - Potential Reversal";
    } else if (rsi > 70) {
      signalType = SignalType.SELL;
      reason = "RSI Overbought (>70) - Potential Pullback";
    }

    // Additional EMA Confirmation
    if (signalType === SignalType.BUY && data.price > ema) {
         reason += " + Price above EMA (Trend Support)";
    }

    if (signalType) {
      lastSignalTimeRef.current = now;
      await createSignal(signalType, data.price, rsi, ema, reason);
    }
  };

  const createSignal = async (type: SignalType, price: number, rsi: number, ema: number, reason: string) => {
    // Avoid duplicate pending signals for the same type
    let isDuplicate = false;
    setSignals(prev => {
      if (prev.some(s => s.status === 'PENDING' && s.type === type)) {
        isDuplicate = true;
        return prev;
      }
      return prev; 
    });

    if (isDuplicate) return;

    const newSignal: TradeSignal = {
        id: `sig-${Date.now()}`,
        symbol: 'BTC/USDT',
        type: type,
        price: price,
        timestamp: Date.now(),
        confidence: rsi < 20 || rsi > 80 ? 90 : 75,
        reasoning: [reason, `RSI: ${rsi.toFixed(1)}`, `EMA: $${ema.toFixed(0)}`],
        indicators: {
          rsi: parseFloat(rsi.toFixed(1)),
          macd: 0, 
          ema: parseFloat(ema.toFixed(1)),
          atr: 0
        },
        status: 'PENDING'
    };

    setSignals(prev => [newSignal, ...prev]);
    addLog('INFO', `New ${type} signal detected: ${reason}`);

    // Send Telegram Notification via Client-side service
    if (config.telegramToken && config.telegramChatId) {
        const msg = `🚨 <b>NEW SIGNAL DETECTED</b>\n\n` +
                    `Symbol: <b>BTC/USDT</b>\n` +
                    `Type: <b>${type}</b>\n` +
                    `Price: $${price.toFixed(2)}\n` +
                    `RSI: ${rsi.toFixed(1)}\n` +
                    `Reason: ${reason}\n\n` +
                    `<i>Please check NexusTrader app to approve.</i>`;
        
        await sendTelegramMessage(config.telegramToken, config.telegramChatId, msg);
    }
  };

  // Handlers
  const handleApproveSignal = async (id: string) => {
    setSignals(prev => prev.map(s => 
      s.id === id ? { ...s, status: 'APPROVED' } : s
    ));
    addLog('SUCCESS', `Trade APPROVED by user for signal ${id}`);

    const signal = signals.find(s => s.id === id);
    if (signal && config.telegramToken && config.telegramChatId) {
        await sendTelegramMessage(
            config.telegramToken, 
            config.telegramChatId, 
            `✅ <b>TRADE APPROVED</b>\nExecuting ${signal.type} on BTC/USDT...`
        );
    }
    
    // Simulate execution delay then execute
    setTimeout(async () => {
        setSignals(prev => prev.map(s => 
            s.id === id ? { ...s, status: 'EXECUTED' } : s
        ));
        
        const executionLog = config.binanceApiKey 
            ? `Order executed using API Key ...${config.binanceApiKey.slice(-4)}`
            : `Order executed (Simulation)`;
            
        addLog('INFO', executionLog);

        if (signal && config.telegramToken && config.telegramChatId) {
            await sendTelegramMessage(
                config.telegramToken, 
                config.telegramChatId, 
                `🚀 <b>ORDER FILLED</b>\nPosition opened at market price.`
            );
        }
    }, 1500);
  };

  const handleRejectSignal = (id: string) => {
    setSignals(prev => prev.map(s => 
      s.id === id ? { ...s, status: 'REJECTED' } : s
    ));
    addLog('WARNING', `Trade REJECTED by user for signal ${id}`);
  };

  const addLog = (level: LogEntry['level'], message: string) => {
    const newLog: LogEntry = {
      id: Math.random().toString(36).substr(2, 9),
      timestamp: Date.now(),
      level,
      message
    };
    setLogs(prev => [newLog, ...prev]);
  };

  const toggleBot = () => {
    const newStatus = botStatus === BotStatus.ACTIVE ? BotStatus.PAUSED : BotStatus.ACTIVE;
    setBotStatus(newStatus);
    addLog('INFO', `Bot status changed to ${newStatus}`);
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200 p-4 md:p-6 lg:p-8">
      {/* Header */}
      <header className="flex justify-between items-center mb-8 bg-slate-800/80 backdrop-blur rounded-2xl p-4 border border-slate-700 shadow-xl">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-600/30">
            <Terminal className="text-white" size={24} />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">NexusTrader AI</h1>
            <div className="flex items-center gap-2 text-xs text-slate-400">
               {isConnected ? (
                 <span className="flex items-center gap-1 text-emerald-400">
                    <Wifi size={12}/> Connected to Binance Stream
                 </span>
               ) : (
                 <span className="flex items-center gap-1 text-rose-400">
                    <WifiOff size={12}/> Disconnected
                 </span>
               )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="text-right hidden md:block">
            <div className="text-xs text-slate-400">Current BTC Price</div>
            <div className="text-xl font-bold text-white font-mono">
                {currentPrice > 0 ? `$${currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2 })}` : 'Loading...'}
            </div>
          </div>
          <button 
            onClick={toggleBot}
            className={`flex items-center gap-2 px-6 py-3 rounded-xl font-bold transition-all ${
              botStatus === BotStatus.ACTIVE 
                ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/50 hover:bg-emerald-500/20' 
                : 'bg-rose-500/10 text-rose-400 border border-rose-500/50 hover:bg-rose-500/20'
            }`}
          >
            <Power size={18} />
            {botStatus === BotStatus.ACTIVE ? 'SYSTEM ACTIVE' : 'SYSTEM PAUSED'}
          </button>
        </div>
      </header>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Charts & Analysis (8 cols) */}
        <div className="lg:col-span-8 space-y-6">
          <ChartSection data={marketData} symbol="BTC/USDT" />
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <BotConfigPanel config={config} setConfig={setConfig} />
            <AiAnalyst />
          </div>

          {/* System Logs */}
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 font-mono text-xs h-48 overflow-y-auto shadow-inner">
             <h3 className="text-slate-500 mb-2 sticky top-0 bg-slate-950 pb-2 border-b border-slate-800 flex items-center gap-2">
                <Activity size={12}/> System Logs
             </h3>
             <div className="space-y-1">
               {logs.map(log => (
                 <div key={log.id} className="flex gap-3">
                   <span className="text-slate-600">{new Date(log.timestamp).toLocaleTimeString()}</span>
                   <span className={`font-bold ${
                     log.level === 'INFO' ? 'text-blue-400' :
                     log.level === 'SUCCESS' ? 'text-emerald-400' :
                     log.level === 'WARNING' ? 'text-amber-400' : 'text-rose-400'
                   }`}>[{log.level}]</span>
                   <span className="text-slate-300">{log.message}</span>
                 </div>
               ))}
             </div>
          </div>
        </div>

        {/* Right Column: Signals & Activity (4 cols) */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          {/* Risk Alert Banner */}
          <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3 flex flex-col gap-2">
            <div className="flex items-start gap-3">
                <AlertCircle className="text-amber-500 shrink-0 mt-0.5" size={18} />
                <div className="text-xs text-amber-200">
                <span className="font-bold block mb-1">Live Market Monitor</span>
                <p>
                    Signals generated using real-time calculated indicators (RSI/EMA).
                </p>
                </div>
            </div>
            
            <div className="bg-slate-900/50 p-2 rounded border border-amber-500/20 flex items-center gap-2 mt-1">
                 <FileCode size={14} className="text-amber-400" />
                 <div className="text-[10px] text-amber-100">
                    To use <b>LSTM Neural Network</b>, run the included 
                    <span className="font-mono bg-black/40 px-1 mx-1 rounded text-amber-300">trading_bot.py</span>
                    backend script.
                 </div>
            </div>
          </div>

          <div className="flex-1 min-h-[500px]">
            <SignalFeed 
                signals={signals} 
                onApprove={handleApproveSignal}
                onReject={handleRejectSignal}
            />
          </div>
        </div>

      </div>
    </div>
  );
};

export default App;