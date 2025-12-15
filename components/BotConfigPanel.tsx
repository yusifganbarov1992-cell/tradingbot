import React, { useState } from 'react';
import { BotConfig } from '../types';
import { Shield, Zap, TrendingUp, BrainCircuit, Key, MessageCircle, RefreshCw, CheckCircle2, Eye, EyeOff } from 'lucide-react';
import { getTelegramChatId, sendTelegramMessage } from '../services/telegramService';

interface BotConfigPanelProps {
  config: BotConfig;
  setConfig: (value: BotConfig | ((val: BotConfig) => BotConfig)) => void;
}

const BotConfigPanel: React.FC<BotConfigPanelProps> = ({ config, setConfig }) => {
  const [connectingTg, setConnectingTg] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);
  const [showToken, setShowToken] = useState(false);

  const handleRangeChange = (key: keyof BotConfig, value: number) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const handleTextChange = (key: keyof BotConfig, value: string) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const handleStrategyChange = (strategy: BotConfig['strategy']) => {
    setConfig(prev => ({ ...prev, strategy }));
  };

  const connectTelegram = async () => {
    setConnectingTg(true);
    const chatId = await getTelegramChatId(config.telegramToken);
    if (chatId) {
        setConfig(prev => ({ ...prev, telegramChatId: chatId }));
        await sendTelegramMessage(config.telegramToken, chatId, "✅ <b>NexusTrader AI Connected</b>\nReady to receive signals.");
    } else {
        alert("Could not find Chat ID. Please send a message to your bot first!");
    }
    setConnectingTg(false);
  };

  return (
    <div className="bg-slate-800/50 backdrop-blur-md border border-slate-700 rounded-xl p-6 shadow-xl h-full flex flex-col">
      <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
        <BrainCircuit className="text-indigo-400" size={24} />
        Agent Configuration
      </h2>

      <div className="space-y-6 overflow-y-auto pr-2 custom-scrollbar">
        {/* Strategy Selection */}
        <div>
          <label className="text-slate-400 text-sm font-medium mb-3 block">Trading Strategy</label>
          <div className="grid grid-cols-3 gap-2">
            {(['CONSERVATIVE', 'MODERATE', 'AGGRESSIVE'] as const).map((strat) => (
              <button
                key={strat}
                onClick={() => handleStrategyChange(strat)}
                className={`py-2 px-1 rounded-lg text-xs font-bold transition-all ${
                  config.strategy === strat 
                    ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/20' 
                    : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                }`}
              >
                {strat}
              </button>
            ))}
          </div>
        </div>

        {/* Risk Sliders */}
        <div className="space-y-4">
          <div>
            <div className="flex justify-between mb-1">
              <label className="text-slate-300 text-sm flex items-center gap-2">
                <Shield size={14} className="text-emerald-400"/> Stop Loss
              </label>
              <span className="text-emerald-400 text-sm font-mono">{config.stopLoss}%</span>
            </div>
            <input 
              type="range" 
              min="0.5" 
              max="10" 
              step="0.1" 
              value={config.stopLoss}
              onChange={(e) => handleRangeChange('stopLoss', parseFloat(e.target.value))}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
            />
          </div>

          <div>
            <div className="flex justify-between mb-1">
              <label className="text-slate-300 text-sm flex items-center gap-2">
                <TrendingUp size={14} className="text-indigo-400"/> Take Profit
              </label>
              <span className="text-indigo-400 text-sm font-mono">{config.takeProfit}%</span>
            </div>
            <input 
              type="range" 
              min="1" 
              max="50" 
              step="0.5" 
              value={config.takeProfit}
              onChange={(e) => handleRangeChange('takeProfit', parseFloat(e.target.value))}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
          </div>
        </div>

        {/* API Integrations */}
        <div className="pt-4 border-t border-slate-700 space-y-4">
            <h3 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                <Key size={14} /> API Integrations
            </h3>
            
            <div className="space-y-2">
                <label className="text-xs text-slate-400">Binance API Key (Public)</label>
                <div className="relative">
                  <input 
                      type={showApiKey ? "text" : "password"} 
                      value={config.binanceApiKey}
                      onChange={(e) => handleTextChange('binanceApiKey', e.target.value)}
                      className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-2 text-xs text-slate-300 focus:border-indigo-500 outline-none font-mono pr-8"
                      placeholder="Enter Binance API Key"
                  />
                  <button 
                    onClick={() => setShowApiKey(!showApiKey)}
                    className="absolute right-2 top-2 text-slate-500 hover:text-slate-300"
                  >
                    {showApiKey ? <EyeOff size={14}/> : <Eye size={14}/>}
                  </button>
                </div>
            </div>

            <div className="space-y-2">
                <label className="text-xs text-slate-400">Telegram Bot Token</label>
                <div className="relative">
                  <input 
                      type={showToken ? "text" : "password"}
                      value={config.telegramToken}
                      onChange={(e) => handleTextChange('telegramToken', e.target.value)}
                      className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-2 text-xs text-slate-300 focus:border-indigo-500 outline-none font-mono pr-8"
                      placeholder="123456:ABC-DEF..."
                  />
                  <button 
                    onClick={() => setShowToken(!showToken)}
                    className="absolute right-2 top-2 text-slate-500 hover:text-slate-300"
                  >
                    {showToken ? <EyeOff size={14}/> : <Eye size={14}/>}
                  </button>
                </div>
            </div>

            <div className="bg-slate-700/30 rounded p-3">
                <div className="flex justify-between items-center mb-2">
                    <label className="text-xs text-slate-400 flex items-center gap-1">
                        <MessageCircle size={12}/> Telegram Chat ID
                    </label>
                    {config.telegramChatId && <CheckCircle2 size={12} className="text-emerald-500"/>}
                </div>
                <div className="flex gap-2">
                    <input 
                        type="text" 
                        value={config.telegramChatId}
                        onChange={(e) => handleTextChange('telegramChatId', e.target.value)}
                        className="flex-1 bg-slate-900 border border-slate-600 rounded px-3 py-2 text-xs text-slate-300 focus:border-indigo-500 outline-none font-mono"
                        placeholder="ID will appear here"
                    />
                    <button 
                        onClick={connectTelegram}
                        disabled={connectingTg || !config.telegramToken}
                        className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white px-3 py-1 rounded text-xs transition-colors"
                        title="Send a message to your bot first!"
                    >
                        {connectingTg ? <RefreshCw className="animate-spin" size={14}/> : 'Auto-Connect'}
                    </button>
                </div>
                <p className="text-[10px] text-slate-500 mt-1">
                    1. Message your bot. 2. Click Auto-Connect.
                </p>
            </div>
        </div>

        {/* AI Toggle */}
        <div className="flex items-center justify-between pt-4 border-t border-slate-700">
           <span className="text-sm text-slate-300">Use AI Sentiment Analysis</span>
           <button 
             onClick={() => setConfig(prev => ({...prev, useAiSentiment: !prev.useAiSentiment}))}
             className={`w-12 h-6 rounded-full relative transition-colors ${config.useAiSentiment ? 'bg-indigo-600' : 'bg-slate-600'}`}
           >
             <span className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${config.useAiSentiment ? 'left-7' : 'left-1'}`} />
           </button>
        </div>
      </div>
    </div>
  );
};

export default BotConfigPanel;