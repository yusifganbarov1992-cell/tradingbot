import React, { useState } from 'react';
import { analyzeMarketSentiment } from '../services/geminiService';
import { Bot, Sparkles, Loader2, MessageSquare } from 'lucide-react';

const AiAnalyst: React.FC = () => {
  const [analysis, setAnalysis] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [symbol, setSymbol] = useState('BTC/USDT');

  const handleAnalysis = async () => {
    setLoading(true);
    // In a real app, 'news' would come from an external news API aggregator
    const mockNews = "Bitcoin ETFs see record inflows. Regulatory concerns ease in Europe. Tech stocks rallying affecting crypto correlation.";
    const result = await analyzeMarketSentiment(symbol, mockNews);
    setAnalysis(result);
    setLoading(false);
  };

  return (
    <div className="bg-indigo-900/20 backdrop-blur-md border border-indigo-500/30 rounded-xl p-6 shadow-xl">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-indigo-600 rounded-lg">
            <Sparkles className="text-white" size={20} />
        </div>
        <div>
            <h2 className="text-lg font-bold text-white">Gemini Market Analyst</h2>
            <p className="text-xs text-indigo-300">AI-powered sentiment & context analysis</p>
        </div>
      </div>

      <div className="mb-4">
        <label className="text-xs text-indigo-200 mb-1 block">Target Asset</label>
        <div className="flex gap-2">
            <select 
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                className="bg-slate-800 border border-slate-600 text-white text-sm rounded-lg p-2.5 flex-1 focus:ring-indigo-500 focus:border-indigo-500"
            >
                <option value="BTC/USDT">BTC/USDT</option>
                <option value="ETH/USDT">ETH/USDT</option>
                <option value="SOL/USDT">SOL/USDT</option>
            </select>
            <button 
                onClick={handleAnalysis}
                disabled={loading}
                className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
            >
                {loading ? <Loader2 className="animate-spin" size={16}/> : <Bot size={16}/>}
                Analyze
            </button>
        </div>
      </div>

      <div className="bg-slate-900/50 rounded-lg p-4 border border-indigo-500/20 min-h-[120px]">
        {loading ? (
            <div className="flex flex-col items-center justify-center h-full text-indigo-300 gap-2">
                <Loader2 className="animate-spin" />
                <span className="text-xs animate-pulse">Processing market data...</span>
            </div>
        ) : analysis ? (
            <div className="text-sm text-slate-200 whitespace-pre-line leading-relaxed">
                {analysis}
            </div>
        ) : (
            <div className="flex flex-col items-center justify-center h-full text-slate-500 gap-2">
                <MessageSquare size={24} opacity={0.5} />
                <span className="text-xs">Ask the AI to analyze current market conditions</span>
            </div>
        )}
      </div>
    </div>
  );
};

export default AiAnalyst;