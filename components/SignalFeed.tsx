import React from 'react';
import { TradeSignal, SignalType } from '../types';
import { Check, X, AlertTriangle, TrendingUp, TrendingDown, Activity } from 'lucide-react';

interface SignalFeedProps {
  signals: TradeSignal[];
  onApprove: (id: string) => void;
  onReject: (id: string) => void;
}

const SignalFeed: React.FC<SignalFeedProps> = ({ signals, onApprove, onReject }) => {
  const pendingSignals = signals.filter(s => s.status === 'PENDING');
  const historySignals = signals.filter(s => s.status !== 'PENDING');

  return (
    <div className="bg-slate-800/50 backdrop-blur-md border border-slate-700 rounded-xl flex flex-col h-full shadow-xl overflow-hidden">
      <div className="p-4 border-b border-slate-700 flex justify-between items-center">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Activity className="text-indigo-400" size={20} />
          Signal Feed
        </h2>
        <span className="bg-indigo-500/20 text-indigo-300 text-xs px-2 py-1 rounded-full border border-indigo-500/30">
          {pendingSignals.length} Pending
        </span>
      </div>

      <div className="overflow-y-auto flex-1 p-4 space-y-4">
        {pendingSignals.length === 0 && (
            <div className="text-center text-slate-500 py-10">
                <p>No pending actions.</p>
                <p className="text-xs mt-2">The bot is analyzing the market...</p>
            </div>
        )}
        
        {pendingSignals.map(signal => (
          <div key={signal.id} className="bg-slate-700/40 border border-slate-600 rounded-lg p-4 relative overflow-hidden group">
            <div className={`absolute top-0 left-0 w-1 h-full ${signal.type === SignalType.BUY ? 'bg-emerald-500' : 'bg-rose-500'}`} />
            
            <div className="flex justify-between items-start mb-3">
              <div>
                <h3 className="font-bold text-white text-lg">{signal.symbol}</h3>
                <span className={`text-xs font-bold px-2 py-0.5 rounded ${signal.type === SignalType.BUY ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}`}>
                  {signal.type} @ ${signal.price.toLocaleString()}
                </span>
              </div>
              <div className="text-right">
                <span className="block text-xs text-slate-400">Confidence</span>
                <span className="font-mono text-indigo-400 font-bold">{signal.confidence}%</span>
              </div>
            </div>

            <div className="space-y-1 mb-4">
                {signal.reasoning.slice(0, 2).map((r, idx) => (
                    <div key={idx} className="flex items-center gap-2 text-xs text-slate-300">
                        <div className="w-1 h-1 rounded-full bg-slate-400" />
                        {r}
                    </div>
                ))}
            </div>

            <div className="grid grid-cols-2 gap-2 mt-2">
                <div className="bg-slate-800 rounded p-2 text-center">
                    <div className="text-[10px] text-slate-500 uppercase">RSI</div>
                    <div className={`text-sm font-mono ${signal.indicators.rsi < 30 || signal.indicators.rsi > 70 ? 'text-amber-400' : 'text-slate-300'}`}>
                        {signal.indicators.rsi}
                    </div>
                </div>
                 <div className="bg-slate-800 rounded p-2 text-center">
                    <div className="text-[10px] text-slate-500 uppercase">MACD</div>
                    <div className="text-sm font-mono text-slate-300">
                        {signal.indicators.macd}
                    </div>
                </div>
            </div>

            <div className="flex gap-2 mt-4">
              <button 
                onClick={() => onReject(signal.id)}
                className="flex-1 bg-slate-700 hover:bg-slate-600 text-slate-300 py-2 rounded-lg flex items-center justify-center gap-2 text-sm transition-colors"
              >
                <X size={16} /> Reject
              </button>
              <button 
                onClick={() => onApprove(signal.id)}
                className="flex-1 bg-emerald-600 hover:bg-emerald-500 text-white py-2 rounded-lg flex items-center justify-center gap-2 text-sm font-medium transition-colors shadow-lg shadow-emerald-900/20"
              >
                <Check size={16} /> Approve
              </button>
            </div>
          </div>
        ))}

        {historySignals.length > 0 && (
             <div className="mt-8 pt-4 border-t border-slate-700">
                <h3 className="text-xs font-semibold text-slate-500 uppercase mb-3">Recent Activity</h3>
                <div className="space-y-2">
                    {historySignals.map(signal => (
                        <div key={signal.id} className="flex justify-between items-center text-sm p-2 bg-slate-800/30 rounded border border-slate-700/50 opacity-60">
                             <span className="font-mono text-slate-400">{signal.symbol}</span>
                             <span className={signal.type === 'BUY' ? 'text-emerald-500' : 'text-rose-500'}>{signal.type}</span>
                             <span className={`text-xs px-1.5 py-0.5 rounded ${
                                 signal.status === 'EXECUTED' ? 'bg-indigo-500/20 text-indigo-400' : 'bg-red-500/20 text-red-400'
                             }`}>{signal.status}</span>
                        </div>
                    ))}
                </div>
             </div>
        )}
      </div>
    </div>
  );
};

export default SignalFeed;