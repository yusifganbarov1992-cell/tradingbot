import React from 'react';
import { 
  ResponsiveContainer, 
  ComposedChart, 
  Line, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend,
  Bar 
} from 'recharts';
import { MarketData } from '../types';

interface ChartSectionProps {
  data: MarketData[];
  symbol: string;
}

const ChartSection: React.FC<ChartSectionProps> = ({ data, symbol }) => {
  return (
    <div className="bg-slate-800/50 backdrop-blur-md border border-slate-700 rounded-xl p-6 shadow-xl">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span className="w-2 h-6 bg-indigo-500 rounded-full"></span>
            {symbol} Market Analysis
          </h2>
          <p className="text-slate-400 text-sm mt-1">Real-time price vs. Technical Indicators</p>
        </div>
        <div className="flex gap-2">
           <span className="px-3 py-1 bg-slate-700 rounded text-xs text-slate-300">1H</span>
           <span className="px-3 py-1 bg-indigo-600 rounded text-xs text-white">4H</span>
           <span className="px-3 py-1 bg-slate-700 rounded text-xs text-slate-300">1D</span>
        </div>
      </div>

      <div className="h-[400px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data}>
            <defs>
              <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#818cf8" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#818cf8" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#334155" strokeDasharray="3 3" vertical={false} />
            <XAxis 
              dataKey="time" 
              stroke="#94a3b8" 
              tick={{fontSize: 12}} 
              tickLine={false}
              axisLine={false}
            />
            <YAxis 
              domain={['auto', 'auto']} 
              stroke="#94a3b8" 
              tick={{fontSize: 12}} 
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `$${value.toLocaleString()}`}
            />
            <Tooltip 
              contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
              itemStyle={{ color: '#e2e8f0' }}
            />
            <Legend />
            <Area 
              type="monotone" 
              dataKey="price" 
              name="Price" 
              stroke="#818cf8" 
              strokeWidth={2}
              fillOpacity={1} 
              fill="url(#colorPrice)" 
            />
            <Line 
              type="monotone" 
              dataKey="ema" 
              name="EMA (20)" 
              stroke="#f59e0b" 
              strokeWidth={2} 
              dot={false} 
            />
            <Line 
              type="monotone" 
              dataKey="predicted" 
              name="AI Prediction" 
              stroke="#34d399" 
              strokeWidth={2} 
              dot={false} 
              strokeDasharray="5 5" 
            />
            <Bar dataKey="volume" name="Volume" barSize={20} fill="#475569" opacity={0.3} yAxisId={0} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ChartSection;