"""
База данных с Supabase для облачного хранения истории сделок
"""
import os
import logging
from datetime import datetime
from typing import Optional, Dict, List
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class SupabaseDatabase:
    def __init__(self):
        """Инициализация Supabase клиента"""
        url = os.getenv('SUPABASE_URL')
        # Используем service_role key для полного доступа
        key = os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_KEY')
        
        if not url or not key:
            raise ValueError("SUPABASE_URL и SUPABASE_SERVICE_KEY должны быть в .env")
        
        self.client: Client = create_client(url, key)
        logger.info("✅ Supabase подключена")
    
    def save_signal(self, trade_id: str, symbol: str, signal_type: str, 
                   price: float, indicators: dict, ai_analysis: dict, 
                   position_info: dict) -> bool:
        """Сохранить торговый сигнал в Supabase"""
        try:
            data = {
                'trade_id': trade_id,
                'symbol': symbol,
                'signal_type': signal_type,
                'price': price,
                'timestamp': datetime.utcnow().isoformat(),
                'rsi': indicators.get('rsi'),
                'ema20': indicators.get('ema20'),
                'ema50': indicators.get('ema50'),
                'macd': indicators.get('macd'),
                'volume': indicators.get('volume'),
                'avg_volume': indicators.get('avg_volume'),
                'atr': indicators.get('atr'),
                'filters_passed': indicators.get('filters_passed', 0),
                'ai_signal': ai_analysis.get('signal'),
                'ai_confidence': ai_analysis.get('confidence'),
                'ai_reason': ai_analysis.get('reason'),
                'status': 'pending',
                'amount': position_info.get('amount'),
                'usdt_amount': position_info.get('usdt_amount'),
                'fee': position_info.get('fee')
            }
            
            response = self.client.table('signals').insert(data).execute()
            logger.info(f"💾 Сигнал сохранён в Supabase: {trade_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения сигнала в Supabase: {e}")
            return False
    
    def save_trade(self, trade_id: str, symbol: str, side: str, 
                   entry_price: float, amount: float, usdt_amount: float,
                   mode: str = 'paper', stop_loss: float = None, 
                   take_profit: float = None, fee: float = None) -> bool:
        """Сохранить сделку в Supabase"""
        try:
            data = {
                'trade_id': trade_id,
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'amount': amount,
                'usdt_amount': usdt_amount,
                'mode': mode,
                'entry_time': datetime.utcnow().isoformat(),
                'status': 'open',
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'fee': fee
            }
            
            response = self.client.table('trades').insert(data).execute()
            logger.info(f"💾 Сделка сохранена в Supabase: {trade_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения сделки в Supabase: {e}")
            return False
    
    def update_trade(self, trade_id: str, exit_price: float, 
                    pnl: float, pnl_percent: float, fee: float = None) -> bool:
        """Обновить сделку при закрытии"""
        try:
            data = {
                'exit_price': exit_price,
                'exit_time': datetime.utcnow().isoformat(),
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'status': 'closed'
            }
            if fee:
                data['fee'] = fee
            
            response = self.client.table('trades').update(data).eq('trade_id', trade_id).execute()
            logger.info(f"💾 Сделка обновлена в Supabase: {trade_id} | PNL: {pnl_percent:.2f}%")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления сделки в Supabase: {e}")
            return False
    
    def get_open_trades(self) -> List[Dict]:
        """Получить открытые сделки"""
        try:
            response = self.client.table('trades').select('*').eq('status', 'open').execute()
            return response.data
        except Exception as e:
            logger.error(f"❌ Ошибка получения открытых сделок: {e}")
            return []
    
    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Получить историю сделок"""
        try:
            response = self.client.table('trades').select('*').order('entry_time', desc=True).limit(limit).execute()
            return response.data
        except Exception as e:
            logger.error(f"❌ Ошибка получения истории: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Получить статистику торговли"""
        try:
            # Все закрытые сделки
            response = self.client.table('trades').select('*').eq('status', 'closed').execute()
            trades = response.data
            
            if not trades:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0
                }
            
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
            losing_trades = sum(1 for t in trades if t.get('pnl', 0) <= 0)
            total_pnl = sum(t.get('pnl', 0) for t in trades)
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                'total_pnl': total_pnl,
                'avg_pnl': total_pnl / total_trades if total_trades > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики: {e}")
            return {}
    
    def save_performance(self, stats: Dict) -> bool:
        """Сохранить показатели производительности"""
        try:
            data = {
                'timestamp': datetime.utcnow().isoformat(),
                'total_trades': stats.get('total_trades', 0),
                'winning_trades': stats.get('winning_trades', 0),
                'losing_trades': stats.get('losing_trades', 0),
                'win_rate': stats.get('win_rate', 0),
                'total_pnl': stats.get('total_pnl', 0),
                'avg_pnl': stats.get('avg_pnl', 0),
                'max_drawdown': stats.get('max_drawdown', 0),
                'balance': stats.get('balance', 0)
            }
            
            response = self.client.table('performance').insert(data).execute()
            logger.info("💾 Показатели производительности сохранены")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения показателей: {e}")
            return False
