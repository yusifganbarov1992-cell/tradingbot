"""
Performance Analyzer - Модуль самоанализа результатов торговли

Функционал:
- Анализ закрытых сделок (win rate, ROI, Sharpe ratio)
- Эффективность каждого фильтра
- Оптимальные параметры (ATR multiplier, confidence threshold)
- Daily/weekly отчеты
- Correlation analysis
- Рекомендации по улучшению

Источник: Адаптировано из backtesting.py + vectorbt
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Анализатор производительности - понимает ПОЧЕМУ сделки успешны или нет
    """
    
    def __init__(self, db_path: str = 'trading_history.db'):
        """
        Args:
            db_path: Путь к SQLite базе данных
        """
        self.db_path = db_path
        
        # Кэш для оптимизации
        self.cache = {
            'last_analysis': None,
            'last_analysis_time': None
        }
        
        logger.info("📊 PerformanceAnalyzer initialized")
    
    def analyze_closed_trades(self, days: int = 30) -> Dict:
        """
        Анализ закрытых сделок за период
        
        Args:
            days: Период анализа (дней)
            
        Returns:
            Dict с метриками
        """
        conn = sqlite3.connect(self.db_path)
        
        # Дата начала периода
        start_date = datetime.now() - timedelta(days=days)
        
        # Получить закрытые сделки
        query = """
        SELECT 
            trade_id, symbol, side, entry_price, exit_price,
            amount, usdt_amount, fee, exit_fee,
            pnl, pnl_percent,
            entry_time, exit_time,
            exit_reason, mode
        FROM trades
        WHERE status = 'closed'
          AND exit_time >= ?
        ORDER BY exit_time DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date.isoformat(),))
        conn.close()
        
        if len(df) == 0:
            return {
                'total_trades': 0,
                'message': 'Нет закрытых сделок за период'
            }
        
        # Базовые метрики
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = df['pnl'].sum()
        avg_pnl = df['pnl'].mean()
        
        # Максимальная просадка
        cumulative_pnl = df['pnl'].cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / running_max.max() * 100) if running_max.max() > 0 else 0
        
        # Sharpe Ratio (упрощенный)
        returns = df['pnl_percent']
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Средняя длительность сделок
        df['duration'] = pd.to_datetime(df['exit_time']) - pd.to_datetime(df['entry_time'])
        avg_duration_hours = df['duration'].mean().total_seconds() / 3600
        
        # Лучшие/худшие сделки
        best_trade = df.loc[df['pnl'].idxmax()] if len(df) > 0 else None
        worst_trade = df.loc[df['pnl'].idxmin()] if len(df) > 0 else None
        
        # ROI
        total_investment = df['usdt_amount'].sum()
        roi = (total_pnl / total_investment * 100) if total_investment > 0 else 0
        
        # Анализ по символам
        symbol_stats = df.groupby('symbol').agg({
            'pnl': ['sum', 'mean', 'count'],
            'pnl_percent': 'mean'
        }).round(2)
        
        # Анализ по причинам закрытия
        exit_reason_stats = df.groupby('exit_reason').agg({
            'pnl': ['sum', 'mean', 'count']
        }).round(2)
        
        result = {
            'period_days': days,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_pnl': round(avg_pnl, 2),
            'roi': round(roi, 2),
            'max_drawdown': round(max_drawdown, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'avg_duration_hours': round(avg_duration_hours, 2),
            'best_trade': {
                'symbol': best_trade['symbol'],
                'pnl': round(best_trade['pnl'], 2),
                'pnl_percent': round(best_trade['pnl_percent'], 2)
            } if best_trade is not None else None,
            'worst_trade': {
                'symbol': worst_trade['symbol'],
                'pnl': round(worst_trade['pnl'], 2),
                'pnl_percent': round(worst_trade['pnl_percent'], 2)
            } if worst_trade is not None else None,
            'by_symbol': symbol_stats.to_dict() if len(symbol_stats) > 0 else {},
            'by_exit_reason': exit_reason_stats.to_dict() if len(exit_reason_stats) > 0 else {}
        }
        
        logger.info(f"📊 Analyzed {total_trades} closed trades (win rate: {win_rate:.1f}%)")
        
        return result
    
    def analyze_filter_effectiveness(self, days: int = 30) -> Dict:
        """
        Анализ эффективности каждого фильтра
        
        Определяет какие технические индикаторы лучше предсказывают успех
        
        Args:
            days: Период анализа
            
        Returns:
            Dict с эффективностью фильтров
        """
        conn = sqlite3.connect(self.db_path)
        
        start_date = datetime.now() - timedelta(days=days)
        
        # Получить сигналы с индикаторами и результаты сделок
        query = """
        SELECT 
            s.signal_id,
            s.symbol,
            s.signal,
            s.indicators,
            t.pnl,
            t.pnl_percent,
            t.status
        FROM signals s
        LEFT JOIN trades t ON s.trade_id = t.trade_id
        WHERE s.timestamp >= ?
          AND t.status = 'closed'
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date.isoformat(),))
        conn.close()
        
        if len(df) == 0:
            return {'message': 'Недостаточно данных для анализа фильтров'}
        
        # Парсинг indicators JSON
        import json
        df['indicators'] = df['indicators'].apply(lambda x: json.loads(x) if isinstance(x, str) else {})
        
        # Анализ корреляции индикаторов с успехом
        filter_stats = {}
        
        # RSI эффективность
        if 'rsi' in df['indicators'].iloc[0]:
            df['rsi'] = df['indicators'].apply(lambda x: x.get('rsi', 50))
            
            # Сделки с RSI < 30 (oversold)
            oversold = df[df['rsi'] < 30]
            if len(oversold) > 0:
                filter_stats['rsi_oversold'] = {
                    'trades': len(oversold),
                    'win_rate': (len(oversold[oversold['pnl'] > 0]) / len(oversold) * 100),
                    'avg_pnl': oversold['pnl'].mean(),
                    'effectiveness': 'HIGH' if oversold['pnl'].mean() > df['pnl'].mean() else 'MEDIUM'
                }
            
            # Сделки с RSI > 70 (overbought)
            overbought = df[df['rsi'] > 70]
            if len(overbought) > 0:
                filter_stats['rsi_overbought'] = {
                    'trades': len(overbought),
                    'win_rate': (len(overbought[overbought['pnl'] > 0]) / len(overbought) * 100),
                    'avg_pnl': overbought['pnl'].mean(),
                    'effectiveness': 'HIGH' if overbought['pnl'].mean() > df['pnl'].mean() else 'MEDIUM'
                }
        
        # Volume spike эффективность
        if 'volume' in df['indicators'].iloc[0] and 'avg_volume' in df['indicators'].iloc[0]:
            df['volume_ratio'] = df['indicators'].apply(
                lambda x: x.get('volume', 0) / x.get('avg_volume', 1) if x.get('avg_volume', 0) > 0 else 1
            )
            
            volume_spike = df[df['volume_ratio'] > 1.5]
            if len(volume_spike) > 0:
                filter_stats['volume_spike'] = {
                    'trades': len(volume_spike),
                    'win_rate': (len(volume_spike[volume_spike['pnl'] > 0]) / len(volume_spike) * 100),
                    'avg_pnl': volume_spike['pnl'].mean(),
                    'effectiveness': 'HIGH' if volume_spike['pnl'].mean() > df['pnl'].mean() else 'MEDIUM'
                }
        
        # EMA trend эффективность
        if 'ema20' in df['indicators'].iloc[0] and 'ema50' in df['indicators'].iloc[0]:
            df['ema_trend'] = df['indicators'].apply(
                lambda x: 'UP' if x.get('ema20', 0) > x.get('ema50', 0) else 'DOWN'
            )
            
            uptrend = df[df['ema_trend'] == 'UP']
            if len(uptrend) > 0:
                filter_stats['ema_uptrend'] = {
                    'trades': len(uptrend),
                    'win_rate': (len(uptrend[uptrend['pnl'] > 0]) / len(uptrend) * 100),
                    'avg_pnl': uptrend['pnl'].mean(),
                    'effectiveness': 'HIGH' if uptrend['pnl'].mean() > df['pnl'].mean() else 'MEDIUM'
                }
        
        logger.info(f"📊 Analyzed effectiveness of {len(filter_stats)} filters")
        
        return filter_stats
    
    def get_optimal_parameters(self, days: int = 30) -> Dict:
        """
        Определение оптимальных параметров на основе исторических данных
        
        Returns:
            Dict с рекомендациями
        """
        conn = sqlite3.connect(self.db_path)
        
        start_date = datetime.now() - timedelta(days=days)
        
        # Получить все сделки с параметрами
        query = """
        SELECT 
            t.*,
            s.ai_analysis
        FROM trades t
        LEFT JOIN signals s ON t.trade_id = s.trade_id
        WHERE t.status = 'closed'
          AND t.exit_time >= ?
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date.isoformat(),))
        conn.close()
        
        if len(df) == 0:
            return {'message': 'Недостаточно данных для оптимизации'}
        
        # Парсинг AI confidence
        import json
        df['ai_analysis'] = df['ai_analysis'].apply(lambda x: json.loads(x) if isinstance(x, str) else {})
        df['ai_confidence'] = df['ai_analysis'].apply(lambda x: x.get('confidence', 0))
        
        # Оптимальный порог confidence
        confidence_ranges = [
            (7.0, 7.5), (7.5, 8.0), (8.0, 8.5), (8.5, 9.0), (9.0, 10.0)
        ]
        
        confidence_stats = {}
        for low, high in confidence_ranges:
            subset = df[(df['ai_confidence'] >= low) & (df['ai_confidence'] < high)]
            if len(subset) > 0:
                confidence_stats[f'{low}-{high}'] = {
                    'trades': len(subset),
                    'win_rate': (len(subset[subset['pnl'] > 0]) / len(subset) * 100),
                    'avg_pnl': subset['pnl'].mean(),
                    'avg_roi': subset['pnl_percent'].mean()
                }
        
        # Найти лучший диапазон
        best_confidence_range = max(
            confidence_stats.items(),
            key=lambda x: x[1]['avg_pnl']
        ) if confidence_stats else None
        
        # Оптимальный ATR multiplier (из stop_loss/take_profit)
        df['stop_loss_pct'] = abs((df['stop_loss'] - df['entry_price']) / df['entry_price'] * 100)
        df['take_profit_pct'] = abs((df['take_profit'] - df['entry_price']) / df['entry_price'] * 100)
        
        avg_stop_loss_pct = df['stop_loss_pct'].mean()
        avg_take_profit_pct = df['take_profit_pct'].mean()
        
        # Рекомендации
        recommendations = {
            'optimal_confidence_range': {
                'range': best_confidence_range[0] if best_confidence_range else 'N/A',
                'win_rate': round(best_confidence_range[1]['win_rate'], 2) if best_confidence_range else 0,
                'avg_pnl': round(best_confidence_range[1]['avg_pnl'], 2) if best_confidence_range else 0
            },
            'confidence_stats': confidence_stats,
            'optimal_stop_loss_pct': round(avg_stop_loss_pct, 2),
            'optimal_take_profit_pct': round(avg_take_profit_pct, 2),
            'current_risk_reward': round(avg_take_profit_pct / avg_stop_loss_pct, 2) if avg_stop_loss_pct > 0 else 0
        }
        
        logger.info(f"📊 Calculated optimal parameters from {len(df)} trades")
        
        return recommendations
    
    def generate_daily_report(self) -> str:
        """
        Генерация ежедневного отчета
        
        Returns:
            Форматированный отчет
        """
        # Анализ за последние 24 часа
        today_analysis = self.analyze_closed_trades(days=1)
        
        # Анализ за последние 7 дней (для сравнения)
        week_analysis = self.analyze_closed_trades(days=7)
        
        report = f"""
📊 DAILY PERFORMANCE REPORT
{'='*50}

📅 Период: Последние 24 часа

💼 СДЕЛКИ:
  • Всего: {today_analysis.get('total_trades', 0)}
  • Прибыльных: {today_analysis.get('winning_trades', 0)}
  • Убыточных: {today_analysis.get('losing_trades', 0)}
  • Win Rate: {today_analysis.get('win_rate', 0)}%

💰 ФИНАНСЫ:
  • Total P&L: ${today_analysis.get('total_pnl', 0)}
  • Average P&L: ${today_analysis.get('avg_pnl', 0)}
  • ROI: {today_analysis.get('roi', 0)}%
  • Max Drawdown: ${today_analysis.get('max_drawdown', 0)} ({today_analysis.get('max_drawdown_pct', 0)}%)

📈 КАЧЕСТВО:
  • Sharpe Ratio: {today_analysis.get('sharpe_ratio', 0)}
  • Avg Duration: {today_analysis.get('avg_duration_hours', 0)}h

🔍 ЛУЧШАЯ/ХУДШАЯ СДЕЛКА:
"""
        
        if today_analysis.get('best_trade'):
            report += f"  ✅ Best: {today_analysis['best_trade']['symbol']} (+${today_analysis['best_trade']['pnl']})\n"
        
        if today_analysis.get('worst_trade'):
            report += f"  ❌ Worst: {today_analysis['worst_trade']['symbol']} (-${abs(today_analysis['worst_trade']['pnl'])})\n"
        
        # Сравнение с недельными показателями
        report += f"\n📊 СРАВНЕНИЕ С НЕДЕЛЕЙ:\n"
        report += f"  • Week Win Rate: {week_analysis.get('win_rate', 0)}%\n"
        report += f"  • Week ROI: {week_analysis.get('roi', 0)}%\n"
        
        improvement = today_analysis.get('win_rate', 0) - week_analysis.get('win_rate', 0)
        if improvement > 0:
            report += f"  ✅ Улучшение: +{improvement:.1f}%\n"
        elif improvement < 0:
            report += f"  ⚠️  Снижение: {improvement:.1f}%\n"
        
        report += f"\n{'='*50}\n"
        
        return report
    
    def get_recommendations(self) -> List[str]:
        """
        Генерация рекомендаций на основе анализа
        
        Returns:
            Список рекомендаций
        """
        recommendations = []
        
        # Анализ за 30 дней
        analysis = self.analyze_closed_trades(days=30)
        
        if analysis.get('total_trades', 0) < 10:
            recommendations.append("⏳ Недостаточно данных. Продолжайте торговать для анализа.")
            return recommendations
        
        # Win rate рекомендации
        win_rate = analysis.get('win_rate', 0)
        if win_rate < 50:
            recommendations.append("⚠️  Win rate < 50%. Рекомендую повысить MIN_CONFIDENCE до 8.0+")
        elif win_rate > 70:
            recommendations.append("✅ Excellent win rate! Можно рассмотреть aggressive mode")
        
        # Sharpe ratio рекомендации
        sharpe = analysis.get('sharpe_ratio', 0)
        if sharpe < 1.0:
            recommendations.append("📉 Low Sharpe ratio. Улучшите risk management (stop loss)")
        elif sharpe > 2.0:
            recommendations.append("✅ Excellent risk-adjusted returns!")
        
        # Drawdown рекомендации
        max_dd_pct = abs(analysis.get('max_drawdown_pct', 0))
        if max_dd_pct > 20:
            recommendations.append("🚨 High drawdown! Уменьшите position size или MAX_POSITIONS")
        
        # ROI рекомендации
        roi = analysis.get('roi', 0)
        if roi < 0:
            recommendations.append("❌ Negative ROI. Пересмотрите стратегию или параметры")
        elif roi > 10:
            recommendations.append("✅ Great ROI! Стратегия работает хорошо")
        
        # Фильтры
        filter_analysis = self.analyze_filter_effectiveness(days=30)
        if filter_analysis.get('volume_spike', {}).get('effectiveness') == 'HIGH':
            recommendations.append("✅ Volume spike filter очень эффективен - продолжайте использовать")
        
        return recommendations


# ========================================
# INTEGRATION EXAMPLE для trading_bot.py
# ========================================

"""
# В __init__ TradingAgent:
from modules.performance_analyzer import PerformanceAnalyzer

self.performance = PerformanceAnalyzer(db_path='trading_history.db')

# После закрытия позиции (в close_position):
def close_position(self, symbol: str, exit_price: float, reason: str):
    # ... existing code ...
    
    # Анализ после закрытия
    if self.performance:
        analysis = self.performance.analyze_closed_trades(days=7)
        logger.info(f"📊 Weekly win rate: {analysis.get('win_rate', 0)}%")

# Telegram команда для отчета:
async def performance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agent = context.bot_data['agent']
    
    # Ежедневный отчет
    report = agent.performance.generate_daily_report()
    await update.message.reply_text(report)

async def recommendations_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agent = context.bot_data['agent']
    
    recs = agent.performance.get_recommendations()
    
    message = "💡 РЕКОМЕНДАЦИИ:\\n\\n"
    for i, rec in enumerate(recs, 1):
        message += f"{i}. {rec}\\n"
    
    await update.message.reply_text(message)
"""
