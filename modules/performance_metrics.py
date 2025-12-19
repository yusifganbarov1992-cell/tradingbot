"""
Performance Metrics - Track and analyze bot performance
"""

import os
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


@dataclass
class DailyMetrics:
    """Daily performance metrics"""
    date: str
    trades_count: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_percent: float
    best_trade: float
    worst_trade: float
    avg_trade_duration: float


@dataclass 
class OverallMetrics:
    """Overall performance metrics"""
    total_days: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_percent: float
    avg_daily_pnl: float
    best_day: float
    worst_day: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_duration: float
    consecutive_wins: int
    consecutive_losses: int


class PerformanceTracker:
    """
    Track and analyze trading performance
    """
    
    def __init__(self):
        from database_supabase import SupabaseDatabase
        self.db = SupabaseDatabase()
    
    def get_all_trades(self, days: int = 30) -> List[Dict]:
        """Get all trades from database"""
        trades = self.db.get_trade_history(limit=1000)
        
        # Filter by date
        cutoff = datetime.now() - timedelta(days=days)
        filtered = []
        
        for t in trades:
            if t.get('trade_id', '').startswith('TEST_'):
                continue
            
            entry_time = t.get('entry_time', '')
            if entry_time:
                try:
                    trade_date = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                    if trade_date.replace(tzinfo=None) > cutoff:
                        filtered.append(t)
                except:
                    filtered.append(t)
        
        return filtered
    
    def calculate_daily_metrics(self, date: str, trades: List[Dict]) -> DailyMetrics:
        """Calculate metrics for a specific day"""
        day_trades = [t for t in trades if t.get('entry_time', '')[:10] == date and t.get('status') == 'closed']
        
        if not day_trades:
            return DailyMetrics(
                date=date, trades_count=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_pnl=0, total_pnl_percent=0,
                best_trade=0, worst_trade=0, avg_trade_duration=0
            )
        
        winning = [t for t in day_trades if (t.get('pnl') or 0) > 0]
        losing = [t for t in day_trades if (t.get('pnl') or 0) <= 0]
        
        pnls = [t.get('pnl_percent', 0) or 0 for t in day_trades]
        
        return DailyMetrics(
            date=date,
            trades_count=len(day_trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=(len(winning) / len(day_trades) * 100) if day_trades else 0,
            total_pnl=sum(t.get('pnl', 0) or 0 for t in day_trades),
            total_pnl_percent=sum(pnls),
            best_trade=max(pnls) if pnls else 0,
            worst_trade=min(pnls) if pnls else 0,
            avg_trade_duration=0  # Would need entry/exit times
        )
    
    def calculate_overall_metrics(self, trades: List[Dict]) -> OverallMetrics:
        """Calculate overall performance metrics"""
        closed_trades = [t for t in trades if t.get('status') == 'closed']
        
        if not closed_trades:
            return OverallMetrics(
                total_days=0, total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_pnl=0, total_pnl_percent=0, avg_daily_pnl=0,
                best_day=0, worst_day=0, max_drawdown=0, sharpe_ratio=0,
                profit_factor=0, avg_trade_duration=0, consecutive_wins=0, consecutive_losses=0
            )
        
        # Basic stats
        winning = [t for t in closed_trades if (t.get('pnl') or 0) > 0]
        losing = [t for t in closed_trades if (t.get('pnl') or 0) <= 0]
        
        total_pnl = sum(t.get('pnl', 0) or 0 for t in closed_trades)
        total_pnl_pct = sum(t.get('pnl_percent', 0) or 0 for t in closed_trades)
        
        # Daily breakdown
        dates = set(t.get('entry_time', '')[:10] for t in closed_trades if t.get('entry_time'))
        daily_pnls = []
        
        for date in dates:
            day_trades = [t for t in closed_trades if t.get('entry_time', '')[:10] == date]
            day_pnl = sum(t.get('pnl_percent', 0) or 0 for t in day_trades)
            daily_pnls.append(day_pnl)
        
        # Profit factor
        gross_profit = sum(t.get('pnl', 0) or 0 for t in winning)
        gross_loss = abs(sum(t.get('pnl', 0) or 0 for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Max drawdown (simplified)
        cumulative = 0
        peak = 0
        max_dd = 0
        for t in closed_trades:
            cumulative += t.get('pnl_percent', 0) or 0
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        
        # Consecutive wins/losses
        results = [1 if (t.get('pnl') or 0) > 0 else 0 for t in closed_trades]
        max_wins = max_losses = current_wins = current_losses = 0
        
        for r in results:
            if r == 1:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        # Sharpe ratio (simplified)
        import statistics
        if len(daily_pnls) > 1:
            sharpe = (statistics.mean(daily_pnls) / statistics.stdev(daily_pnls)) * (252 ** 0.5)
        else:
            sharpe = 0
        
        return OverallMetrics(
            total_days=len(dates),
            total_trades=len(closed_trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=(len(winning) / len(closed_trades) * 100) if closed_trades else 0,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_pct,
            avg_daily_pnl=statistics.mean(daily_pnls) if daily_pnls else 0,
            best_day=max(daily_pnls) if daily_pnls else 0,
            worst_day=min(daily_pnls) if daily_pnls else 0,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_trade_duration=0,
            consecutive_wins=max_wins,
            consecutive_losses=max_losses
        )
    
    def get_performance_report(self, days: int = 30) -> str:
        """Generate performance report"""
        trades = self.get_all_trades(days)
        metrics = self.calculate_overall_metrics(trades)
        
        # Calculate score
        score = 0
        if metrics.win_rate > 50: score += 2
        if metrics.total_pnl_percent > 0: score += 2
        if metrics.max_drawdown < 10: score += 2
        if metrics.sharpe_ratio > 1: score += 2
        if metrics.profit_factor > 1.5: score += 2
        
        report = []
        report.append("=" * 60)
        report.append("📊 PERFORMANCE REPORT")
        report.append(f"Period: Last {days} days")
        report.append("=" * 60)
        report.append("")
        report.append("📈 TRADING STATS:")
        report.append(f"   Total Trades: {metrics.total_trades}")
        report.append(f"   Winning: {metrics.winning_trades} | Losing: {metrics.losing_trades}")
        report.append(f"   Win Rate: {metrics.win_rate:.1f}%")
        report.append("")
        report.append("💰 P&L:")
        report.append(f"   Total P&L: ${metrics.total_pnl:.2f} ({metrics.total_pnl_percent:+.2f}%)")
        report.append(f"   Avg Daily: {metrics.avg_daily_pnl:+.2f}%")
        report.append(f"   Best Day: {metrics.best_day:+.2f}%")
        report.append(f"   Worst Day: {metrics.worst_day:+.2f}%")
        report.append("")
        report.append("📉 RISK METRICS:")
        report.append(f"   Max Drawdown: {metrics.max_drawdown:.2f}%")
        report.append(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        report.append(f"   Profit Factor: {metrics.profit_factor:.2f}")
        report.append("")
        report.append("🔥 STREAKS:")
        report.append(f"   Max Consecutive Wins: {metrics.consecutive_wins}")
        report.append(f"   Max Consecutive Losses: {metrics.consecutive_losses}")
        report.append("")
        report.append("=" * 60)
        report.append(f"SCORE: {'⭐' * score} ({score}/10)")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def get_metrics_json(self, days: int = 30) -> Dict:
        """Get metrics as JSON for API/dashboard"""
        trades = self.get_all_trades(days)
        metrics = self.calculate_overall_metrics(trades)
        
        return {
            'period_days': days,
            'total_trades': metrics.total_trades,
            'winning_trades': metrics.winning_trades,
            'losing_trades': metrics.losing_trades,
            'win_rate': round(metrics.win_rate, 2),
            'total_pnl': round(metrics.total_pnl, 4),
            'total_pnl_percent': round(metrics.total_pnl_percent, 2),
            'avg_daily_pnl': round(metrics.avg_daily_pnl, 2),
            'best_day': round(metrics.best_day, 2),
            'worst_day': round(metrics.worst_day, 2),
            'max_drawdown': round(metrics.max_drawdown, 2),
            'sharpe_ratio': round(metrics.sharpe_ratio, 2),
            'profit_factor': round(metrics.profit_factor, 2),
            'consecutive_wins': metrics.consecutive_wins,
            'consecutive_losses': metrics.consecutive_losses,
        }


def main():
    """Print performance report"""
    tracker = PerformanceTracker()
    print(tracker.get_performance_report(days=30))


if __name__ == "__main__":
    main()
