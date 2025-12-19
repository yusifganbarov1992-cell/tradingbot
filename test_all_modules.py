"""Test all new modules"""
import sys
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

print('Testing all new modules...')
print()

# 1. Backtester
print('1. BACKTESTER')
from modules.backtester import Backtester
bt = Backtester(initial_balance=1000)
result = bt.run_backtest('BTC/USDT', '1h', days=7)
print(f'   Trades: {result.total_trades}')
print(f'   Win Rate: {result.win_rate:.1f}%')
print(f'   P&L: {result.total_pnl_percent:+.2f}%')
print()

# 2. Health Monitor
print('2. HEALTH MONITOR')
import asyncio
from modules.health_monitor import HealthMonitor
monitor = HealthMonitor()
asyncio.run(monitor.run_all_checks())
print(f'   Status: {monitor.get_overall_status()}')
print()

# 3. Performance Metrics
print('3. PERFORMANCE METRICS')
from modules.performance_metrics import PerformanceTracker
tracker = PerformanceTracker()
metrics = tracker.get_metrics_json(days=30)
print(f'   Trades: {metrics["total_trades"]}')
print(f'   Win Rate: {metrics["win_rate"]}%')
print()

# 4. Retry Utils
print('4. RETRY UTILS')
from modules.retry_utils import safe_execute, error_tracker
def risky(): raise ValueError('test')
result = safe_execute(risky, default='OK')
print(f'   Safe execute: {result}')
print()

print('=' * 50)
print('ALL MODULES WORKING!')
print('=' * 50)
