"""
Health Monitor - Bot health checks with Telegram alerts
FREE - Uses only Telegram (already configured)
"""

import os
import sys
import time
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, List
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health check status"""
    component: str
    status: str  # 'healthy', 'warning', 'critical'
    message: str
    last_check: datetime
    details: Optional[Dict] = None


class HealthMonitor:
    """
    Health Monitor for Trading Bot
    - Checks all components
    - Sends Telegram alerts on issues
    - Auto-recovery attempts
    """
    
    def __init__(self):
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.checks: Dict[str, HealthStatus] = {}
        self.alert_cooldown: Dict[str, datetime] = {}
        self.cooldown_minutes = 30  # Don't spam alerts
        
        # Thresholds
        self.max_api_latency_ms = 5000
        self.min_balance_usd = 10
        self.max_open_positions = 10
        self.max_daily_loss_pct = 5
        
    async def send_telegram_alert(self, message: str, level: str = 'warning'):
        """Send alert via Telegram"""
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("Telegram not configured")
            return
        
        try:
            import aiohttp
            
            emoji = {
                'info': 'ℹ️',
                'warning': '⚠️',
                'critical': '🚨',
                'success': '✅'
            }.get(level, '📢')
            
            text = f"{emoji} *NexusTrader Alert*\n\n{message}"
            
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': text,
                'parse_mode': 'Markdown'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        logger.error(f"Telegram alert failed: {await resp.text()}")
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
    
    def should_alert(self, component: str) -> bool:
        """Check if we should send alert (cooldown)"""
        if component not in self.alert_cooldown:
            return True
        
        last_alert = self.alert_cooldown[component]
        if datetime.now() - last_alert > timedelta(minutes=self.cooldown_minutes):
            return True
        
        return False
    
    def mark_alerted(self, component: str):
        """Mark component as alerted"""
        self.alert_cooldown[component] = datetime.now()
    
    async def check_exchange_connection(self) -> HealthStatus:
        """Check Binance API connection"""
        component = 'exchange'
        try:
            import ccxt
            
            start = time.time()
            exchange = ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_SECRET_KEY'),
            })
            
            # Test connection
            balance = exchange.fetch_balance()
            latency = (time.time() - start) * 1000
            
            if latency > self.max_api_latency_ms:
                status = HealthStatus(
                    component=component,
                    status='warning',
                    message=f'High API latency: {latency:.0f}ms',
                    last_check=datetime.now(),
                    details={'latency_ms': latency}
                )
            else:
                status = HealthStatus(
                    component=component,
                    status='healthy',
                    message=f'Connected ({latency:.0f}ms)',
                    last_check=datetime.now(),
                    details={'latency_ms': latency}
                )
            
        except Exception as e:
            status = HealthStatus(
                component=component,
                status='critical',
                message=f'Connection failed: {str(e)[:50]}',
                last_check=datetime.now()
            )
            
            if self.should_alert(component):
                await self.send_telegram_alert(
                    f"🔴 *Exchange Connection Failed*\n\nError: {str(e)[:100]}\n\nBot may not be able to trade!",
                    level='critical'
                )
                self.mark_alerted(component)
        
        self.checks[component] = status
        return status
    
    async def check_balance(self) -> HealthStatus:
        """Check trading balance"""
        component = 'balance'
        try:
            import ccxt
            
            exchange = ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_SECRET_KEY'),
            })
            
            balance = exchange.fetch_balance()
            
            # Calculate free USD
            free_usd = 0
            for cur in ['USDT', 'USDC', 'BUSD', 'FDUSD']:
                free_usd += balance['free'].get(cur, 0)
            
            if free_usd < self.min_balance_usd:
                status = HealthStatus(
                    component=component,
                    status='warning',
                    message=f'Low balance: ${free_usd:.2f}',
                    last_check=datetime.now(),
                    details={'free_usd': free_usd}
                )
                
                if self.should_alert(component):
                    await self.send_telegram_alert(
                        f"💰 *Low Trading Balance*\n\nFree: ${free_usd:.2f}\nMinimum: ${self.min_balance_usd}\n\nBot may not be able to open new positions.",
                        level='warning'
                    )
                    self.mark_alerted(component)
            else:
                status = HealthStatus(
                    component=component,
                    status='healthy',
                    message=f'Balance OK: ${free_usd:.2f}',
                    last_check=datetime.now(),
                    details={'free_usd': free_usd}
                )
        
        except Exception as e:
            status = HealthStatus(
                component=component,
                status='critical',
                message=f'Balance check failed: {str(e)[:50]}',
                last_check=datetime.now()
            )
        
        self.checks[component] = status
        return status
    
    async def check_database(self) -> HealthStatus:
        """Check Supabase connection"""
        component = 'database'
        try:
            from database_supabase import SupabaseDatabase
            
            start = time.time()
            db = SupabaseDatabase()
            
            # Test query
            trades = db.get_trade_history(limit=1)
            latency = (time.time() - start) * 1000
            
            status = HealthStatus(
                component=component,
                status='healthy',
                message=f'Connected ({latency:.0f}ms)',
                last_check=datetime.now(),
                details={'latency_ms': latency}
            )
            
        except Exception as e:
            status = HealthStatus(
                component=component,
                status='critical',
                message=f'Database error: {str(e)[:50]}',
                last_check=datetime.now()
            )
            
            if self.should_alert(component):
                await self.send_telegram_alert(
                    f"🗄️ *Database Connection Failed*\n\nError: {str(e)[:100]}\n\nTrade history may not be saved!",
                    level='critical'
                )
                self.mark_alerted(component)
        
        self.checks[component] = status
        return status
    
    async def check_open_positions(self) -> HealthStatus:
        """Check open positions count"""
        component = 'positions'
        try:
            from database_supabase import SupabaseDatabase
            
            db = SupabaseDatabase()
            trades = db.get_trade_history(limit=100)
            open_positions = [t for t in trades if t.get('status') == 'open']
            
            if len(open_positions) > self.max_open_positions:
                status = HealthStatus(
                    component=component,
                    status='warning',
                    message=f'Too many positions: {len(open_positions)}',
                    last_check=datetime.now(),
                    details={'count': len(open_positions)}
                )
                
                if self.should_alert(component):
                    await self.send_telegram_alert(
                        f"📊 *Too Many Open Positions*\n\nOpen: {len(open_positions)}\nMax: {self.max_open_positions}\n\nConsider closing some positions.",
                        level='warning'
                    )
                    self.mark_alerted(component)
            else:
                status = HealthStatus(
                    component=component,
                    status='healthy',
                    message=f'{len(open_positions)} open positions',
                    last_check=datetime.now(),
                    details={'count': len(open_positions)}
                )
        
        except Exception as e:
            status = HealthStatus(
                component=component,
                status='warning',
                message=f'Check failed: {str(e)[:50]}',
                last_check=datetime.now()
            )
        
        self.checks[component] = status
        return status
    
    async def check_daily_pnl(self) -> HealthStatus:
        """Check daily P&L"""
        component = 'daily_pnl'
        try:
            from database_supabase import SupabaseDatabase
            
            db = SupabaseDatabase()
            trades = db.get_trade_history(limit=100)
            
            # Filter today's closed trades
            today = datetime.now().date()
            today_trades = [
                t for t in trades 
                if t.get('status') == 'closed' and 
                t.get('exit_time', '')[:10] == str(today)
            ]
            
            daily_pnl = sum(t.get('pnl', 0) or 0 for t in today_trades)
            daily_pnl_pct = sum(t.get('pnl_percent', 0) or 0 for t in today_trades)
            
            if daily_pnl_pct < -self.max_daily_loss_pct:
                status = HealthStatus(
                    component=component,
                    status='critical',
                    message=f'Daily loss: {daily_pnl_pct:.2f}%',
                    last_check=datetime.now(),
                    details={'pnl': daily_pnl, 'pnl_pct': daily_pnl_pct}
                )
                
                if self.should_alert(component):
                    await self.send_telegram_alert(
                        f"📉 *Daily Loss Limit Reached*\n\nDaily P&L: ${daily_pnl:.2f} ({daily_pnl_pct:.2f}%)\nLimit: -{self.max_daily_loss_pct}%\n\n⚠️ Consider pausing trading!",
                        level='critical'
                    )
                    self.mark_alerted(component)
            else:
                emoji = "📈" if daily_pnl >= 0 else "📉"
                status = HealthStatus(
                    component=component,
                    status='healthy',
                    message=f'{emoji} Today: ${daily_pnl:.2f} ({daily_pnl_pct:+.2f}%)',
                    last_check=datetime.now(),
                    details={'pnl': daily_pnl, 'pnl_pct': daily_pnl_pct}
                )
        
        except Exception as e:
            status = HealthStatus(
                component=component,
                status='warning',
                message=f'Check failed: {str(e)[:50]}',
                last_check=datetime.now()
            )
        
        self.checks[component] = status
        return status
    
    async def run_all_checks(self) -> Dict[str, HealthStatus]:
        """Run all health checks"""
        await self.check_exchange_connection()
        await self.check_balance()
        await self.check_database()
        await self.check_open_positions()
        await self.check_daily_pnl()
        
        return self.checks
    
    def get_overall_status(self) -> str:
        """Get overall system status"""
        if any(c.status == 'critical' for c in self.checks.values()):
            return 'critical'
        if any(c.status == 'warning' for c in self.checks.values()):
            return 'warning'
        return 'healthy'
    
    def print_status(self):
        """Print health status"""
        print("\n" + "="*50)
        print("🏥 HEALTH MONITOR STATUS")
        print("="*50)
        
        for name, check in self.checks.items():
            emoji = {
                'healthy': '✅',
                'warning': '⚠️',
                'critical': '🔴'
            }.get(check.status, '❓')
            
            print(f"{emoji} {name.upper()}: {check.message}")
        
        overall = self.get_overall_status()
        print("="*50)
        print(f"Overall: {overall.upper()}")
        print("="*50)


class HealthMonitorDaemon:
    """Background health monitoring daemon"""
    
    def __init__(self, check_interval: int = 300):  # 5 minutes
        self.monitor = HealthMonitor()
        self.check_interval = check_interval
        self.running = False
        self.thread = None
    
    def start(self):
        """Start monitoring daemon"""
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("Health monitor daemon started")
    
    def stop(self):
        """Stop monitoring daemon"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Health monitor daemon stopped")
    
    def _run_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                asyncio.run(self.monitor.run_all_checks())
                self.monitor.print_status()
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            time.sleep(self.check_interval)


async def main():
    """Run health checks"""
    monitor = HealthMonitor()
    await monitor.run_all_checks()
    monitor.print_status()
    
    # Send startup notification
    await monitor.send_telegram_alert(
        "🤖 *NexusTrader Health Check*\n\n" + 
        "\n".join([f"• {c.component}: {c.status}" for c in monitor.checks.values()]),
        level='info'
    )


if __name__ == "__main__":
    asyncio.run(main())
