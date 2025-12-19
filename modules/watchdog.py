"""
Auto-Restart Watchdog - Keeps bot running 24/7
Monitors process and restarts on crash
"""

import os
import sys
import time
import subprocess
import logging
from datetime import datetime, timedelta
import signal
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('watchdog.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BotWatchdog:
    """
    Watchdog for NexusTrader Bot
    - Monitors bot process
    - Auto-restarts on crash
    - Sends Telegram alerts
    - Handles graceful shutdown
    """
    
    def __init__(self):
        self.process = None
        self.running = False
        self.restart_count = 0
        self.max_restarts = 10
        self.restart_cooldown = 60  # seconds between restarts
        self.last_restart = None
        
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Handle signals
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def send_telegram(self, message: str):
        """Send Telegram notification"""
        if not self.telegram_token or not self.telegram_chat_id:
            return
        
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            requests.post(url, json={
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }, timeout=10)
        except Exception as e:
            logger.error(f"Failed to send Telegram: {e}")
    
    def start_bot(self):
        """Start the trading bot process"""
        try:
            # Use venv python if available
            python_path = sys.executable
            bot_path = os.path.join(os.path.dirname(__file__), '..', 'trading_bot.py')
            bot_path = os.path.abspath(bot_path)
            
            logger.info(f"Starting bot: {python_path} {bot_path}")
            
            self.process = subprocess.Popen(
                [python_path, bot_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(bot_path)
            )
            
            self.last_restart = datetime.now()
            self.restart_count += 1
            
            logger.info(f"Bot started with PID {self.process.pid}")
            self.send_telegram(f"🤖 *NexusTrader Started*\n\nPID: {self.process.pid}\nRestart #{self.restart_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start bot: {e}")
            self.send_telegram(f"🔴 *Failed to Start Bot*\n\nError: {str(e)[:100]}")
            return False
    
    def stop_bot(self):
        """Stop the trading bot process"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
                logger.info("Bot stopped gracefully")
            except subprocess.TimeoutExpired:
                self.process.kill()
                logger.warning("Bot killed forcefully")
            except Exception as e:
                logger.error(f"Error stopping bot: {e}")
            
            self.process = None
    
    def is_bot_running(self) -> bool:
        """Check if bot process is running"""
        if self.process is None:
            return False
        
        return self.process.poll() is None
    
    def check_and_restart(self):
        """Check bot status and restart if needed"""
        if self.is_bot_running():
            return True
        
        # Bot crashed
        exit_code = self.process.returncode if self.process else -1
        logger.warning(f"Bot stopped with exit code {exit_code}")
        
        # Check restart limits
        if self.restart_count >= self.max_restarts:
            logger.error("Max restart limit reached!")
            self.send_telegram(f"🔴 *Max Restarts Reached*\n\nBot crashed {self.max_restarts} times.\nManual intervention required!")
            return False
        
        # Check cooldown
        if self.last_restart:
            elapsed = (datetime.now() - self.last_restart).total_seconds()
            if elapsed < self.restart_cooldown:
                wait_time = self.restart_cooldown - elapsed
                logger.info(f"Waiting {wait_time:.0f}s before restart...")
                time.sleep(wait_time)
        
        # Restart
        self.send_telegram(f"⚠️ *Bot Crashed*\n\nExit code: {exit_code}\nRestarting... ({self.restart_count + 1}/{self.max_restarts})")
        return self.start_bot()
    
    def run(self):
        """Main watchdog loop"""
        logger.info("="*50)
        logger.info("NEXUSTRADER WATCHDOG STARTED")
        logger.info("="*50)
        
        self.running = True
        
        # Initial start
        if not self.start_bot():
            logger.error("Failed to start bot initially")
            return
        
        # Monitor loop
        while self.running:
            try:
                if not self.check_and_restart():
                    logger.error("Could not restart bot, stopping watchdog")
                    break
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                time.sleep(30)
        
        self.stop_bot()
        logger.info("Watchdog stopped")
    
    def stop(self):
        """Stop watchdog"""
        self.running = False
        self.stop_bot()
        self.send_telegram("🛑 *NexusTrader Stopped*")


def main():
    """Run watchdog"""
    watchdog = BotWatchdog()
    watchdog.run()


if __name__ == "__main__":
    main()
