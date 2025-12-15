"""
Auto-restart wrapper for trading bot
Handles the telegram-bot v22 stop issue by auto-restarting
"""
import subprocess
import time
import logging
from datetime import datetime

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def run_bot():
    """Run bot and return exit code"""
    logger.info("Starting trading bot...")
    process = subprocess.Popen(
        ['.venv\\Scripts\\python.exe', 'trading_bot.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream output
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def main():
    logger.info("=" * 60)
    logger.info("NEXUSTRADER - AUTO-RESTART WRAPPER")
    logger.info("Bot will automatically restart every ~10 seconds")
    logger.info("This fixes python-telegram-bot v22 stop issue")
    logger.info("Press Ctrl+C to stop completely")
    logger.info("=" * 60)
    
    restart_count = 0
    
    try:
        while True:
            restart_count += 1
            logger.info(f"\\n=== BOT START #{restart_count} at {datetime.now()} ===")
            
            exit_code = run_bot()
            
            logger.info(f"=== BOT STOPPED (exit code: {exit_code}) ===")
            
            if exit_code != 0:
                logger.warning(f"Bot crashed with code {exit_code}")
            
            logger.info("Restarting in 2 seconds...")
            time.sleep(2)
            
    except KeyboardInterrupt:
        logger.info("\\n\\nShutdown requested by user")
        logger.info(f"Total restarts: {restart_count}")
        logger.info("Goodbye!")

if __name__ == '__main__':
    main()
