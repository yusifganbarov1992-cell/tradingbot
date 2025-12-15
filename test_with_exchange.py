"""
Test bot with TradingAgent to find what stops polling
"""
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import logging
import ccxt

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Test ccxt exchange
logger.info("Creating ccxt exchange...")
exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_API_SECRET'),
    'enableRateLimit': True
})
logger.info(f"Exchange created: {exchange.id}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Bot with exchange working!')
    logger.info(f"Received /start")

def main():
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    logger.info(f"Starting bot with ccxt exchange...")
    
    application = Application.builder().token(token).build()
    application.bot_data['exchange'] = exchange
    application.add_handler(CommandHandler("start", start))
    
    logger.info("Starting polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Polling stopped")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
