"""
Minimal Telegram bot test to verify polling works
"""
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import logging

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Bot is working!')
    logger.info(f"Received /start from {update.effective_user.id}")

def main():
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    logger.info(f"Starting minimal bot with token: {token[:10]}...")
    
    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    
    logger.info("Starting polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Polling stopped")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
