import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import time

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = "8193800020:AAEbM9jKBiKhCifVOGcvsavSqEDZ0K77tAs"

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("✅ Bot is working!")

def main():
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start_command))
    
    logger.info("Starting minimal test bot...")
    
    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        logger.info("Bot stopped")

if __name__ == '__main__':
    main()
