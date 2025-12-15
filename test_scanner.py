import ccxt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test scanner
exchange = ccxt.binance()

try:
    logger.info("Loading markets...")
    markets = exchange.load_markets()
    logger.info(f"Loaded {len(markets)} markets")
    
    usdt_pairs = [symbol for symbol in markets if '/USDT' in symbol and markets[symbol].get('active', False)]
    logger.info(f"Found {len(usdt_pairs)} USDT pairs")
    
    # Try to fetch first ticker
    if usdt_pairs:
        symbol = usdt_pairs[0]
        logger.info(f"Fetching ticker for {symbol}...")
        ticker = exchange.fetch_ticker(symbol)
        logger.info(f"Success! {symbol}: ${ticker['last']}, Vol: ${ticker.get('quoteVolume', 0):,.0f}")
        
except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
