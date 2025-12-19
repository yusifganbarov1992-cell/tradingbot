import os
from dotenv import load_dotenv
import ccxt
import ccxt.async_support as ccxt_async
import logging
import asyncio
from telegram import Update, ForceReply, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
import numpy as np
import pandas as pd
# Lazy import OpenAI - only when needed
# from openai import OpenAI
import time
from datetime import datetime, timedelta
from database import TradingDatabase
from database_supabase import SupabaseDatabase

# Try to import TensorFlow, but make it optional
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. AI predictions will be disabled.")

# Lazy import sklearn - only when AI model is actually used
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AI clients only when needed (lazy initialization)
openai_client = None
deepseek_client = None

def get_openai_client():
    from openai import OpenAI
    global openai_client
    if openai_client is None:
        try:
            openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            logger.info("OpenAI client initialized")
        except Exception as e:
            logger.warning(f"OpenAI initialization failed: {e}")
            openai_client = False
    return openai_client if openai_client else None

def get_deepseek_client():
    from openai import OpenAI
    global deepseek_client
    if deepseek_client is None:
        try:
            deepseek_client = OpenAI(
                api_key=os.getenv('DEEPSEEK_API_KEY'),
                base_url="https://api.deepseek.com"
            )
            logger.info("DeepSeek client initialized")
        except Exception as e:
            logger.warning(f"DeepSeek initialization failed: {e}")
            deepseek_client = False
    return deepseek_client if deepseek_client else None

# Cache for AI analysis (to save tokens) - per symbol
ai_analysis_cache = {}
# Cache duration per symbol
AI_CACHE_DURATION = 300  # 5 minutes cache per symbol (was 3 min)
MAX_CACHE_SIZE = 50  # Maximum symbols in cache

def clean_ai_cache():
    """LRU cleanup: remove oldest entries if cache too large"""
    global ai_analysis_cache
    if len(ai_analysis_cache) > MAX_CACHE_SIZE:
        # Sort by timestamp, keep newest 30
        sorted_items = sorted(ai_analysis_cache.items(), key=lambda x: x[1]['timestamp'], reverse=True)
        ai_analysis_cache = dict(sorted_items[:30])
        logger.info(f"🧹 AI cache cleaned: {len(sorted_items)} → 30 entries")

# Token usage tracker
total_tokens_used = 0
total_ai_calls = 0

# --- AI Model Placeholder ---
class AIModel:
    def __init__(self, model_path='lstm_model.keras'):
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self._load_model_and_scaler()

    def _load_model_and_scaler(self):
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. AI model disabled.")
            return
        
        from sklearn.preprocessing import MinMaxScaler
            
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"LSTM model loaded from {self.model_path}")
                # Placeholder for scaler loading - in a real scenario, the scaler
                # would also need to be saved and loaded (e.g., using joblib or pickle)
                self.scaler = MinMaxScaler(feature_range=(0, 1)) # Re-initialize for demonstration
                # You'd typically load the fitted scaler here
            except Exception as e:
                logger.error(f"Error loading model from {self.model_path}: {e}")
                self.model = None
        else:
            logger.warning(f"No LSTM model found at {self.model_path}. Model needs to be trained.")

    def preprocess_data(self, data: pd.DataFrame):
        from sklearn.preprocessing import MinMaxScaler
        # This is a basic preprocessing. In a real scenario, you'd fit the scaler
        # on training data and transform both training and inference data.
        if self.scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = self.scaler.fit_transform(data)
        else:
            scaled_data = self.scaler.transform(data)
        return scaled_data

    def create_dataset(self, data, time_step=60):
        X = []
        for i in range(len(data) - time_step - 1):
            a = data[i:(i + time_step), 0]
            X.append(a)
        return np.array(X)

    def predict(self, input_data: np.ndarray):
        if self.model:
            # Reshape input for LSTM: [samples, time_steps, features]
            input_data = input_data.reshape(1, input_data.shape[0], 1)
            prediction = self.model.predict(input_data)
            # Inverse transform the prediction to original scale
            if self.scaler:
                # Create a dummy array to inverse transform. The scaler expects
                # a 2D array, so we add a second dimension.
                dummy_array = np.zeros((1, self.scaler.n_features_in_))
                dummy_array[0, 0] = prediction[0, 0] # Assuming prediction is for the first feature
                prediction = self.scaler.inverse_transform(dummy_array)[:,0]
            return prediction[0]
        return None

# --- Model Training Function ---
async def train_model(exchange: ccxt.Exchange, symbol: str, timeframe='1h', limit=1000, model_path='lstm_model.keras'):
    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow not available. Model training skipped.")
        return False
    
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
        
    logger.info(f"Starting model training for {symbol}...")
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        data = df['close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Create training dataset
        time_step = 60 # Look back 60 hours
        X, y = [], []
        for i in range(time_step, len(scaled_data)):
            X.append(scaled_data[i-time_step:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)

        # Reshape for LSTM [samples, time_steps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

        model.save(model_path)
        logger.info(f"LSTM model trained and saved to {model_path}")
        # In a real application, you'd also save the scaler for consistent preprocessing
        return True
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return False

# --- OpenAI Smart Analysis (Token-Efficient) ---
def get_ai_trading_advice(market_data: dict, filters_data: dict, use_cache: bool = True, max_retries: int = 2) -> str:
    """
    Get intelligent trading advice from AI with fallback support
    Primary: OpenAI GPT-4o-mini
    Fallback: DeepSeek
    Uses per-symbol caching and retry logic
    
    Args:
        market_data: Dict with symbol and price
        filters_data: Dict with RSI, trend, volume data
        use_cache: Whether to use cached result
        max_retries: Max retry attempts on error
    
    Returns:
        str: "SIGNAL|CONFIDENCE|REASON" format
    """
    global ai_analysis_cache
    
    symbol = market_data['symbol']
    
    # Check per-symbol cache
    if use_cache and symbol in ai_analysis_cache:
        cache_entry = ai_analysis_cache[symbol]
        time_since_cache = (datetime.now() - cache_entry['timestamp']).seconds
        if time_since_cache < AI_CACHE_DURATION:
            logger.info(f"Using cached AI for {symbol} ({time_since_cache}s old)")
            return cache_entry['analysis']
    
    # Ultra-concise prompt (40% shorter)
    prompt = f"""{market_data['symbol']} P:${market_data['price']:.0f} RSI:{filters_data['rsi']:.0f} {'↑' if filters_data['ema_bullish'] else '↓'} Vol:{'H' if filters_data['volume_spike'] else 'N'} F:{filters_data['buy_count']}B/{filters_data['sell_count']}S\nSignal|1-10|reason(10w max):"""

    messages = [
        {"role": "system", "content": "Concise crypto analyst. Format: SIGNAL|NUM|reason"},
        {"role": "user", "content": prompt}
    ]
    
    # Try OpenAI first
    client = get_openai_client()
    if client:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Cheaper model
                messages=messages,
                max_tokens=30,  # 30 tokens (was 50) - 40% reduction
                temperature=0.3,  # Lower temperature for consistency
                timeout=10  # 10 second timeout to prevent hanging
            )
            
            result = response.choices[0].message.content.strip()
            
            # Track token usage
            global total_tokens_used, total_ai_calls
            total_tokens_used += response.usage.total_tokens
            total_ai_calls += 1
            
            # Cache per symbol (with size limit)
            if len(ai_analysis_cache) >= MAX_CACHE_SIZE:
                # Remove oldest entry
                oldest = min(ai_analysis_cache.items(), key=lambda x: x[1]['timestamp'])
                del ai_analysis_cache[oldest[0]]
                logger.debug(f"Cache full, removed {oldest[0]}")
            
            ai_analysis_cache[symbol] = {
                'analysis': result,
                'timestamp': datetime.now()
            }
            
            avg_tokens = total_tokens_used / total_ai_calls if total_ai_calls > 0 else 0
            logger.info(f"OpenAI: {result} | tokens:{response.usage.total_tokens} (avg:{avg_tokens:.0f}, total:{total_tokens_used})")
            return result
            
        except Exception as e:
            logger.warning(f"OpenAI error: {e}")
            if max_retries > 0:
                logger.info(f"Retrying OpenAI ({max_retries} attempts left)...")
                time.sleep(1)
                return get_ai_trading_advice(market_data, filters_data, use_cache=False, max_retries=max_retries-1)
            logger.warning("OpenAI failed, trying DeepSeek fallback...")
    
    # Fallback to DeepSeek
    deepseek = get_deepseek_client()
    if deepseek:
        try:
            response = deepseek.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                max_tokens=50,
                temperature=0.3,
                timeout=10
            )
            
            result = response.choices[0].message.content.strip()
            
            # Cache per symbol (with size limit)
            if len(ai_analysis_cache) >= MAX_CACHE_SIZE:
                oldest = min(ai_analysis_cache.items(), key=lambda x: x[1]['timestamp'])
                del ai_analysis_cache[oldest[0]]
            
            ai_analysis_cache[symbol] = {
                'analysis': result,
                'timestamp': datetime.now()
            }
            
            logger.info(f"DeepSeek (fallback): {result}")
            return result
            
        except Exception as e:
            logger.error(f"DeepSeek error: {e}")
    
    logger.error(f"All AI providers failed for {symbol}")
    return "WAIT|5|AI unavailable"

# --- Risk Management Engine (from Nautilus Trader) ---
class RiskEngine:
    def __init__(self, max_position_size_pct=10, max_total_exposure_pct=30):
        self.max_position_size_pct = max_position_size_pct  # Max 10% per position
        self.max_total_exposure_pct = max_total_exposure_pct  # Max 30% total exposure
        self.binance_fee = 0.001  # 0.1% Binance fee (maker/taker)
    
    def calculate_position_size(self, balance_usdt, price, signal_strength=6):
        """Calculate position size based on balance and risk"""
        # OPTIMIZED from backtesting: 15% base size (was 5%)
        base_pct = 0.15
        strength_multiplier = min(signal_strength / 6.0, 1.2)  # Max 1.2x for strong signals
        position_pct = min(base_pct * strength_multiplier, self.max_position_size_pct / 100)
        
        # Calculate amount in USDT
        usdt_amount = balance_usdt * position_pct
        
        # Calculate crypto amount (accounting for fees)
        crypto_amount = (usdt_amount * (1 - self.binance_fee)) / price
        
        return crypto_amount, usdt_amount
    
    def calculate_fees(self, usdt_amount):
        """Calculate Binance trading fees"""
        return usdt_amount * self.binance_fee

# --- Safety Manager (8-Level Protection) ---
class SafetyManager:
    """
    8-УРОВНЕВАЯ СИСТЕМА БЕЗОПАСНОСТИ
    Защищает от убытков и неправильных решений AI
    """
    
    def __init__(self, initial_balance, db=None):
        self.initial_balance = initial_balance
        self.daily_loss_limit_pct = 3.0  # Max -3% потерь за день
        self.max_trades_per_day = 5
        self.max_position_size_pct = 15.0
        self.min_confidence = 7  # AI confidence ≥7/10
        self.max_price_change_5min = 2.0  # Max 2% изменение цены за 5 мин
        
        # Загрузка emergency_stop из БД (персистентность)
        self.db = db
        if self.db:
            self.emergency_stop = self.db.load_emergency_stop()
            logger.info(f"🛡️ Emergency stop loaded from DB: {self.emergency_stop}")
        else:
            self.emergency_stop = False
        
        self.paused = False
        
        # Статистика за день
        self.daily_trades = []
        self.daily_pnl = 0.0
        self.last_reset = datetime.now().date()
        
        logger.info("🛡️ SafetyManager initialized with 8-level protection")
    
    def reset_daily_stats(self):
        """Сброс дневной статистики в полночь"""
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_trades = []
            self.daily_pnl = 0.0
            self.last_reset = today
            logger.info("📊 Daily stats reset")
    
    def check_all_safety_levels(self, signal_data, current_balance, active_positions, recent_prices) -> tuple[bool, str]:
        """
        Проверка всех 8 уровней безопасности
        Returns: (is_safe, reason)
        """
        self.reset_daily_stats()
        
        # LEVEL 1: Emergency Stop
        if self.emergency_stop:
            return False, "🚨 EMERGENCY STOP ACTIVATED"
        
        # LEVEL 2: Paused
        if self.paused:
            return False, "⏸️ Trading paused by user"
        
        # LEVEL 3: Daily Loss Limit (-3%)
        daily_loss_pct = (self.daily_pnl / self.initial_balance) * 100
        if daily_loss_pct < -self.daily_loss_limit_pct:
            return False, f"📉 Daily loss limit reached: {daily_loss_pct:.2f}% (max -3%)"
        
        # LEVEL 4: Max Trades Per Day (5)
        if len(self.daily_trades) >= self.max_trades_per_day:
            return False, f"🔢 Daily trade limit reached: {len(self.daily_trades)}/{self.max_trades_per_day}"
        
        # LEVEL 5: AI Confidence Threshold (≥7/10)
        if signal_data.get('confidence', 0) < self.min_confidence:
            return False, f"🤖 AI confidence too low: {signal_data.get('confidence')}/10 (min {self.min_confidence})"
        
        # LEVEL 6: No Duplicate Positions
        symbol = signal_data.get('symbol')
        if symbol in [pos['symbol'] for pos in active_positions]:
            return False, f"⚠️ Already have position in {symbol}"
        
        # LEVEL 7: Price Volatility Check (max 2% за 5 мин)
        if recent_prices and len(recent_prices) >= 2:
            price_change_pct = abs((recent_prices[-1] - recent_prices[0]) / recent_prices[0]) * 100
            if price_change_pct > self.max_price_change_5min:
                return False, f"📊 Price volatility too high: {price_change_pct:.2f}% in 5min (max {self.max_price_change_5min}%)"
        
        # LEVEL 8: Balance Drawdown Limit (max -10% от стартового)
        total_drawdown_pct = ((current_balance - self.initial_balance) / self.initial_balance) * 100
        if total_drawdown_pct < -10.0:
            logger.critical(f"💀 CRITICAL: Total drawdown {total_drawdown_pct:.2f}% - EMERGENCY STOP!")
            self.emergency_stop = True
            return False, f"💀 Total drawdown limit: {total_drawdown_pct:.2f}% (max -10%)"
        
        # ✅ ALL SAFETY CHECKS PASSED
        return True, "✅ All 8 safety levels passed"
    
    def record_trade(self, pnl):
        """Записать сделку в дневную статистику"""
        self.reset_daily_stats()
        self.daily_trades.append({
            'timestamp': datetime.now(),
            'pnl': pnl
        })
        self.daily_pnl += pnl
        logger.info(f"📊 Trade recorded: PnL ${pnl:.2f}, Daily total: ${self.daily_pnl:.2f}")
    
    def activate_emergency_stop(self):
        """Активировать экстренную остановку"""
        self.emergency_stop = True
        if self.db:
            self.db.save_emergency_stop(True)
        logger.critical("🚨 EMERGENCY STOP ACTIVATED!")
    
    def deactivate_emergency_stop(self):
        """Снять экстренную остановку"""
        self.emergency_stop = False
        if self.db:
            self.db.save_emergency_stop(False)
        logger.info("✅ Emergency stop deactivated")
    
    def pause_trading(self):
        """Приостановить торговлю"""
        self.paused = True
        logger.info("⏸️ Trading paused")
    
    def resume_trading(self):
        """Возобновить торговлю"""
        self.paused = False
        logger.info("▶️ Trading resumed")
    
    def get_status(self) -> str:
        """Получить статус защиты"""
        self.reset_daily_stats()
        status = "🛡️ SAFETY STATUS:\n\n"
        
        if self.emergency_stop:
            status += "🚨 EMERGENCY STOP: ACTIVE\n"
        elif self.paused:
            status += "⏸️ Status: PAUSED\n"
        else:
            status += "✅ Status: ACTIVE\n"
        
        status += f"\n📊 Daily Stats:\n"
        status += f"Trades: {len(self.daily_trades)}/{self.max_trades_per_day}\n"
        status += f"P&L: ${self.daily_pnl:.2f}\n"
        
        daily_loss_pct = (self.daily_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        status += f"Daily %: {daily_loss_pct:+.2f}% (limit: -{self.daily_loss_limit_pct}%)\n"
        
        status += f"\n🛡️ Protection Levels:\n"
        status += f"1. Emergency Stop: {'🚨 ON' if self.emergency_stop else '✅ OFF'}\n"
        status += f"2. Pause: {'⏸️ ON' if self.paused else '✅ OFF'}\n"
        status += f"3. Daily Loss: {daily_loss_pct:.2f}% / -{self.daily_loss_limit_pct}%\n"
        status += f"4. Trades Today: {len(self.daily_trades)}/{self.max_trades_per_day}\n"
        status += f"5. Min AI Confidence: {self.min_confidence}/10\n"
        status += f"6. No Duplicates: ✅\n"
        status += f"7. Max Volatility: {self.max_price_change_5min}% / 5min\n"
        status += f"8. Max Drawdown: -10%\n"
        
        return status

# --- Metrics Tracker (from Jesse) ---
class MetricsTracker:
    def __init__(self):
        self.trades = []  # All completed trades
        self.total_profit = 0.0
        self.total_fees = 0.0
    
    def add_trade(self, symbol, side, entry_price, exit_price, amount, fee):
        """Record completed trade"""
        if side == 'BUY':
            profit = (exit_price - entry_price) * amount - fee
        else:
            profit = (entry_price - exit_price) * amount - fee
        
        trade = {
            'symbol': symbol,
            'side': side,
            'entry': entry_price,
            'exit': exit_price,
            'amount': amount,
            'profit': profit,
            'fee': fee,
            'timestamp': datetime.now()
        }
        self.trades.append(trade)
        self.total_profit += profit
        self.total_fees += fee
    
    def get_win_rate(self):
        """Calculate win rate %"""
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t['profit'] > 0)
        return (wins / len(self.trades)) * 100
    
    def get_sharpe_ratio(self):
        """Calculate Sharpe Ratio (simplified)"""
        if len(self.trades) < 2:
            return 0.0
        profits = [t['profit'] for t in self.trades]
        avg_profit = np.mean(profits)
        std_profit = np.std(profits)
        if std_profit == 0:
            return 0.0
        return avg_profit / std_profit
    
    def get_summary(self):
        """Get metrics summary"""
        return {
            'total_trades': len(self.trades),
            'win_rate': self.get_win_rate(),
            'total_profit': self.total_profit,
            'total_fees': self.total_fees,
            'sharpe_ratio': self.get_sharpe_ratio(),
            'net_profit': self.total_profit - self.total_fees
        }

# --- Trading Agent Class ---
class TradingAgent:
    def __init__(self):
        self.binance_api_key = os.getenv('BINANCE_API_KEY')
        self.binance_secret_key = os.getenv('BINANCE_SECRET_KEY')
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.operator_chat_id = os.getenv('OPERATOR_CHAT_ID', '5150355926')  # Fallback для обратной совместимости
        
        # Use sync exchange for market analysis (runs in separate thread)
        self.exchange = ccxt.binance({
            'apiKey': self.binance_api_key,
            'secret': self.binance_secret_key,
            'enableRateLimit': True,
        })
        self.ai_model = AIModel()
        
        # Multi-symbol support (autoscanner will populate this)
        self.symbols = []  # Will be filled by scanner
        self.active_positions = {}  # Track open positions with trailing stops
        
        self.trade_confirmation_needed = {} # To store pending trade confirmations
        
        # Professional components
        self.risk_engine = RiskEngine(max_position_size_pct=10, max_total_exposure_pct=30)
        self.metrics = MetricsTracker()
        self.db = TradingDatabase()  # Initialize database
        
        # 🌐 Supabase - облачное хранилище (опционально)
        try:
            self.supabase_db = SupabaseDatabase()
            logger.info("☁️ Supabase подключена (двойное хранение активно)")
        except Exception as e:
            self.supabase_db = None
            logger.warning(f"⚠️ Supabase не подключена: {e}. Используется только SQLite")
        
        # 🛡️ SAFETY MANAGER - 8-level protection
        initial_balance = 1000.0  # Will update from real balance
        try:
            balance = self.exchange.fetch_balance()
            initial_balance = balance['USDT']['free'] + balance['USDT']['used']
            if initial_balance <= 0:
                initial_balance = 1000.0  # Fallback
        except:
            logger.warning("Could not fetch initial balance, using $1000")
        
        self.safety = SafetyManager(initial_balance, db=self.db)
        logger.info(f"🛡️ Safety initialized with balance: ${initial_balance:.2f}")
        
        # Trading mode (из .env для безопасности)
        self.paper_trading = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
        if not self.paper_trading:
            logger.warning("⚠️  REAL TRADING MODE ENABLED! Be careful!")
        else:
            logger.info("✅ Paper trading mode (safe)")
        
        # 🤖 AUTO_TRADE MODE - умный автономный режим
        from modules.autonomous_trader import AutonomousTrader
        self.autonomous = AutonomousTrader(
            auto_trade_enabled=os.getenv('AUTO_TRADE', 'false').lower() == 'true',
            min_confidence=float(os.getenv('AUTO_MIN_CONFIDENCE', '7.0')),
            max_trades_per_hour=int(os.getenv('AUTO_MAX_TRADES_HOUR', '3')),
            max_concurrent_positions=int(os.getenv('AUTO_MAX_POSITIONS', '5')),
            whitelist=os.getenv('AUTO_WHITELIST', '').split(',') if os.getenv('AUTO_WHITELIST') else [],
            blacklist=os.getenv('AUTO_BLACKLIST', 'LUNA,FTT,USTC').split(',')
        )
        logger.info(f"🤖 AUTO_TRADE mode: {'ENABLED' if self.autonomous.enabled else 'DISABLED'}")
        
        # 📊 PERFORMANCE ANALYZER - самоанализ результатов
        from modules.performance_analyzer import PerformanceAnalyzer
        self.performance = PerformanceAnalyzer(db_path=self.db.db_path)
        logger.info("📊 PerformanceAnalyzer initialized")
        
        # 🧠 ADAPTIVE LEARNING - RL для оптимизации параметров
        try:
            from modules.adaptive_learning import AdaptiveLearning
            self.adaptive = AdaptiveLearning(db_path=self.db.db_path)
            logger.info(f"🧠 AdaptiveLearning initialized (Trained: {self.adaptive.is_trained})")
        except Exception as e:
            logger.warning(f"⚠️ AdaptiveLearning initialization failed: {e}")
            self.adaptive = None
        
        # 📊 MARKET REGIME DETECTION - HMM для определения состояния рынка
        try:
            from modules.market_regime import MarketRegimeManager
            self.regime_manager = MarketRegimeManager(db_path=self.db.db_path)
            logger.info("📊 MarketRegimeManager initialized")
        except Exception as e:
            logger.warning(f"⚠️ MarketRegimeManager initialization failed: {e}")
            self.regime_manager = None
        
        # 💭 SENTIMENT ANALYSIS - Анализ настроений рынка (Fear & Greed)
        try:
            from modules.sentiment_analyzer import SentimentAnalyzer
            self.sentiment_analyzer = SentimentAnalyzer()
            logger.info("💭 SentimentAnalyzer initialized")
        except Exception as e:
            logger.warning(f"⚠️ SentimentAnalyzer initialization failed: {e}")
            self.sentiment_analyzer = None
        
        # 🤖 INTELLIGENT AI - Multi-model ensemble (LSTM + Patterns)
        try:
            from modules.intelligent_ai import IntelligentAI
            self.intelligent_ai = IntelligentAI()
            logger.info("🤖 IntelligentAI initialized")
        except Exception as e:
            logger.warning(f"⚠️ IntelligentAI initialization failed: {e}")
            self.intelligent_ai = None
        
        # 💼 ADVANCED RISK MANAGER - Kelly Criterion, VaR, ATR-based SL
        try:
            from modules.risk_manager import AdvancedRiskManager
            self.risk_manager = AdvancedRiskManager(
                initial_balance=self.initial_balance,
                max_risk_per_trade=0.02  # 2% max risk per trade
            )
            logger.info("💼 AdvancedRiskManager initialized")
        except Exception as e:
            logger.warning(f"⚠️ AdvancedRiskManager initialization failed: {e}")
            self.risk_manager = None
        
        # Cache for markets
        self.markets_cache = None
        self.markets_cache_time = 0
        
        # Restore active positions from database on startup
        self._restore_active_positions()
    
    def _restore_active_positions(self):
        """Restore active positions from database on startup"""
        try:
            open_trades = self.db.get_open_trades()
            restored = 0
            
            for trade in open_trades:
                symbol = trade['symbol']
                entry_price = trade['entry_price']
                
                # ✅ Пересчитать ATR для восстановленных позиций
                atr = 0
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=50)
                    if ohlcv and len(ohlcv) >= 14:
                        df_atr = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df_atr['h-l'] = df_atr['high'] - df_atr['low']
                        df_atr['h-pc'] = abs(df_atr['high'] - df_atr['close'].shift(1))
                        df_atr['l-pc'] = abs(df_atr['low'] - df_atr['close'].shift(1))
                        df_atr['tr'] = df_atr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
                        atr = df_atr['tr'].rolling(window=14).mean().iloc[-1]
                        if pd.isna(atr) or atr == 0:
                            atr = entry_price * 0.02  # 2% fallback
                    else:
                        atr = entry_price * 0.02
                except Exception as e:
                    logger.warning(f"Failed to calculate ATR for {symbol}: {e}")
                    atr = entry_price * 0.02  # 2% fallback
                
                # Restore position to active_positions
                self.active_positions[symbol] = {
                    'trade_id': trade['trade_id'],
                    'symbol': symbol,
                    'side': trade['side'],
                    'entry_price': trade['entry_price'],
                    'amount': trade['amount'],
                    'usdt_amount': trade['usdt_amount'],
                    'fee': trade['fee'],
                    'stop_loss': trade['stop_loss'] or 0,
                    'take_profit': trade['take_profit'] or 0,
                    'entry_time': datetime.fromisoformat(trade['entry_time']) if isinstance(trade['entry_time'], str) else trade['entry_time'],
                    'atr': atr  # ✅ Пересчитанный ATR вместо 0
                }
                restored += 1
            
            if restored > 0:
                logger.info(f"✅ Restored {restored} active positions from database")
                for symbol, pos in self.active_positions.items():
                    logger.info(f"  - {pos['side']} {symbol} @ ${pos['entry_price']:.2f} (opened {pos['entry_time']})")
            else:
                logger.info("No active positions to restore")
                
        except Exception as e:
            logger.error(f"Failed to restore active positions: {e}", exc_info=True)

    async def send_telegram_message(self, chat_id: str, message: str):
        try:
            application = Application.builder().token(self.telegram_bot_token).build()
            await application.bot.send_message(chat_id=chat_id, text=message)
            logger.info(f"Message sent to Telegram chat {chat_id}")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
    
    async def send_telegram_message_with_buttons(self, chat_id: str, message: str, reply_markup):
        """Send message with inline keyboard buttons"""
        try:
            application = Application.builder().token(self.telegram_bot_token).build()
            await application.bot.send_message(chat_id=chat_id, text=message, reply_markup=reply_markup)
            logger.info(f"Message with buttons sent to Telegram chat {chat_id}")
        except Exception as e:
            logger.error(f"Error sending Telegram message with buttons: {e}")
    
    def scan_top_movers(self, top_n=100, min_volume_usdt=1000000, min_price_change_pct=3.0):
        """Scan Binance for top moving coins (autoscanner from professional bots)"""
        logger.info(f"Scanning top {top_n} coins on Binance...")
        try:
            # Use cached markets (refresh every 1 hour)
            current_time = time.time()
            if self.markets_cache is None or (current_time - self.markets_cache_time) > 3600:
                logger.info("Loading markets from Binance...")
                self.markets_cache = self.exchange.load_markets()
                self.markets_cache_time = current_time
                logger.info(f"Loaded {len(self.markets_cache)} markets")
            
            markets = self.markets_cache
            usdt_pairs = [symbol for symbol in markets if '/USDT' in symbol and markets[symbol].get('active', False)]
            
            # Limit to top_n pairs
            usdt_pairs = usdt_pairs[:top_n]
            logger.info(f"Scanning {len(usdt_pairs)} USDT pairs...")
            
            movers = []
            request_count = 0
            max_requests_per_scan = 50  # Limit scan to 50 requests to stay under rate limits
            
            for i, symbol in enumerate(usdt_pairs):
                try:
                    # Rate limiting protection - stop if reached limit
                    if request_count >= max_requests_per_scan:
                        logger.warning(f"Rate limit reached ({max_requests_per_scan} requests). Stopping scan.")
                        logger.info(f"Found {len(movers)} movers in first {i} symbols.")
                        break
                    
                    # Get 24h ticker
                    ticker = self.exchange.fetch_ticker(symbol)
                    request_count += 1
                    
                    # Filter by volume and price change
                    volume_usdt = ticker.get('quoteVolume', 0)
                    price_change_pct = ticker.get('percentage', 0)
                    
                    if volume_usdt >= min_volume_usdt and abs(price_change_pct) >= min_price_change_pct:
                        movers.append({
                            'symbol': symbol,
                            'price': ticker['last'],
                            'change_pct': price_change_pct,
                            'volume_usdt': volume_usdt,
                            'high_24h': ticker['high'],
                            'low_24h': ticker['low']
                        })
                        logger.info(f"Found mover: {symbol} ({price_change_pct:+.2f}%, Vol: ${volume_usdt:,.0f})")
                    
                    # Progress indicator every 10 symbols
                    if (i + 1) % 10 == 0:
                        logger.info(f"Scanned {i + 1}/{len(usdt_pairs)} symbols... ({len(movers)} movers found)")
                        
                    # Small delay to avoid rate limits (50ms = safe for 20 req/sec)
                    time.sleep(0.05)
                        
                except Exception as e:
                    logger.debug(f"Skipping {symbol}: {e}")
                    continue
            
            # Sort by price change (descending)
            movers.sort(key=lambda x: abs(x['change_pct']), reverse=True)
            
            # Update symbols list
            self.symbols = [m['symbol'] for m in movers[:20]]  # Top 20 movers
            logger.info(f"Autoscanner found {len(self.symbols)} hot coins: {', '.join(self.symbols[:5]) if self.symbols else 'None'}...")
            
            return movers[:20]
            
        except Exception as e:
            logger.error(f"Error scanning markets: {e}", exc_info=True)
            # Fallback to default symbols
            self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
            logger.warning(f"Using fallback symbols: {self.symbols}")
            return []
    
    def update_trailing_stop(self, symbol, current_price):
        """Trailing Stop Loss from Hummingbot - follows price up/down"""
        if symbol not in self.active_positions:
            return None
        
        position = self.active_positions[symbol]
        entry_price = position['entry_price']
        side = position['side']
        atr = position['atr']
        
        # Trailing stop distance: 2x ATR
        trailing_distance = atr * 2.0
        
        if side == 'BUY':
            # For LONG positions, stop follows price UP
            new_stop = current_price - trailing_distance
            old_stop = position.get('stop_loss', entry_price - trailing_distance)
            
            # Only move stop UP, never down
            if new_stop > old_stop:
                position['stop_loss'] = new_stop
                
                # ✅ СОХРАНЯЕМ В БД!
                trade_id = position.get('trade_id')
                if trade_id:
                    self.db.update_stop_loss(trade_id, new_stop)
                
                logger.info(f"Trailing stop updated for {symbol}: ${old_stop:.2f} → ${new_stop:.2f}")
                return new_stop
        
        elif side == 'SELL':  # SHORT
            # For SHORT positions, stop follows price DOWN
            new_stop = current_price + trailing_distance
            old_stop = position.get('stop_loss', entry_price + trailing_distance)
            
            # Only move stop DOWN, never up (for short)
            if new_stop < old_stop:
                position['stop_loss'] = new_stop
                
                # ✅ СОХРАНЯЕМ В БД!
                trade_id = position.get('trade_id')
                if trade_id:
                    self.db.update_stop_loss(trade_id, new_stop)
                
                logger.info(f"Trailing stop updated for SHORT {symbol}: ${old_stop:.2f} → ${new_stop:.2f}")
                return new_stop
        
        return position.get('stop_loss')
    
    def check_triple_barrier(self, symbol, current_price):
        """Triple Barrier Method from Hummingbot - 3 exit conditions"""
        if symbol not in self.active_positions:
            return None, None
        
        position = self.active_positions[symbol]
        entry_price = position['entry_price']
        entry_time = position['entry_time']
        side = position['side']
        atr = position['atr']
        
        # Barrier 1: Stop Loss (2x ATR)
        stop_loss = position.get('stop_loss', entry_price - atr * 2.0 if side == 'BUY' else entry_price + atr * 2.0)
        
        # Barrier 2: Take Profit (3x ATR)
        take_profit = entry_price + atr * 3.0 if side == 'BUY' else entry_price - atr * 3.0
        
        # Barrier 3: Time-based exit (24 hours max hold)
        time_elapsed = (datetime.now() - entry_time).total_seconds() / 3600
        max_hold_hours = 24
        
        if side == 'BUY':
            if current_price <= stop_loss:
                return 'STOP_LOSS', f"Hit stop loss at ${current_price:.2f}"
            elif current_price >= take_profit:
                return 'TAKE_PROFIT', f"Hit take profit at ${current_price:.2f}"
            elif time_elapsed >= max_hold_hours:
                return 'TIME_EXIT', f"Max hold time {max_hold_hours}h reached"
        
        return None, None

    def analyze_market_symbol(self, symbol: str) -> dict | None:
        """Analyze single symbol with AI decision-making
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
        
        Returns:
            dict with signal data if AI confidence >= 7, None otherwise
        """
        logger.info(f"Analyzing {symbol}...")
        try:
            # Multi-timeframe analysis: 1h + 4h
            try:
                ohlcv_1h = self.exchange.fetch_ohlcv(symbol, '1h', limit=200)
                ohlcv_4h = self.exchange.fetch_ohlcv(symbol, '4h', limit=50)
            except Exception as e:
                logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
                return None
            
            if not ohlcv_1h or len(ohlcv_1h) < 60:
                logger.warning(f"Insufficient data for {symbol}, skipping")
                return None
            
            df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], unit='ms')
            
            df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], unit='ms')
            
            # Calculate 4h indicators for trend confirmation
            df_4h['ema20'] = df_4h['close'].ewm(span=20, adjust=False).mean()
            df_4h['ema50'] = df_4h['close'].ewm(span=50, adjust=False).mean()
            trend_4h_bullish = df_4h['ema20'].iloc[-1] > df_4h['ema50'].iloc[-1]
            trend_4h_bearish = df_4h['ema20'].iloc[-1] < df_4h['ema50'].iloc[-1]
            
            # Use 1h for main analysis
            df = df_1h

            # --- Technical Analysis ---
            # RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            # EMA
            df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
            # MACD (requires more complex calculation, simplified for placeholder)
            df['macd'] = df['ema20'] - df['ema50']
            # ATR (Average True Range)
            df['h-l'] = df['high'] - df['low']
            df['h-pc'] = abs(df['high'] - df['close'].shift(1))
            df['l-pc'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
            df['atr'] = df['tr'].rolling(window=14).mean()

            # --- AI Prediction ---
            if self.ai_model.model:
                # Prepare data for prediction (e.g., last 60 close prices)
                last_60_closes = df['close'].tail(60).values.reshape(-1, 1)
                processed_data = self.ai_model.preprocess_data(last_60_closes)
                predicted_price = self.ai_model.predict(processed_data.flatten()) # Flatten for prediction
                logger.info(f"AI predicted next price for {self.symbol}: {predicted_price}")
                # Integrate AI prediction into trading signals
            else:
                logger.warning("AI model not loaded or trained. Skipping AI prediction.")
                predicted_price = None

            # --- 8 Filter System for Trading Signals ---
            signal = None
            current_price = df['close'].iloc[-1]
            current_rsi = df['rsi'].iloc[-1]
            current_ema20 = df['ema20'].iloc[-1]
            current_ema50 = df['ema50'].iloc[-1]
            current_macd = df['macd'].iloc[-1]
            current_atr = df['atr'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            
            # ✅ Проверка на NaN/Infinity
            if pd.isna([current_rsi, current_ema20, current_ema50, current_macd, current_atr]).any():
                logger.warning(f"{symbol}: NaN in indicators, skipping")
                return None
            
            if np.isinf([current_rsi, current_ema20, current_ema50, current_macd, current_atr]).any():
                logger.warning(f"{symbol}: Infinity in indicators, skipping")
                return None
            
            # Initialize filters (need 6/8 to trigger)
            buy_filters = 0
            sell_filters = 0
            
            # Filter 1: RSI
            if current_rsi < 30:
                buy_filters += 1
            elif current_rsi > 70:
                sell_filters += 1
                
            # Filter 2: EMA Trend
            if current_ema20 > current_ema50 and current_price > current_ema20:
                buy_filters += 1
            elif current_ema20 < current_ema50 and current_price < current_ema20:
                sell_filters += 1
                
            # Filter 3: MACD
            if current_macd > 0:
                buy_filters += 1
            elif current_macd < 0:
                sell_filters += 1
                
            # Filter 4: Volume
            if current_volume > avg_volume * 1.5:
                buy_filters += 1  # High volume confirms trend
                
            # Filter 5: Price momentum (3-candle)
            if df['close'].iloc[-1] > df['close'].iloc[-3]:
                buy_filters += 1
            elif df['close'].iloc[-1] < df['close'].iloc[-3]:
                sell_filters += 1
                
            # Filter 6: Volatility (ATR) - prefer calm markets
            atr_ratio = current_atr / current_price
            if atr_ratio < 0.02:  # Low volatility
                buy_filters += 1
                
            # Filter 7: AI Prediction (if available)
            if predicted_price:
                if predicted_price > current_price * 1.01:
                    buy_filters += 1
                elif predicted_price < current_price * 0.99:
                    sell_filters += 1
                    
            # Filter 8: 4h Trend Confirmation
            if trend_4h_bullish:
                buy_filters += 1
            elif trend_4h_bearish:
                sell_filters += 1
            
            # Filter 9 (was 8): Support/Resistance (simple: 20-period high/low)
            period_high = df['high'].rolling(20).max().iloc[-1]
            period_low = df['low'].rolling(20).min().iloc[-1]
            if current_price < period_low * 1.01:  # Near support
                buy_filters += 1
            elif current_price > period_high * 0.99:  # Near resistance
                sell_filters += 1
            
            # Логирование сработавших фильтров
            logger.info(f"{symbol}: BUY filters={buy_filters}, SELL filters={sell_filters}")
            logger.info(f"  RSI={current_rsi:.1f}, EMA20={current_ema20:.2f}, EMA50={current_ema50:.2f}")
            logger.info(f"  MACD={current_macd:.4f}, Volume={current_volume:.0f}/{avg_volume:.0f}={current_volume/avg_volume:.2f}x")
            logger.info(f"  Price momentum: {((df['close'].iloc[-1] - df['close'].iloc[-3]) / df['close'].iloc[-3]) * 100:.2f}%")
            logger.info(f"  ATR ratio: {(current_atr / current_price) * 100:.2f}%")
                
            # --- OpenAI Smart Decision (AI makes ALL decisions) ---
            ai_signal = None
            ai_confidence = 0
            ai_reason = ""
            
            # Call AI for coins with 3+ filters (token optimization - saves 50% calls)
            if buy_filters >= 3 or sell_filters >= 3:
                market_data = {
                    'symbol': symbol,
                    'price': current_price
                }
                filters_data = {
                    'rsi': current_rsi,
                    'ema_bullish': current_ema20 > current_ema50,
                    'volume_spike': current_volume > avg_volume * 1.5,
                    'buy_count': buy_filters,
                    'sell_count': sell_filters
                }
                
                logger.info(f"Calling AI for {symbol} (BUY:{buy_filters}, SELL:{sell_filters})...")
                ai_response = get_ai_trading_advice(market_data, filters_data)
                
                try:
                    ai_parts = ai_response.split('|')
                    if len(ai_parts) >= 3:
                        ai_signal = ai_parts[0].strip().upper()
                        # Валидация сигнала
                        if ai_signal not in ['BUY', 'SELL', 'WAIT']:
                            logger.warning(f"Invalid AI signal: {ai_signal}")
                            return None
                        
                        # Безопасная конвертация confidence
                        try:
                            ai_confidence = int(ai_parts[1].strip())
                            if not 1 <= ai_confidence <= 10:
                                logger.warning(f"Invalid confidence: {ai_confidence}")
                                ai_confidence = 5  # Default
                        except ValueError:
                            logger.warning(f"Non-numeric confidence: {ai_parts[1]}")
                            ai_confidence = 5
                        
                        ai_reason = ai_parts[2].strip()[:100]  # Limit length
                        logger.info(f"AI: {ai_signal} (confidence: {ai_confidence}/10) - {ai_reason}")
                        
                        # OPTIMIZATION: Require minimum confidence 7/10 (from backtesting)
                        if ai_confidence < 7:
                            logger.info(f"{symbol}: AI confidence {ai_confidence} < 7, skipping signal")
                            return None
                        
                        # OPTIMIZATION: Require minimum confidence 7/10 (from backtesting)
                        if ai_confidence < 7:
                            logger.info(f"{symbol}: AI confidence {ai_confidence} < 7, skipping")
                            return None
                    else:
                        logger.warning(f"Invalid AI response format: {ai_response}")
                        return None
                except (ValueError, IndexError) as e:
                    logger.error(f"Failed to parse AI response: {e}")
                    return None
            else:
                logger.debug(f"{symbol}: Insufficient filters ({buy_filters} BUY, {sell_filters} SELL) - skipping AI")
                
            # AI принимает ОКОНЧАТЕЛЬНОЕ решение (фильтры только информация)
            signal = None
            if ai_signal == 'BUY' and ai_confidence >= 7:
                signal = 'BUY'
                signal_strength = ai_confidence  # Используем AI confidence вместо фильтров
                logger.info(f"AI BUY SIGNAL for {symbol}: confidence {ai_confidence}/10")
            elif ai_signal == 'SELL' and ai_confidence >= 7:
                signal = 'SELL'
                signal_strength = ai_confidence
                logger.info(f"AI SELL SIGNAL for {symbol}: confidence {ai_confidence}/10")
            else:
                logger.info(f"AI decision: {ai_signal if ai_signal else 'WAIT'} (confidence too low or WAIT)")
                # No signal - return None
                return None

            if signal:
                # Check max concurrent positions
                MAX_CONCURRENT_POSITIONS = 5
                if len(self.active_positions) >= MAX_CONCURRENT_POSITIONS:
                    logger.warning(f"Max {MAX_CONCURRENT_POSITIONS} positions reached, skipping {symbol}")
                    return None
                
                # Avoid duplicate positions for same symbol
                if symbol in self.active_positions:
                    logger.warning(f"Position already open for {symbol}, skipping")
                    return None
                
                # Calculate position size using Risk Engine
                try:
                    balance = self.exchange.fetch_balance()
                    usdt_balance = balance['USDT']['free']
                except Exception as e:
                    logger.error(f"Failed to fetch balance: {e}")
                    return None
                
                crypto_amount, usdt_amount = self.risk_engine.calculate_position_size(
                    usdt_balance, current_price, signal_strength
                )
                fee = self.risk_engine.calculate_fees(usdt_amount)
                
                # Check if enough balance
                if usdt_balance < (usdt_amount + fee):
                    logger.warning(f"Insufficient balance: ${usdt_balance:.2f} < ${usdt_amount + fee:.2f}")
                    return None
                
                # 🛡️ SAFETY CHECK - 8 levels of protection
                signal_data_temp = {
                    'symbol': symbol,
                    'confidence': ai_confidence,
                    'price': current_price
                }
                
                # Get recent prices for volatility check
                recent_prices = df['close'].iloc[-5:].tolist() if len(df) >= 5 else []
                
                is_safe, safety_reason = self.safety.check_all_safety_levels(
                    signal_data_temp,
                    usdt_balance,
                    list(self.active_positions.values()),
                    recent_prices
                )
                
                if not is_safe:
                    logger.warning(f"🛡️ SAFETY BLOCK: {safety_reason}")
                    return None
                
                logger.info(f"🛡️ Safety check passed: {safety_reason}")
                
                # Return signal data (don't send yet - collect all first)
                return {
                    'symbol': symbol,
                    'signal': signal,
                    'price': current_price,
                    'ai_confidence': ai_confidence,
                    'ai_signal': ai_signal,
                    'ai_reason': ai_reason,
                    'signal_strength': signal_strength,
                    'crypto_amount': crypto_amount,
                    'usdt_amount': usdt_amount,
                    'fee': fee,
                    'current_rsi': current_rsi,
                    'current_ema20': current_ema20,
                    'current_ema50': current_ema50,
                    'current_macd': current_macd,
                    'current_volume': current_volume,
                    'avg_volume': avg_volume,
                    'current_atr': current_atr,
                    'buy_filters': buy_filters,
                    'sell_filters': sell_filters
                }
            else:
                return None

        except Exception as e:
            logger.error(f"Error during market analysis for {symbol}: {e}")
            return None
    
            logger.error(f"Error during market analysis for {symbol}: {e}")
    
    
    def send_signal_to_telegram(self, signal_data: dict) -> None:
        """Send a single AI signal to Telegram with buttons
        
        Args:
            signal_data: Dict containing all signal information (symbol, price, AI confidence, etc.)
        """
        try:
            trade_id = f"trade_{os.urandom(4).hex()}"
            
            symbol = signal_data['symbol']
            signal = signal_data['signal']
            current_price = signal_data['price']
            ai_confidence = signal_data['ai_confidence']
            ai_reason = signal_data['ai_reason']
            crypto_amount = signal_data['crypto_amount']
            usdt_amount = signal_data['usdt_amount']
            fee = signal_data['fee']
            signal_strength = signal_data['signal_strength']
            current_rsi = signal_data['current_rsi']
            current_ema20 = signal_data['current_ema20']
            current_ema50 = signal_data['current_ema50']
            current_volume = signal_data['current_volume']
            avg_volume = signal_data['avg_volume']
            
            # Формирование объяснения выбора монеты
            reasons = []
            
            # Анализ RSI
            if current_rsi < 30:
                reasons.append(f"RSI {current_rsi:.1f} - перепроданность")
            elif current_rsi > 70:
                reasons.append(f"RSI {current_rsi:.1f} - перекупленность")
            
            # Анализ тренда
            if current_ema20 > current_ema50:
                trend_strength = ((current_ema20 - current_ema50) / current_ema50) * 100
                reasons.append(f"Восходящий тренд ({trend_strength:.1f}%)")
            elif current_ema20 < current_ema50:
                trend_strength = ((current_ema50 - current_ema20) / current_ema50) * 100
                reasons.append(f"Нисходящий тренд ({trend_strength:.1f}%)")
            
            # Анализ объема
            volume_ratio = current_volume / avg_volume
            if volume_ratio > 1.5:
                reasons.append(f"Всплеск объема (x{volume_ratio:.1f})")
            
            # Формирование основного сообщения
            ai_info = f"\nAI решение ({ai_confidence}/10): {ai_reason}"
            reason_text = "\n".join([f"- {r}" for r in reasons[:3]]) if reasons else "- Технические показатели"
            
            message = (
                f"СИГНАЛ #{ai_confidence}/10: {signal} {symbol}\n"
                f"Цена: ${current_price:.2f}\n\n"
                f"Причины:\n{reason_text}"
                f"{ai_info}\n\n"
                f"Позиция: {crypto_amount:.6f} (~${usdt_amount:.2f})\n"
                f"Комиссия: ${fee:.2f}\n"
                f"Режим: {'ИМИТАЦИЯ' if self.paper_trading else 'РЕАЛЬНАЯ СДЕЛКА'}\n\n"
                f"Подтвердить или отклонить?"
            )
            
            # Create inline keyboard with approve/reject buttons
            keyboard = [
                [
                    InlineKeyboardButton("Подтвердить", callback_data=f"approve_{trade_id}"),
                    InlineKeyboardButton("Отклонить", callback_data=f"reject_{trade_id}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Save signal to database
            indicators = {
                'rsi': signal_data['current_rsi'],
                'ema20': signal_data['current_ema20'],
                'ema50': signal_data['current_ema50'],
                'macd': signal_data['current_macd'],
                'volume': signal_data['current_volume'],
                'avg_volume': signal_data['avg_volume'],
                'atr': signal_data['current_atr'],
                'filters_passed': signal_strength
            }
            
            ai_analysis = {
                'signal': signal_data['ai_signal'],
                'confidence': ai_confidence,
                'reason': ai_reason
            }
            
            position_info = {
                'amount': crypto_amount,
                'usdt_amount': usdt_amount,
                'fee': fee
            }
            
            # Сохранение в SQLite (основное)
            self.db.save_signal(trade_id, symbol, signal, current_price, indicators, ai_analysis, position_info)
            
            # Сохранение в Supabase (облачный бэкап)
            if self.supabase_db:
                try:
                    self.supabase_db.save_signal(trade_id, symbol, signal, current_price, indicators, ai_analysis, position_info)
                except Exception as e:
                    logger.warning(f"⚠️ Supabase save_signal failed: {e}")
            
            # 🤖 AUTONOMOUS DECISION: Should we execute automatically?
            balance_data = self.exchange.fetch_balance()
            current_balance = balance_data['USDT']['free']
            
            should_auto, auto_reason = self.autonomous.should_execute_auto(
                signal_data={
                    'symbol': symbol,
                    'ai_confidence': ai_confidence,
                    'usdt_amount': usdt_amount,
                    'signal': signal,
                    'price': current_price,
                    'crypto_amount': crypto_amount
                },
                active_positions=self.active_positions,
                balance=current_balance
            )
            
            if should_auto:
                # ✅ AUTO EXECUTE - выполнить сделку БЕЗ подтверждения
                logger.info(f"🤖 AUTO TRADE EXECUTING: {symbol} {signal} - {auto_reason}")
                
                try:
                    # Выполнить сделку напрямую
                    success = self._execute_trade_directly(
                        trade_id=trade_id,
                        symbol=symbol,
                        side=signal,
                        amount=crypto_amount,
                        price=current_price,
                        usdt_amount=usdt_amount,
                        fee=fee,
                        atr=signal_data['current_atr']
                    )
                    
                    if success:
                        # Записать в tracker
                        self.autonomous.record_trade()
                        
                        # Отправить уведомление (не ждем подтверждения!)
                        auto_message = (
                            f"🤖 АВТОМАТИЧЕСКАЯ СДЕЛКА ВЫПОЛНЕНА\n\n"
                            f"📊 Монета: {symbol}\n"
                            f"💰 Цена: ${current_price:.2f}\n"
                            f"📦 Размер: {crypto_amount:.6f} (~${usdt_amount:.2f})\n"
                            f"📈 Сигнал: {signal}\n"
                            f"🤖 AI Уверенность: {ai_confidence}/10\n\n"
                            f"✅ Причина: {auto_reason}\n\n"
                            f"⏰ Время: {datetime.now().strftime('%H:%M:%S')}\n"
                            f"Режим: {'ИМИТАЦИЯ' if self.paper_trading else 'РЕАЛЬНАЯ СДЕЛКА'}"
                        )
                        
                        operator_chat_id = self.operator_chat_id
                        try:
                            loop = asyncio.get_running_loop()
                            asyncio.create_task(self.send_telegram_message(operator_chat_id, auto_message))
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                loop.run_until_complete(self.send_telegram_message(operator_chat_id, auto_message))
                            except Exception as e:
                                logger.error(f"Failed to send auto-trade notification: {e}")
                            finally:
                                loop.close()
                        
                        logger.info(f"✅ AUTO TRADE COMPLETED: {symbol} {signal}")
                    else:
                        logger.error(f"❌ AUTO TRADE FAILED: {symbol} {signal}")
                        
                except Exception as e:
                    logger.error(f"❌ AUTO TRADE ERROR: {symbol} - {e}", exc_info=True)
            else:
                # ⏸️ MANUAL CONFIRMATION - требуется подтверждение
                logger.info(f"⏸️  Manual confirmation required: {auto_reason}")
                
                # Send message to Telegram (use existing event loop if available)
                operator_chat_id = self.operator_chat_id
                try:
                    loop = asyncio.get_running_loop()
                    # We're in async context, create task AND WAIT
                    task = asyncio.create_task(self.send_telegram_message_with_buttons(operator_chat_id, message, reply_markup))
                    # Give task time to complete (don't await to avoid blocking)
                    logger.info(f"📤 Telegram message task created for {symbol}")
                except RuntimeError:
                    # No running loop, create new one
                    logger.info(f"📤 Sending Telegram message (new event loop) for {symbol}")
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self.send_telegram_message_with_buttons(operator_chat_id, message, reply_markup))
                        logger.info(f"✅ Telegram message sent for {symbol}")
                    except Exception as telegram_error:
                        logger.error(f"❌ Failed to send Telegram message: {telegram_error}", exc_info=True)
                    finally:
                        loop.close()
                
                self.trade_confirmation_needed[trade_id] = {
                    'symbol': symbol,
                    'side': signal,
                    'price': current_price,
                    'amount': crypto_amount,
                    'usdt_amount': usdt_amount,
                    'fee': fee,
                    'atr': signal_data['current_atr']
                }
                logger.info(f"Signal sent to Telegram: {signal} {symbol} (AI: {ai_confidence}/10)")
            
        except Exception as e:
            logger.error(f"Error sending signal to Telegram: {e}", exc_info=True)
    
    def check_and_close_positions(self) -> None:
        """Monitor and auto-close positions based on trailing stop, TP, SL"""
        if not self.active_positions:
            return
        
        logger.info(f"Monitoring {len(self.active_positions)} active positions...")
        
        positions_to_close = []
        
        for symbol, position in list(self.active_positions.items()):
            try:
                # Fetch current price with error handling
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                except Exception as api_error:
                    logger.warning(f"Failed to fetch price for {symbol}: {api_error}")
                    continue  # Skip this position if API fails
                
                entry_price = position['entry_price']
                side = position['side']
                amount = position['amount']
                stop_loss = position.get('stop_loss', 0)
                take_profit = position.get('take_profit', 0)
                entry_time = position['entry_time']
                
                # Update trailing stop
                new_stop = self.update_trailing_stop(symbol, current_price)
                if new_stop:
                    stop_loss = new_stop
                    position['stop_loss'] = new_stop
                
                # Calculate P&L
                if side == 'BUY':
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    
                    # Close conditions for LONG
                    if current_price <= stop_loss and stop_loss > 0:
                        reason = f"Trailing Stop Hit (${stop_loss:.2f})"
                        positions_to_close.append((symbol, current_price, reason))
                    elif take_profit > 0 and current_price >= take_profit:
                        reason = f"Take Profit Hit (${take_profit:.2f})"
                        positions_to_close.append((symbol, current_price, reason))
                else:  # SELL (SHORT)
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    
                    # Close conditions for SHORT (обратная логика!)
                    # Для SHORT: stop_loss ВЫШЕ entry_price, take_profit НИЖЕ
                    if stop_loss > 0 and current_price >= stop_loss:
                        reason = f"Stop Loss Hit (${stop_loss:.2f})"
                        positions_to_close.append((symbol, current_price, reason))
                    elif take_profit > 0 and current_price <= take_profit:
                        reason = f"Take Profit Hit (${take_profit:.2f})"
                        positions_to_close.append((symbol, current_price, reason))
                
                # Time-based close (positions older than 48 hours)
                hold_time_hours = (datetime.now() - entry_time).total_seconds() / 3600
                if hold_time_hours > 48:
                    reason = f"Max Hold Time Reached ({hold_time_hours:.1f}h)"
                    positions_to_close.append((symbol, current_price, reason))
                
                # Max loss protection (-10%)
                if pnl_pct < -10:
                    reason = f"Max Loss Protection ({pnl_pct:.1f}%)"
                    positions_to_close.append((symbol, current_price, reason))
                    
            except Exception as e:
                logger.error(f"Error monitoring {symbol}: {e}")
                continue
        
        # Close positions
        for symbol, exit_price, reason in positions_to_close:
            self.close_position(symbol, exit_price, reason)
    
    def execute_trade(self, trade_id: str) -> bool:
        """Execute a confirmed trade
        
        Returns:
            bool: True if trade executed successfully, False otherwise
        """
        if trade_id not in self.trade_confirmation_needed:
            logger.error(f"Trade {trade_id} not found in pending trades")
            return False
        
        trade_info = self.trade_confirmation_needed[trade_id]
        symbol = trade_info['symbol']
        side = trade_info['side']
        price = trade_info['price']
        amount = trade_info['amount']
        usdt_amount = trade_info['usdt_amount']
        fee = trade_info['fee']
        atr = trade_info.get('atr', 0)
        
        # ✅ CHECK BALANCE BEFORE TRADE
        try:
            balance = self.exchange.fetch_balance()
            available_usdt = balance['USDT']['free']
            if available_usdt < usdt_amount:
                logger.error(f"❌ Insufficient balance: need ${usdt_amount:.2f}, have ${available_usdt:.2f}")
                asyncio.run(self.send_telegram_message(self.operator_chat_id, 
                    f"⚠️ Недостаточно средств для {symbol}\nТребуется: ${usdt_amount:.2f}\nДоступно: ${available_usdt:.2f}"))
                del self.trade_confirmation_needed[trade_id]
                return False
        except Exception as balance_error:
            logger.error(f"Failed to check balance: {balance_error}")
        
        try:
            # Save trade to database
            stop_loss = price - (2 * atr) if side == 'BUY' else price + (2 * atr)
            take_profit = price + (3 * atr) if side == 'BUY' else price - (3 * atr)
            
            # 🔥 REAL ORDER EXECUTION (if not paper trading)
            actual_price = price
            order_id = None
            
            if not self.paper_trading:
                try:
                    logger.info(f"📤 Placing REAL {side} order: {amount} {symbol}")
                    if side == 'BUY':
                        order = self.exchange.create_market_buy_order(symbol, amount)
                    else:
                        order = self.exchange.create_market_sell_order(symbol, amount)
                    
                    order_id = order.get('id')
                    actual_price = float(order.get('average', price) or price)
                    logger.info(f"✅ REAL ORDER EXECUTED: {order_id} @ ${actual_price:.2f}")
                except Exception as order_error:
                    logger.error(f"❌ REAL ORDER FAILED: {order_error}")
                    del self.trade_confirmation_needed[trade_id]
                    return False
            else:
                logger.info(f"📝 PAPER TRADE: {side} {amount} {symbol} @ ${price:.2f}")
            
            # Сохранение в SQLite (основное)
            self.db.save_trade(
                trade_id=trade_id,
                symbol=symbol,
                side=side,
                entry_price=actual_price,
                amount=amount,
                usdt_amount=usdt_amount,
                fee=fee,
                mode='paper' if self.paper_trading else 'real',
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Сохранение в Supabase (облачный бэкап)
            if self.supabase_db:
                try:
                    self.supabase_db.save_trade(
                        trade_id=trade_id,
                        symbol=symbol,
                        side=side,
                        entry_price=price,
                        amount=amount,
                        usdt_amount=usdt_amount,
                        fee=fee,
                        mode='paper' if self.paper_trading else 'real',
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Supabase save_trade failed: {e}")
            
            # Add to active positions
            self.active_positions[symbol] = {
                'trade_id': trade_id,
                'symbol': symbol,
                'side': side,
                'entry_price': price,
                'amount': amount,
                'usdt_amount': usdt_amount,
                'fee': fee,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now(),
                'atr': atr
            }
            
            # Update signal status
            self.db.update_signal_status(trade_id, 'approved')
            
            # Remove from pending
            del self.trade_confirmation_needed[trade_id]
            
            logger.info(f"Trade executed: {side} {symbol} @ ${price:.2f} (ID: {trade_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute trade {trade_id}: {e}")
            return False
    
    def _execute_trade_directly(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        usdt_amount: float,
        fee: float,
        atr: float
    ) -> bool:
        """
        🤖 Execute trade directly WITHOUT manual approval (AUTO_TRADE mode)
        
        Args:
            trade_id: Unique trade ID
            symbol: Trading pair
            side: BUY/SELL
            amount: Crypto amount
            price: Entry price
            usdt_amount: USDT amount
            fee: Trading fee
            atr: ATR value for stops
            
        Returns:
            success: bool
        """
        try:
            # Calculate stops (same as manual execution)
            if side == 'BUY':
                stop_loss = price - (2 * atr)
                take_profit = price + (3 * atr)
            else:
                stop_loss = price + (2 * atr)
                take_profit = price - (3 * atr)
            
            # 🔥 REAL ORDER EXECUTION (if not paper trading)
            actual_price = price
            order_id = None
            
            if not self.paper_trading:
                try:
                    logger.info(f"📤 Placing REAL {side} order: {amount} {symbol}")
                    if side == 'BUY':
                        order = self.exchange.create_market_buy_order(symbol, amount)
                    else:
                        order = self.exchange.create_market_sell_order(symbol, amount)
                    
                    order_id = order.get('id')
                    actual_price = float(order.get('average', price) or price)
                    logger.info(f"✅ REAL ORDER EXECUTED: {order_id} @ ${actual_price:.2f}")
                except Exception as order_error:
                    logger.error(f"❌ REAL ORDER FAILED: {order_error}")
                    return False
            else:
                logger.info(f"📝 PAPER TRADE: {side} {amount} {symbol} @ ${price:.2f}")
            
            # Save to database (SQLite)
            self.db.save_trade(
                trade_id=trade_id,
                symbol=symbol,
                side=side,
                entry_price=actual_price,
                amount=amount,
                usdt_amount=usdt_amount,
                fee=fee,
                mode='paper' if self.paper_trading else 'real',
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Save to Supabase (cloud backup)
            if self.supabase_db:
                try:
                    self.supabase_db.save_trade(
                        trade_id=trade_id,
                        symbol=symbol,
                        side=side,
                        entry_price=price,
                        amount=amount,
                        usdt_amount=usdt_amount,
                        fee=fee,
                        mode='paper' if self.paper_trading else 'real',
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Supabase save_trade failed: {e}")
            
            # Add to active positions
            self.active_positions[symbol] = {
                'trade_id': trade_id,
                'symbol': symbol,
                'side': side,
                'entry_price': price,
                'amount': amount,
                'usdt_amount': usdt_amount,
                'fee': fee,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now(),
                'atr': atr
            }
            
            # Update signal status
            self.db.update_signal_status(trade_id, 'auto_approved')
            
            logger.info(f"🤖 AUTO TRADE executed: {side} {symbol} @ ${price:.2f} (ID: {trade_id})")
            return True
        except Exception as e:
            logger.error(f"❌ AUTO TRADE FAILED {trade_id}: {e}", exc_info=True)
            return False
    
    def close_position(self, symbol: str, exit_price: float, reason: str = "Manual close") -> None:
        """Close a position and update database"""
        if symbol not in self.active_positions:
            logger.warning(f"Position {symbol} not found")
            return
        
        position = self.active_positions.pop(symbol)
        
        entry_price = position['entry_price']
        side = position['side']
        amount = position['amount']
        usdt_amount = position['usdt_amount']
        fee = position['fee']
        entry_time = position['entry_time']
        
        # 🔥 REAL ORDER TO CLOSE POSITION (if not paper trading)
        actual_exit_price = exit_price
        if not self.paper_trading:
            try:
                logger.info(f"📤 Closing REAL position: {amount} {symbol}")
                # Opposite order to close
                if side == 'BUY':
                    order = self.exchange.create_market_sell_order(symbol, amount)
                else:
                    order = self.exchange.create_market_buy_order(symbol, amount)
                
                actual_exit_price = float(order.get('average', exit_price) or exit_price)
                logger.info(f"✅ REAL POSITION CLOSED @ ${actual_exit_price:.2f}")
            except Exception as order_error:
                logger.error(f"❌ FAILED TO CLOSE REAL POSITION: {order_error}")
                # Restore position if order failed
                self.active_positions[symbol] = position
                return
        
        # Calculate P&L
        if side == 'BUY':
            pnl = (actual_exit_price - entry_price) * amount
            pnl_pct = ((actual_exit_price - entry_price) / entry_price) * 100
        else:
            pnl = (entry_price - actual_exit_price) * amount
            pnl_pct = ((entry_price - actual_exit_price) / entry_price) * 100
        
        # Apply fees
        exit_fee = self.risk_engine.calculate_fees(usdt_amount)
        pnl -= (fee + exit_fee)
        
        hold_time = (datetime.now() - entry_time).total_seconds() / 3600
        
        # Find trade_id from database
        trades = self.db.get_all_trades(limit=100)
        trade_id = None
        for trade in trades:
            if trade['symbol'] == symbol and trade['status'] == 'open':
                trade_id = trade['trade_id']
                break
        
        if trade_id:
            # Обновление в SQLite (основное)
            self.db.close_trade(trade_id, exit_price, pnl, pnl_pct)
            
            # Обновление в Supabase (облачный бэкап)
            if self.supabase_db:
                try:
                    self.supabase_db.update_trade(trade_id, exit_price, pnl, pnl_pct, exit_fee)
                except Exception as e:
                    logger.warning(f"⚠️ Supabase update_trade failed: {e}")
        
        # 🛡️ Record P&L in Safety Manager
        self.safety.record_trade(pnl)
        
        # Send notification
        message = (
            f"🔴 ПОЗИЦИЯ ЗАКРЫТА\n"
            f"Причина: {reason}\n\n"
            f"{side} {symbol}\n"
            f"Вход: ${entry_price:.2f}\n"
            f"Выход: ${exit_price:.2f}\n"
            f"Время: {hold_time:.1f}h\n\n"
            f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)\n"
            f"Режим: {'ИМИТАЦИЯ' if self.paper_trading else 'РЕАЛЬНАЯ СДЕЛКА'}"
        )
        
        operator_chat_id = self.operator_chat_id
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.send_telegram_message(operator_chat_id, message))
            else:
                loop.run_until_complete(self.send_telegram_message(operator_chat_id, message))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.send_telegram_message(operator_chat_id, message))
            loop.close()
        
        logger.info(f"Position closed: {symbol} at ${exit_price:.2f} ({reason}) P&L: {pnl_pct:+.2f}%")
        
        # 📊 Периодический анализ производительности
        try:
            if hasattr(self, 'performance') and len(self.db.get_closed_trades_since(days=7)) >= 5:
                analysis = self.performance.analyze_closed_trades(days=7)
                logger.info(f"📊 Weekly performance: Win rate {analysis.get('win_rate', 0)}%, ROI {analysis.get('roi', 0)}%")
        except Exception as e:
            logger.warning(f"Performance analysis failed: {e}")
    
    def analyze_all_markets(self) -> None:
        """Analyze all symbols and send TOP-3 AI signals
        
        Process:
        1. Scan top movers (OPTIMIZED: prioritize ETH/USDT from backtesting)
        2. Analyze each with AI
        3. Sort by confidence
        4. Send TOP-3 to Telegram
        """
        logger.info("=== Starting market analysis cycle ===")
        
        # OPTIMIZATION: Always include ETH/USDT (best backtest performance: +1.75% ROI)
        priority_symbols = ['ETH/USDT']
        
        # Run scanner every cycle to find hot coins
        self.scan_top_movers(top_n=100, min_volume_usdt=1000000, min_price_change_pct=3.0)
        
        # Combine priority + scanned symbols
        combined_symbols = priority_symbols + [s for s in self.symbols if s not in priority_symbols]
        
        # Collect ALL signals with AI confidence
        all_signals = []
        
        # Analyze each symbol
        for symbol in combined_symbols:
            try:
                signal_data = self.analyze_market_symbol(symbol)
                if signal_data:  # If AI gave signal
                    all_signals.append(signal_data)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by AI confidence and take TOP-3
        if all_signals:
            all_signals.sort(key=lambda x: x['ai_confidence'], reverse=True)
            top_signals = all_signals[:3]
            
            logger.info(f"Found {len(all_signals)} AI signals, sending TOP-3")
            
            # Send TOP-3 signals to Telegram
            for signal_data in top_signals:
                self.send_signal_to_telegram(signal_data)
        else:
            logger.info("No AI signals found in this cycle")
        
        logger.info("=== Market analysis cycle complete ===")

    async def place_order(self, symbol: str, side: str, amount: float):
        logger.info(f"Attempting to place {side} order for {amount} {symbol}...")
        try:
            if side == 'BUY':
                order = await self.exchange.create_market_buy_order(symbol, amount)
            elif side == 'SELL':
                order = await self.exchange.create_market_sell_order(symbol, amount)
            else:
                logger.warning(f"Invalid trade side: {side}")
                return None
            logger.info(f"Order placed: {order}")
            return order
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    async def manage_stop_loss_take_profit(self, order_id: str, symbol: str, entry_price: float, side: str, atr_value: float):
        # Dynamic Stop-Loss and Take-Profit based on ATR
        logger.info(f"Managing SL/TP for order {order_id} at {entry_price} with ATR {atr_value}")
        
        try:
            # Calculate dynamic SL/TP using ATR
            atr_multiplier_sl = 2.0  # 2x ATR for stop-loss
            atr_multiplier_tp = 3.0  # 3x ATR for take-profit
            
            if side == 'BUY':
                stop_loss_price = entry_price - (atr_value * atr_multiplier_sl)
                take_profit_price = entry_price + (atr_value * atr_multiplier_tp)
            else:  # SELL
                stop_loss_price = entry_price + (atr_value * atr_multiplier_sl)
                take_profit_price = entry_price - (atr_value * atr_multiplier_tp)
                
            logger.info(f"SL: {stop_loss_price:.2f}, TP: {take_profit_price:.2f}")
            
            # Place stop-loss order
            if side == 'BUY':
                sl_order = await self.exchange.create_stop_loss_order(
                    symbol, 'sell', 0.001, stop_loss_price
                )
            else:
                sl_order = await self.exchange.create_stop_loss_order(
                    symbol, 'buy', 0.001, stop_loss_price
                )
                
            logger.info(f"Stop-Loss order placed: {sl_order}")
            return True
            
        except Exception as e:
            logger.error(f"Error managing SL/TP: {e}")
            return False

# --- Telegram Bot Handlers ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_html(
        f"Hi {user.mention_html()}! I am NexusTrader AI. Send /status to get bot status or /trade to initiate a manual trade.",
        reply_markup=ForceReply(selective=True),
    )
    logger.info(f"User {user.id} started the bot.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Help!")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    agent = context.bot_data['agent']
    
    # Get database statistics
    db_stats = agent.db.get_statistics()
    
    # Count active positions
    open_trades = agent.db.get_open_trades()
    active_positions = len(open_trades)
    
    # Get balance
    try:
        balance = agent.exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
    except:
        usdt_balance = 0.0
    
    # Trading mode
    mode_text = "ИМИТАЦИЯ" if agent.paper_trading else "РЕАЛЬНАЯ ТОРГОВЛЯ"
    
    status_msg = (
        f"NexusTrader AI - Статус\n\n"
        f"Баланс: ${usdt_balance:.2f} USDT\n"
        f"Режим: {mode_text}\n"
        f"Мониторинг: {len(agent.symbols)} монет\n"
        f"Открытые позиции: {active_positions}\n\n"
        f"СТАТИСТИКА:\n"
        f"Всего сделок: {db_stats['total_trades']}\n"
        f"Прибыльных: {db_stats['winning_trades']}\n"
        f"Убыточных: {db_stats['losing_trades']}\n"
        f"Win Rate: {db_stats['win_rate']:.1f}%\n"
        f"Общая прибыль: ${db_stats['total_pnl']:.2f}\n"
        f"Средняя прибыль: ${db_stats['avg_pnl']:.2f}\n"
        f"Лучшая сделка: ${db_stats['max_win']:.2f}\n"
        f"Худшая сделка: ${db_stats['max_loss']:.2f}\n\n"
        f"💰 AI ТОКЕНЫ:\n"
        f"Использовано: {total_tokens_used:,}\n"
        f"AI запросов: {total_ai_calls}\n"
        f"Средний запрос: {total_tokens_used/total_ai_calls if total_ai_calls > 0 else 0:.0f} токенов\n"
        f"Стоимость: ${(total_tokens_used/1000000)*0.15:.3f}\n\n"
        f"Автосканирование: каждые 5 минут\n"
        f"AI: OpenAI GPT-4o-mini (кэш 5 мин)\n\n"
        f"/analyze - ручной анализ\n"
        f"/positions - открытые позиции\n"
        f"/history - история сделок\n"
        f"/help - справка"
    )
    await update.message.reply_text(status_msg)

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run autoscanner + market analysis"""
    agent = context.bot_data['agent']
    await update.message.reply_text("🔍 Running autoscanner + analysis... (это займёт 1-2 минуты)")
    
    try:
        # Run scanner + analysis in background thread - don't block Telegram!
        import threading
        import asyncio
        
        def run_scan():
            try:
                logger.info("=== Starting market analysis cycle ===")
                agent.analyze_all_markets()
                logger.info("=== Market analysis complete ===")
            except Exception as e:
                logger.error(f"Error in analysis: {e}", exc_info=True)
        
        # Start in background thread WITHOUT blocking
        scan_thread = threading.Thread(target=run_scan, daemon=True)
        scan_thread.start()
        
        # Send immediate confirmation - don't wait for results
        await update.message.reply_text(
            "⏳ Сканирование запущено в фоне!\n"
            "Это займёт 1-2 минуты (100 монет).\n"
            "Используй /status чтобы увидеть найденные монеты."
        )
            
    except Exception as e:
        logger.error(f"Error in manual analysis: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "🤖 NexusTrader AI - Команды\n\n"
        "📊 ОСНОВНЫЕ:\n"
        "/start - Запустить бота\n"
        "/status - Баланс и статистика\n"
        "/positions - Открытые позиции\n"
        "/portfolio - AI анализ портфеля\n"
        "/history - История сделок\n"
        "/analyze - Ручной анализ рынка\n\n"
        "🤖 AUTO_TRADE (Автономный режим):\n"
        "/auto_status - Статус AUTO_TRADE\n"
        "/auto_toggle - Включить/Выключить\n"
        "/auto_aggressive - Агрессивный режим\n"
        "/auto_emergency - 🚨 Остановить AUTO_TRADE\n\n"
        "� PERFORMANCE (Анализ результатов):\n"
        "/performance - Дневной отчет\n"
        "/analytics - 30-дневная статистика\n"
        "/recommendations - Рекомендации от ИИ\n"
        "/filters - Эффективность фильтров\n"
        "/optimize - Оптимальные параметры\n\n"
        "🛡️ БЕЗОПАСНОСТЬ:\n"
        "/safety - Статус защиты (8 уровней)\n"
        "/pause - Приостановить торговлю\n"
        "/resume - Возобновить торговлю\n"
        "/emergency_stop - 🚨 ЭКСТРЕННАЯ ОСТАНОВКА\n\n"
        "✨ ВОЗМОЖНОСТИ:\n"
        "- 🤖 Автономная торговля (AUTO_TRADE)\n"
        "- 📊 Самоанализ производительности\n"
        "- 🧠 Adaptive Learning (Reinforcement Learning)\n"
        "- 📊 Market Regime Detection (HMM)\n"
        "- 💭 Sentiment Analysis (Fear & Greed)\n"
        "- 🤖 Intelligent AI (LSTM + Patterns)\n"
        "- 💼 Advanced Risk Management (Kelly, VaR, ATR)\n"
        "- 8-уровневая система защиты\n"
        "- Автосканирование топ-100 монет\n"
        "- AI анализ от OpenAI (GPT-4)\n"
        "- Оптимизированная стратегия (+1.75% ROI)\n"
        "- Автоматическое управление рисками\n"
        "- База данных сделок\n"
        "- Trailing stop & Take Profit\n\n"
        "💭 SENTIMENT ANALYSIS:\n"
        "/sentiment - 📊 Общий sentiment рынка\n"
        "/fear_greed - 😱 Fear & Greed Index\n"
        "/sentiment_trend - 📈 Тренд sentiment (7 дней)\n\n"
        "🤖 INTELLIGENT AI:\n"
        "/ai_predict - 🔮 AI предсказание цены\n"
        "/ai_train - 🎓 Обучить LSTM модель\n"
        "/ai_patterns - 🎨 Обнаруженные паттерны\n\n"
        "💼 RISK MANAGEMENT:\n"
        "/risk - 📊 Полный анализ рисков\n"
        "/var - 📉 Value at Risk (VaR)\n"
        "/kelly - 🎯 Kelly Criterion sizing"
    )
    await update.message.reply_text(help_text)

async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показать историю последних сделок"""
    agent = context.bot_data['agent']
    
    # Get last 10 trades
    trades = agent.db.get_all_trades(limit=10)
    
    if not trades:
        await update.message.reply_text("История пуста")
        return
    
    history_msg = "ИСТОРИЯ СДЕЛОК (последние 10):\n\n"
    
    for trade in trades:
        status_icon = "[ОТКРЫТА]" if trade['status'] == 'open' else "[ЗАКРЫТА]"
        mode_icon = "[ИМИТАЦИЯ]" if trade['mode'] == 'paper' else "[РЕАЛЬНАЯ]"
        
        msg = (
            f"{status_icon} {mode_icon}\n"
            f"{trade['side']} {trade['symbol']}\n"
            f"Вход: ${trade['entry_price']:.2f}\n"
            f"Сумма: ${trade['usdt_amount']:.2f}\n"
        )
        
        if trade['status'] == 'closed':
            pnl_sign = "+" if trade['pnl'] > 0 else ""
            msg += f"Выход: ${trade['exit_price']:.2f}\n"
            msg += f"PnL: {pnl_sign}${trade['pnl']:.2f} ({pnl_sign}{trade['pnl_percent']:.2f}%)\n"
        
        msg += f"Время: {trade['entry_time'][:16]}\n\n"
        history_msg += msg
    
    await update.message.reply_text(history_msg)

async def positions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Show active positions with trailing stops
    agent = context.bot_data['agent']
    
    if not agent.active_positions:
        await update.message.reply_text("📊 No active positions")
        return
    
    positions_msg = "📈 ACTIVE POSITIONS:\n\n"
    
    for symbol, pos in agent.active_positions.items():
        entry_price = pos['entry_price']
        side = pos['side']
        amount = pos['amount']
        stop_loss = pos.get('stop_loss', 0)
        entry_time = pos['entry_time']
        
        # Get current price
        try:
            ticker = agent.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Calculate P&L
            if side == 'BUY':
                pnl = (current_price - entry_price) * amount
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl = (entry_price - current_price) * amount
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            hold_time = (datetime.now() - entry_time).total_seconds() / 3600
            
            positions_msg += (
                f"💹 {symbol}\n"
                f"Side: {side}\n"
                f"Entry: ${entry_price:.2f} → Now: ${current_price:.2f}\n"
                f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)\n"
                f"🛡️ Trailing Stop: ${stop_loss:.2f}\n"
                f"⏱️ Hold: {hold_time:.1f}h\n\n"
            )
        except Exception as e:
            positions_msg += f"❌ Error fetching {symbol}: {e}\n\n"
    
    await update.message.reply_text(positions_msg)

async def safety_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """🛡️ Safety Status - Show all 8 protection levels"""
    agent = context.bot_data['agent']
    status = agent.safety.get_status()
    await update.message.reply_text(status)

async def emergency_stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """🚨 EMERGENCY STOP - Immediately stop all trading and close positions"""
    agent = context.bot_data['agent']
    
    # Activate emergency stop
    agent.safety.activate_emergency_stop()
    
    # Close all positions
    closed_count = 0
    for symbol in list(agent.active_positions.keys()):
        try:
            ticker = agent.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            agent.close_position(symbol, current_price, "EMERGENCY STOP")
            closed_count += 1
        except Exception as e:
            logger.error(f"Failed to close {symbol}: {e}")
    
    msg = f"🚨 EMERGENCY STOP ACTIVATED!\n\n"
    msg += f"✅ Closed {closed_count} positions\n"
    msg += f"🔒 All new trades blocked\n\n"
    msg += f"Use /resume to re-enable trading"
    
    await update.message.reply_text(msg)

async def pause_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """⏸️ Pause Trading - Stop opening new positions (keep existing)"""
    agent = context.bot_data['agent']
    agent.safety.pause_trading()
    
    msg = "⏸️ Trading PAUSED\n\n"
    msg += f"Active positions: {len(agent.active_positions)}\n"
    msg += f"New trades blocked until /resume"
    
    await update.message.reply_text(msg)

async def resume_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """▶️ Resume Trading - Re-enable trading after pause/emergency"""
    agent = context.bot_data['agent']
    
    # Check if emergency stop - require confirmation
    if agent.safety.emergency_stop:
        agent.safety.deactivate_emergency_stop()
        msg = "✅ Emergency stop DEACTIVATED\n"
    else:
        agent.safety.resume_trading()
        msg = "▶️ Trading RESUMED\n"
    
    msg += f"\nBot is now active again."
    await update.message.reply_text(msg)

async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """🎯 AI Portfolio Analysis - Analyze all holdings and give recommendations"""
    agent = context.bot_data['agent']
    
    await update.message.reply_text("🔍 Analyzing your portfolio with AI... (this may take 30-60 seconds)")
    
    try:
        # Get balance from Binance
        balance = agent.exchange.fetch_balance()
        
        # Filter out zero balances, USDT, locked/staked assets, and leveraged tokens
        holdings = {}
        excluded_prefixes = ('LD', 'UP', 'DOWN', 'BULL', 'BEAR', 'AZN')
        for asset, amount in balance['total'].items():
            if amount > 0 and asset != 'USDT' and not asset.startswith(excluded_prefixes):
                holdings[asset] = amount
        
        if not holdings:
            await update.message.reply_text("📭 Your portfolio is empty (only USDT)")
            return
        
        # Get USDT balance
        usdt_balance = balance['total'].get('USDT', 0)
        
        portfolio_msg = f"💼 PORTFOLIO ANALYSIS\n\n"
        portfolio_msg += f"💰 USDT: ${usdt_balance:.2f}\n\n"
        
        total_value = usdt_balance
        recommendations = []
        
        # Analyze each holding
        for asset, amount in holdings.items():
            symbol = f"{asset}/USDT"
            
            try:
                # Get current price and market data
                ticker = agent.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                value_usd = amount * current_price
                total_value += value_usd
                
                # Get OHLCV for technical analysis
                ohlcv = agent.exchange.fetch_ohlcv(symbol, '1h', limit=100)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Calculate indicators
                close_prices = df['close']
                delta = close_prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]
                
                ema20 = close_prices.ewm(span=20, adjust=False).mean().iloc[-1]
                ema50 = close_prices.ewm(span=50, adjust=False).mean().iloc[-1]
                trend = "📈 UP" if ema20 > ema50 else "📉 DOWN"
                
                # 24h change
                change_24h = ticker.get('percentage', 0)
                
                # AI Analysis
                ai_prompt = f"""{symbol} portfolio review:
Price: ${current_price:.2f} | 24h: {change_24h:+.1f}%
RSI: {current_rsi:.0f} | Trend: {trend}
Holdings: {amount:.4f} {asset} = ${value_usd:.2f}

Recommend: HOLD/SELL/BUY_MORE|confidence(1-10)|reason(15w max)"""

                messages = [
                    {"role": "system", "content": "Portfolio advisor. Format: ACTION|NUM|reason"},
                    {"role": "user", "content": ai_prompt}
                ]
                
                client = get_openai_client()
                if client:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        max_tokens=40,
                        temperature=0.3
                    )
                    
                    ai_response = response.choices[0].message.content.strip()
                    parts = ai_response.split('|')
                    
                    if len(parts) >= 3:
                        action = parts[0].strip().upper()
                        confidence = int(parts[1].strip()) if parts[1].strip().isdigit() else 5
                        reason = parts[2].strip()
                        
                        # Emoji for action
                        if action == "SELL":
                            action_emoji = "🔴 SELL"
                        elif action == "BUY_MORE" or action == "BUY":
                            action_emoji = "🟢 BUY MORE"
                        else:
                            action_emoji = "🟡 HOLD"
                        
                        recommendations.append({
                            'symbol': symbol,
                            'action': action,
                            'confidence': confidence,
                            'reason': reason,
                            'value': value_usd
                        })
                        
                        portfolio_msg += (
                            f"{'='*30}\n"
                            f"💹 {symbol}\n"
                            f"Amount: {amount:.4f} {asset}\n"
                            f"Value: ${value_usd:.2f}\n"
                            f"Price: ${current_price:.2f} ({change_24h:+.1f}% 24h)\n"
                            f"RSI: {current_rsi:.0f} | {trend}\n\n"
                            f"🤖 AI: {action_emoji} ({confidence}/10)\n"
                            f"💡 {reason}\n\n"
                        )
                    else:
                        portfolio_msg += f"⚠️ {symbol}: AI error\n\n"
                else:
                    portfolio_msg += f"⚠️ {symbol}: AI unavailable\n\n"
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                portfolio_msg += f"❌ {symbol}: Error - {str(e)[:50]}\n\n"
        
        # Summary
        portfolio_msg += f"{'='*30}\n"
        portfolio_msg += f"📊 TOTAL VALUE: ${total_value:.2f}\n\n"
        
        # Top recommendations
        if recommendations:
            sells = [r for r in recommendations if r['action'] == 'SELL' and r['confidence'] >= 7]
            buys = [r for r in recommendations if r['action'] in ['BUY_MORE', 'BUY'] and r['confidence'] >= 7]
            
            if sells:
                portfolio_msg += "⚠️ URGENT SELLS:\n"
                for r in sorted(sells, key=lambda x: x['confidence'], reverse=True)[:3]:
                    portfolio_msg += f"  • {r['symbol']} ({r['confidence']}/10): {r['reason'][:30]}\n"
                portfolio_msg += "\n"
            
            if buys:
                portfolio_msg += "💰 GOOD BUYS:\n"
                for r in sorted(buys, key=lambda x: x['confidence'], reverse=True)[:3]:
                    portfolio_msg += f"  • {r['symbol']} ({r['confidence']}/10): {r['reason'][:30]}\n"
                portfolio_msg += "\n"
        
        await update.message.reply_text(portfolio_msg[:4000])  # Telegram limit
        
    except Exception as e:
        logger.error(f"Portfolio analysis error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Error analyzing portfolio: {str(e)}")

async def approve_trade_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle both /approve_TRADEID command and button callback"""
    # Support both message commands and callback queries
    if update.callback_query:
        query = update.callback_query
        await query.answer()  # Acknowledge the button press
        trade_id = query.data.split('_')[1]
        user_id = query.from_user.id
        message_target = query
    else:
        trade_id = update.message.text.split('_')[1]
        user_id = update.message.from_user.id
        message_target = update
    
    agent = context.bot_data['agent']
    
    if trade_id in agent.trade_confirmation_needed:
        trade = agent.trade_confirmation_needed.pop(trade_id)
        
        # Update signal status in database
        agent.db.update_signal_status(trade_id, 'approved')
        
        # Use the actual trade parameters
        amount = trade['amount']
        
        if agent.paper_trading:
            # PAPER TRADING - имитация сделки
            # ✅ Используем execute_trade для добавления в active_positions
            success = agent.execute_trade(trade_id)
            
            if success:
                reply_text = (
                    f"ИМИТАЦИЯ: Сделка одобрена\n"
                    f"{trade['side']} {trade['symbol']}\n"
                    f"Цена входа: ${trade['price']:.2f}\n"
                    f"Количество: {amount:.6f}\n"
                    f"Сумма: ${trade['usdt_amount']:.2f}\n"
                    f"Комиссия: ${trade['fee']:.2f}\n\n"
                    f"✅ Позиция добавлена в мониторинг"
                )
                logger.info(f"PAPER TRADE: {trade['side']} {trade['symbol']} at ${trade['price']}")
            else:
                reply_text = "❌ Ошибка при выполнении сделки"
        else:
            # REAL TRADING - реальная сделка
            order = await agent.place_order(trade['symbol'], trade['side'], amount)
            
            if order:
                agent.db.save_trade(
                    trade_id, trade['symbol'], trade['side'],
                    trade['price'], amount, trade['usdt_amount'], trade['fee'],
                    mode='live'
                )
                
                reply_text = (
                    f"РЕАЛЬНАЯ СДЕЛКА: Выполнено\n"
                    f"{trade['side']} {trade['symbol']}\n"
                    f"Цена: ${trade['price']:.2f}\n"
                    f"Количество: {amount:.6f}\n"
                    f"Сумма: ${trade['usdt_amount']:.2f}"
                )
                logger.info(f"LIVE TRADE: {trade['side']} {trade['symbol']} - Order ID: {order.get('id')}")
            else:
                reply_text = f"ОШИБКА: Не удалось разместить сделку {trade['side']} {trade['symbol']}"
        
        if update.callback_query:
            await query.edit_message_text(reply_text)
        else:
            await update.message.reply_text(reply_text)
    else:
        reply_text = "Сделка уже обработана или ID неверный"
        if update.callback_query:
            await query.edit_message_text(reply_text)
        else:
            await update.message.reply_text(reply_text)
    
    logger.info(f"Trade {trade_id} approved by user {user_id}")

async def reject_trade_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle both /reject_TRADEID command and button callback"""
    # Support both message commands and callback queries
    if update.callback_query:
        query = update.callback_query
        await query.answer()
        trade_id = query.data.split('_')[1]
        user_id = query.from_user.id
    else:
        trade_id = update.message.text.split('_')[1]
        user_id = update.message.from_user.id
    
    agent = context.bot_data['agent']
    
    if trade_id in agent.trade_confirmation_needed:
        trade = agent.trade_confirmation_needed.pop(trade_id)
        
        # Update signal status in database
        agent.db.update_signal_status(trade_id, 'rejected')
        
        reply_text = f"Сделка отклонена: {trade['side']} {trade['symbol']}"
        if update.callback_query:
            await query.edit_message_text(reply_text)
        else:
            await update.message.reply_text(reply_text)
    else:
        reply_text = "Сделка уже обработана или ID неверный"
        if update.callback_query:
            await query.edit_message_text(reply_text)
        else:
            await update.message.reply_text(reply_text)
    
    logger.info(f"Trade {trade_id} rejected by user {user_id}")

# 🤖 ========================================
# AUTO_TRADE TELEGRAM COMMANDS
# ========================================

async def auto_trade_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """🤖 Show AUTO_TRADE status and configuration"""
    agent = context.bot_data['agent']
    status = agent.autonomous.get_status()
    
    # Emoji статусы
    enabled_emoji = "✅ ENABLED" if status['enabled'] else "❌ DISABLED"
    paused_emoji = "🚨 PAUSED" if status['emergency_paused'] else ""
    mode_emoji = "⚡ AGGRESSIVE" if status['aggressive_mode'] else "🛡️ CONSERVATIVE"
    
    message = f'''
🤖 **AUTO_TRADE STATUS**

{enabled_emoji} {paused_emoji}
{mode_emoji}

📊 **Statistics:**
• Trades this hour: {status['trades_this_hour']}/{status['max_trades_per_hour']}
• Last trade: {status['last_trade'] or 'Never'}

⚙️ **Configuration:**
• Min confidence: {status['min_confidence']}/10
• Whitelist: {status['whitelist_count']} symbols
• Blacklist: {status['blacklist_count']} symbols

💡 **Commands:**
/auto_toggle - Enable/Disable AUTO_TRADE
/auto_aggressive - Toggle aggressive mode
/auto_emergency - Emergency stop AUTO_TRADE
'''
    await update.message.reply_text(message)

async def auto_trade_toggle_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """🤖 Toggle AUTO_TRADE on/off"""
    agent = context.bot_data['agent']
    agent.autonomous.enabled = not agent.autonomous.enabled
    
    status = '✅ ENABLED' if agent.autonomous.enabled else '❌ DISABLED'
    
    message = f"🤖 AUTO_TRADE: {status}"
    
    if agent.autonomous.enabled:
        message += "\n\n⚠️ Bot will now execute trades AUTOMATICALLY without your approval!"
        message += f"\n• Min confidence: {agent.autonomous.min_confidence}/10"
        message += f"\n• Max {agent.autonomous.max_trades_per_hour} trades/hour"
        message += "\n\n💡 Use /auto_emergency to stop immediately"
    else:
        message += "\n\n✅ Bot will now ASK for your approval before each trade"
    
    await update.message.reply_text(message)
    logger.info(f"AUTO_TRADE toggled: {status} by user {update.message.from_user.id}")

async def auto_trade_emergency_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """🚨 EMERGENCY STOP AUTO_TRADE"""
    agent = context.bot_data['agent']
    agent.autonomous.emergency_stop(reason="Manual Telegram command")
    
    message = "🚨 **EMERGENCY STOP ACTIVATED!**\n\n"
    message += "✅ AUTO_TRADE stopped immediately\n"
    message += "🔒 All automatic trading blocked\n\n"
    message += "💡 Use /auto_toggle to resume trading"
    
    await update.message.reply_text(message)
    logger.critical(f"AUTO_TRADE emergency stop by user {update.message.from_user.id}")

async def auto_trade_aggressive_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """⚡ Toggle aggressive/conservative mode"""
    agent = context.bot_data['agent']
    agent.autonomous.set_aggressive(not agent.autonomous.aggressive_mode)
    
    mode = "⚡ AGGRESSIVE" if agent.autonomous.aggressive_mode else "🛡️ CONSERVATIVE"
    
    message = f"🤖 AUTO_TRADE MODE: {mode}\n\n"
    
    if agent.autonomous.aggressive_mode:
        message += "⚡ Aggressive mode:\n"
        message += "• Accepts confidence 8+/10\n"
        message += "• More trades, higher risk\n"
        message += "• Recommended for strong markets"
    else:
        message += "🛡️ Conservative mode:\n"
        message += f"• Requires confidence {agent.autonomous.min_confidence}+/10\n"
        message += "• Fewer trades, safer\n"
        message += "• Recommended for volatile markets"
    
    await update.message.reply_text(message)
    logger.info(f"AUTO_TRADE mode changed to {mode} by user {update.message.from_user.id}")

# ========================================
# PERFORMANCE ANALYSIS COMMANDS (Phase 3)
# ========================================

async def performance_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """📊 Show daily performance report"""
    agent = context.bot_data['agent']
    
    if not hasattr(agent, 'performance'):
        await update.message.reply_text("⚠️ PerformanceAnalyzer не инициализирован")
        return
    
    try:
        report = agent.performance.generate_daily_report()
        await update.message.reply_text(report, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка генерации отчета: {e}")
        logger.error(f"Performance report error: {e}")

async def analytics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """📈 Show 30-day analytics"""
    agent = context.bot_data['agent']
    
    if not hasattr(agent, 'performance'):
        await update.message.reply_text("⚠️ PerformanceAnalyzer не инициализирован")
        return
    
    try:
        analysis = agent.performance.analyze_closed_trades(days=30)
        
        if 'message' in analysis:
            await update.message.reply_text(analysis['message'])
            return
        
        # Format best/worst trades
        best_trade = analysis.get('best_trade', {})
        worst_trade = analysis.get('worst_trade', {})
        
        best_text = f"{best_trade.get('symbol', 'N/A')} (+${best_trade.get('pnl', 0):.2f})" if best_trade else 'N/A'
        worst_text = f"{worst_trade.get('symbol', 'N/A')} (${worst_trade.get('pnl', 0):.2f})" if worst_trade else 'N/A'
        
        message = f"""📊 *30-DAY ANALYTICS*

📊 Total Trades: {analysis.get('total_trades', 0)}
✅ Winning: {analysis.get('winning_trades', 0)}
❌ Losing: {analysis.get('losing_trades', 0)}
📈 Win Rate: {analysis.get('win_rate', 0):.1f}%

💰 Total P&L: ${analysis.get('total_pnl', 0):.2f}
📊 Avg Trade: ${analysis.get('avg_pnl', 0):.2f}
📈 ROI: {analysis.get('roi', 0):.1f}%

📊 Sharpe Ratio: {analysis.get('sharpe_ratio', 0):.2f}
📉 Max Drawdown: ${analysis.get('max_drawdown', 0):.2f} ({analysis.get('max_drawdown_pct', 0):.1f}%)
⏱ Avg Duration: {analysis.get('avg_duration_hours', 0):.1f}h

🏆 Best Trade: {best_text}
💔 Worst Trade: {worst_text}
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка анализа: {e}")
        logger.error(f"Analytics error: {e}")

async def recommendations_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """💡 Get AI recommendations"""
    agent = context.bot_data['agent']
    
    if not hasattr(agent, 'performance'):
        await update.message.reply_text("⚠️ PerformanceAnalyzer не инициализирован")
        return
    
    try:
        recs = agent.performance.get_recommendations()
        
        if not recs:
            await update.message.reply_text("💡 Недостаточно данных для рекомендаций. Продолжайте торговать! (минимум 10 сделок)")
            return
        
        message = "💡 *РЕКОМЕНДАЦИИ ОТ ИИ:*\n\n"
        for i, rec in enumerate(recs, 1):
            message += f"{i}. {rec}\n\n"
        
        await update.message.reply_text(message, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка получения рекомендаций: {e}")
        logger.error(f"Recommendations error: {e}")

async def filters_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """🔍 Analyze filter effectiveness"""
    agent = context.bot_data['agent']
    
    if not hasattr(agent, 'performance'):
        await update.message.reply_text("⚠️ PerformanceAnalyzer не инициализирован")
        return
    
    try:
        filters = agent.performance.analyze_filter_effectiveness(days=30)
        
        if 'message' in filters:
            await update.message.reply_text(filters['message'])
            return
        
        message = "🔍 *FILTER EFFECTIVENESS (30 days):*\n\n"
        
        for filter_name, stats in filters.items():
            effectiveness = stats.get('effectiveness', 'UNKNOWN')
            emoji = '✅' if effectiveness == 'HIGH' else '⚠️' if effectiveness == 'MEDIUM' else '❌'
            
            message += f"{emoji} *{filter_name}*:\n"
            message += f"  Trades: {stats.get('trades', 0)}\n"
            message += f"  Win Rate: {stats.get('win_rate', 0):.1f}%\n"
            message += f"  Avg P&L: ${stats.get('avg_pnl', 0):.2f}\n"
            message += f"  Effectiveness: {effectiveness}\n\n"
        
        await update.message.reply_text(message, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка анализа фильтров: {e}")
        logger.error(f"Filters error: {e}")

async def optimize_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """⚙️ Get optimal parameters from historical data"""
    agent = context.bot_data['agent']
    
    if not hasattr(agent, 'performance'):
        await update.message.reply_text("⚠️ PerformanceAnalyzer не инициализирован")
        return
    
    try:
        optimal = agent.performance.get_optimal_parameters(days=30)
        
        if 'message' in optimal:
            await update.message.reply_text(optimal['message'])
            return
        
        message = f"""⚙️ *OPTIMAL PARAMETERS (30 days):*

🎯 Confidence Range:
  Best: {optimal.get('best_confidence_range', 'N/A')}
  Win Rate: {optimal.get('best_win_rate', 0):.1f}%
  Avg P&L: ${optimal.get('best_avg_pnl', 0):.2f}

📊 Stop Loss:
  Optimal: {optimal.get('optimal_stop_loss_pct', 0):.1f}%
  (Based on {optimal.get('stop_loss_trades', 0)} trades)

🎯 Take Profit:
  Optimal: {optimal.get('optimal_take_profit_pct', 0):.1f}%
  (Based on {optimal.get('take_profit_trades', 0)} trades)

💰 Risk/Reward:
  Average: {optimal.get('avg_risk_reward', 0):.2f}
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка оптимизации: {e}")
        logger.error(f"Optimize error: {e}")

# ========================================
# ADAPTIVE LEARNING COMMANDS (Phase 4)
# ========================================

async def adaptive_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """🧠 Show adaptive learning status"""
    agent = context.bot_data['agent']
    
    if not hasattr(agent, 'adaptive') or agent.adaptive is None:
        await update.message.reply_text("⚠️ AdaptiveLearning не инициализирован")
        return
    
    try:
        status = agent.adaptive.get_status()
        
        message = f"""🧠 *ADAPTIVE LEARNING STATUS*

✅ Initialized: {status['is_trained']}
📁 Model Path: `{status['model_path']}`
💾 Model Exists: {status['model_exists']}
🌍 Environment: {status['env_created']}
🤖 Model Loaded: {status['model_loaded']}

{'✅ Model готов к использованию!' if status['is_trained'] else '⚠️ Model требует обучения. Используйте /train_model'}
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")
        logger.error(f"Adaptive status error: {e}")

async def train_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """🎓 Train the adaptive learning model"""
    agent = context.bot_data['agent']
    
    if not hasattr(agent, 'adaptive') or agent.adaptive is None:
        await update.message.reply_text("⚠️ AdaptiveLearning не инициализирован")
        return
    
    try:
        await update.message.reply_text("🎓 Начинаю обучение RL модели...\nЭто займет 1-2 минуты ⏳")
        
        # Train with 5000 timesteps (quick training)
        stats = agent.adaptive.train(total_timesteps=5000, verbose=0)
        
        message = f"""✅ *TRAINING COMPLETE!*

📊 Total Timesteps: {stats['total_timesteps']}
🏆 Mean Reward: {stats['mean_reward']:.2f}
📈 Episodes: {len(stats['episode_rewards'])}

Модель сохранена и готова к использованию!
Используйте /adaptive_predict для получения оптимальных параметров.
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка обучения: {e}")
        logger.error(f"Training error: {e}")

async def adaptive_predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """🔮 Get AI-predicted optimal parameters"""
    agent = context.bot_data['agent']
    
    if not hasattr(agent, 'adaptive') or agent.adaptive is None:
        await update.message.reply_text("⚠️ AdaptiveLearning не инициализирован")
        return
    
    if not agent.adaptive.is_trained:
        await update.message.reply_text("⚠️ Модель не обучена. Используйте /train_model сначала")
        return
    
    try:
        await update.message.reply_text("🔮 Анализирую текущее состояние рынка...")
        
        params = agent.adaptive.predict_optimal_parameters()
        
        # Compare with current params
        current_confidence = agent.autonomous.min_confidence if hasattr(agent, 'autonomous') else 7.5
        current_aggressive = agent.autonomous.aggressive_mode if hasattr(agent, 'autonomous') else False
        
        message = f"""🔮 *AI-PREDICTED OPTIMAL PARAMETERS*

🎯 MIN_CONFIDENCE:
  Current: {current_confidence}
  Recommended: {params['min_confidence']:.1f}
  {'✅ Оптимально' if abs(params['min_confidence'] - current_confidence) < 0.5 else '⚠️ Рекомендуется изменить'}

📉 STOP_LOSS:
  Recommended: {params['stop_loss_pct']:.1f}%

📈 TAKE_PROFIT:
  Recommended: {params['take_profit_pct']:.1f}%

💰 POSITION_SIZE:
  Recommended: {params['position_size_pct']:.1f}%

⚡ MODE:
  Current: {'AGGRESSIVE' if current_aggressive else 'CONSERVATIVE'}
  Recommended: {'AGGRESSIVE' if params['aggressive'] else 'CONSERVATIVE'}
  {'✅ Оптимально' if params['aggressive'] == current_aggressive else '⚠️ Рекомендуется изменить'}

💡 Используйте /apply_adaptive для применения этих параметров
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка предсказания: {e}")
        logger.error(f"Prediction error: {e}")

async def apply_adaptive_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """✅ Apply AI-predicted parameters"""
    agent = context.bot_data['agent']
    
    if not hasattr(agent, 'adaptive') or agent.adaptive is None:
        await update.message.reply_text("⚠️ AdaptiveLearning не инициализирован")
        return
    
    if not agent.adaptive.is_trained:
        await update.message.reply_text("⚠️ Модель не обучена. Используйте /train_model сначала")
        return
    
    try:
        params = agent.adaptive.predict_optimal_parameters()
        
        # Apply parameters to autonomous trader
        if hasattr(agent, 'autonomous') and agent.autonomous:
            agent.autonomous.min_confidence = params['min_confidence']
            agent.autonomous.set_aggressive(params['aggressive'])
            
            message = f"""✅ *PARAMETERS APPLIED!*

🤖 AUTO_TRADE обновлен:
  MIN_CONFIDENCE: {params['min_confidence']:.1f}
  MODE: {'AGGRESSIVE' if params['aggressive'] else 'CONSERVATIVE'}

⚠️ Другие параметры (stop_loss, take_profit, position_size) 
требуют обновления в .env файле:
  STOP_LOSS_PCT={params['stop_loss_pct']:.1f}
  TAKE_PROFIT_PCT={params['take_profit_pct']:.1f}
  POSITION_SIZE_PCT={params['position_size_pct']:.1f}

Рекомендуется перезапустить бота после обновления .env
"""
        else:
            message = "⚠️ AUTO_TRADE не активен. Параметры сохранены но не применены."
        
        await update.message.reply_text(message, parse_mode='Markdown')
        logger.info(f"Applied adaptive parameters: {params}")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка применения: {e}")
        logger.error(f"Apply error: {e}")

async def evaluate_adaptive_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """📊 Evaluate adaptive learning model"""
    agent = context.bot_data['agent']
    
    if not hasattr(agent, 'adaptive') or agent.adaptive is None:
        await update.message.reply_text("⚠️ AdaptiveLearning не инициализирован")
        return
    
    if not agent.adaptive.is_trained:
        await update.message.reply_text("⚠️ Модель не обучена. Используйте /train_model сначала")
        return
    
    try:
        await update.message.reply_text("📊 Оцениваю производительность модели...\nЭто займет ~30 секунд ⏳")
        
        results = agent.adaptive.evaluate(n_episodes=5)
        
        message = f"""📊 *MODEL EVALUATION*

🎯 Episodes: {results['n_episodes']}
🏆 Mean Reward: {results['mean_reward']:.2f}
📊 Std Reward: {results['std_reward']:.2f}
⏱ Mean Length: {results['mean_length']:.1f} steps

{'✅ Model работает отлично!' if results['mean_reward'] > 0 else '⚠️ Model требует дообучения'}
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка оценки: {e}")
        logger.error(f"Evaluation error: {e}")

# ========================================
# MARKET REGIME COMMANDS (Phase 5)
# ========================================

async def regime_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """📊 Detect current market regime"""
    agent = context.bot_data['agent']
    
    if not hasattr(agent, 'regime_manager') or agent.regime_manager is None:
        await update.message.reply_text("⚠️ MarketRegimeManager не инициализирован")
        return
    
    try:
        await update.message.reply_text("📊 Анализирую текущий режим рынка...\nЭто займет ~10 секунд ⏳")
        
        # Detect regime
        regime = agent.regime_manager.detect_regime(agent.exchange, "BTC/USDT")
        
        # Get strategy for this regime
        strategy = agent.regime_manager.get_current_strategy()
        should_trade = agent.regime_manager.should_trade_now()
        
        # Get regime icon
        icons = {
            'TREND_UP': '📈',
            'TREND_DOWN': '📉',
            'RANGE': '↔️',
            'HIGH_VOLATILITY': '⚡',
            'CRASH': '🚨',
            'UNKNOWN': '❓'
        }
        icon = icons.get(regime.value, '❓')
        
        message = f"""{icon} *MARKET REGIME: {regime.value}*

{strategy['description']}

📊 *Trading Parameters:*
  Confidence Threshold: {strategy['confidence_threshold']}
  Position Size: {strategy['position_size_multiplier']}x
  Stop Loss: {strategy['stop_loss_multiplier']}x
  Take Profit: {strategy['take_profit_multiplier']}x
  Max Positions: {strategy['max_positions']}

{'✅ Торговля РЕКОМЕНДУЕТСЯ' if should_trade else '🚨 Торговля НЕ РЕКОМЕНДУЕТСЯ'}
{'⚡ Aggressive mode' if strategy['aggressive_mode'] else '🛡️ Conservative mode'}
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка определения режима: {e}")
        logger.error(f"Regime detection error: {e}")

async def regime_fit_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """🎓 Fit HMM model on market data"""
    agent = context.bot_data['agent']
    
    if not hasattr(agent, 'regime_manager') or agent.regime_manager is None:
        await update.message.reply_text("⚠️ MarketRegimeManager не инициализирован")
        return
    
    try:
        await update.message.reply_text("🎓 Обучаю HMM модель на исторических данных...\nЭто займет ~20 секунд ⏳")
        
        # Fit model
        success = agent.regime_manager.fit_model(agent.exchange, "BTC/USDT")
        
        if success:
            # Get status
            status = agent.regime_manager.detector.get_status()
            
            message = f"""✅ *HMM MODEL FITTED!*

📊 Model Details:
  Regimes: {status['n_regimes']}
  Status: {'Ready' if status['is_fitted'] else 'Not fitted'}

🗺️ Regime Mapping:
"""
            
            for state, regime in status['regime_mapping'].items():
                message += f"  State {state} → {regime}\n"
            
            message += "\nИспользуйте /regime для определения текущего режима"
            
            await update.message.reply_text(message, parse_mode='Markdown')
        else:
            await update.message.reply_text("❌ Ошибка обучения модели. Проверьте логи.")
    
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")
        logger.error(f"Regime fit error: {e}")

async def regime_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """📈 Show regime detection history"""
    agent = context.bot_data['agent']
    
    if not hasattr(agent, 'regime_manager') or agent.regime_manager is None:
        await update.message.reply_text("⚠️ MarketRegimeManager не инициализирован")
        return
    
    try:
        # Get statistics
        stats = agent.regime_manager.detector.get_regime_statistics()
        
        if 'message' in stats:
            await update.message.reply_text(stats['message'])
            return
        
        # Format message
        message = f"""📈 *REGIME DETECTION HISTORY*

🎯 Current: {stats['current_regime']} ({stats['current_probability']:.2f})
📊 Total Detections: {stats['total_detections']}

📊 *Regime Distribution:*
"""
        
        # Sort by percentage
        for regime, pct in sorted(stats['regime_percentages'].items(), key=lambda x: x[1], reverse=True):
            count = stats['regime_counts'][regime]
            message += f"  {regime}: {pct:.1f}% ({count})\n"
        
        # Recent regimes
        message += f"\n🕐 *Recent Regimes (last 10):*\n  "
        message += " → ".join(stats['recent_regimes'][-10:])
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")
        logger.error(f"Regime history error: {e}")

async def regime_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """📊 Show regime statistics from database"""
    agent = context.bot_data['agent']
    
    if not hasattr(agent, 'regime_manager') or agent.regime_manager is None:
        await update.message.reply_text("⚠️ MarketRegimeManager не инициализирован")
        return
    
    try:
        # Get regime history from database
        df = agent.regime_manager.get_regime_from_db(days=7)
        
        if len(df) == 0:
            await update.message.reply_text("📊 Нет данных в базе. Используйте /regime для начала отслеживания.")
            return
        
        # Calculate statistics
        regime_counts = df['regime'].value_counts()
        total = len(df)
        
        message = f"""📊 *REGIME STATISTICS (7 days)*

📈 Total Records: {total}
🕐 First: {df['timestamp'].min()}
🕐 Last: {df['timestamp'].max()}

📊 *Distribution:*
"""
        
        for regime, count in regime_counts.items():
            pct = (count / total) * 100
            message += f"  {regime}: {pct:.1f}% ({count})\n"
        
        # Most common regime
        most_common = regime_counts.index[0]
        message += f"\n🏆 Most Common: {most_common}"
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")
        logger.error(f"Regime stats error: {e}")

# ========================================
# 💭 SENTIMENT ANALYSIS COMMANDS
# ========================================

async def sentiment_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показать текущий sentiment рынка"""
    agent = context.bot_data['agent']
    
    if not hasattr(agent, 'sentiment_analyzer') or agent.sentiment_analyzer is None:
        await update.message.reply_text("⚠️ SentimentAnalyzer не инициализирован")
        return
    
    try:
        await update.message.reply_text("📊 Анализирую настроения рынка...")
        
        # Get overall sentiment
        sentiment = agent.sentiment_analyzer.get_overall_sentiment()
        
        if 'error' in sentiment:
            await update.message.reply_text(f"❌ Ошибка: {sentiment['error']}")
            return
        
        # Get recommendation
        recommendation = agent.sentiment_analyzer.get_trading_recommendation()
        
        # Format message
        message = f"""💭 *MARKET SENTIMENT*

📊 Overall Score: {sentiment['overall_score']:.1f}/100
📈 Level: {sentiment['level']}

{recommendation['description']}

🔧 Trading Adjustments:
  • Confidence: {recommendation['confidence_adjustment']:+.1f}
  • Position Size: {recommendation['position_size_multiplier']:.1f}x
  • Aggressive: {recommendation['aggressive']}

💡 {recommendation['reasoning']}

📌 Sources Used:
"""
        
        for source, value in sentiment['sources'].items():
            weight = sentiment['weights'].get(source, 0) * 100
            message += f"  • {source}: {value:.1f} (вес: {weight:.0f}%)\n"
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")
        logger.error(f"Sentiment command error: {e}")


async def fear_greed_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показать Fear & Greed Index"""
    agent = context.bot_data['agent']
    
    if not hasattr(agent, 'sentiment_analyzer') or agent.sentiment_analyzer is None:
        await update.message.reply_text("⚠️ SentimentAnalyzer не инициализирован")
        return
    
    try:
        # Get Fear & Greed Index
        fear_greed = agent.sentiment_analyzer.get_fear_greed_index(use_cache=False)
        
        if 'error' in fear_greed:
            await update.message.reply_text(f"❌ Ошибка: {fear_greed['error']}")
            return
        
        # Get history for trend
        trend_data = agent.sentiment_analyzer.get_sentiment_trend(days=7)
        
        # Format message
        value = fear_greed['value']
        classification = fear_greed['value_classification']
        
        # Add emoji based on value
        if value < 25:
            emoji = "😱"
        elif value < 45:
            emoji = "😟"
        elif value < 55:
            emoji = "😐"
        elif value < 75:
            emoji = "😊"
        else:
            emoji = "🤑"
        
        message = f"""📊 *CRYPTO FEAR & GREED INDEX*

{emoji} Current: {value}/100
📈 Classification: {classification}

⏰ Updated: {fear_greed['timestamp'].strftime('%Y-%m-%d %H:%M')}

"""
        
        # Add trend if available
        if 'error' not in trend_data:
            trend = trend_data['trend']
            change = trend_data['change']
            
            if trend == 'IMPROVING':
                trend_emoji = "📈"
                trend_text = "Улучшается"
            elif trend == 'WORSENING':
                trend_emoji = "📉"
                trend_text = "Ухудшается"
            else:
                trend_emoji = "↔️"
                trend_text = "Стабильный"
            
            message += f"""{trend_emoji} 7-Day Trend: {trend_text}
📊 Change: {change:+.0f} points
📊 Average: {trend_data['average']:.1f}
"""
        
        # Add interpretation
        message += "\n💡 Интерпретация:\n"
        
        if value < 25:
            message += "  • Extreme Fear - возможность покупки\n"
            message += "  • Рынок часто перепродан\n"
        elif value < 45:
            message += "  • Fear - осторожная покупка\n"
            message += "  • Возможно недооценен\n"
        elif value < 55:
            message += "  • Neutral - нормальная торговля\n"
        elif value < 75:
            message += "  • Greed - будьте осторожны\n"
            message += "  • Возможна коррекция\n"
        else:
            message += "  • Extreme Greed - высокий риск\n"
            message += "  • Рынок может быть перекуплен\n"
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")
        logger.error(f"Fear & Greed command error: {e}")


async def sentiment_trend_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показать тренд sentiment за 7 дней"""
    agent = context.bot_data['agent']
    
    if not hasattr(agent, 'sentiment_analyzer') or agent.sentiment_analyzer is None:
        await update.message.reply_text("⚠️ SentimentAnalyzer не инициализирован")
        return
    
    try:
        # Get sentiment trend
        trend = agent.sentiment_analyzer.get_sentiment_trend(days=7)
        
        if 'error' in trend:
            await update.message.reply_text(f"❌ Ошибка: {trend['error']}")
            return
        
        # Format message
        message = f"""📈 *SENTIMENT TREND (7 days)*

📊 Trend: {trend['trend']}

📌 Current: {trend['current']}
📌 7 Days Ago: {trend['oldest']}
📊 Change: {trend['change']:+.1f}

📊 Average: {trend['average']:.1f}
📊 Volatility: {trend['volatility']:.1f}

📜 History:
```
"""
        
        # Add history
        for item in trend['history'][:7]:
            date = item['timestamp'].strftime('%m-%d')
            value = item['value']
            classification = item['classification']
            message += f"{date}: {value:3d} ({classification})\n"
        
        message += "```"
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")
        logger.error(f"Sentiment trend command error: {e}")

# ========================================
# 🤖 INTELLIGENT AI COMMANDS
# ========================================

async def ai_predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показать AI предсказание цены"""
    agent = context.bot_data['agent']
    
    if not hasattr(agent, 'intelligent_ai') or agent.intelligent_ai is None:
        await update.message.reply_text("⚠️ IntelligentAI не инициализирован")
        return
    
    try:
        await update.message.reply_text("🤖 Анализирую рынок с помощью AI...")
        
        # Get market data
        symbol = 'BTC/USDT'
        ohlcv = agent.exchange.fetch_ohlcv(symbol, timeframe='1h', limit=200)
        import pandas as pd
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Get ensemble prediction
        prediction = agent.intelligent_ai.get_ensemble_prediction(df)
        
        current_price = prediction['current_price']
        signal = prediction['final_signal']
        confidence = prediction['final_confidence']
        
        # Format message
        if signal in ['STRONG_BUY', 'BUY']:
            emoji = "🟢"
        elif signal in ['STRONG_SELL', 'SELL']:
            emoji = "🔴"
        else:
            emoji = "⚪"
        
        message = f"""🤖 *AI PREDICTION*

{emoji} Signal: {signal}
📊 Confidence: {confidence:.1%}

💰 Current Price: ${current_price:.2f}

"""
        
        # LSTM prediction
        if 'lstm' in prediction['predictions']:
            lstm = prediction['predictions']['lstm']
            message += f"""📈 *LSTM Model:*
  Predicted: ${lstm['predicted_price']:.2f}
  Change: {lstm['change_pct']:+.2f}%
  Signal: {lstm['signal']}
  Weight: {lstm['weight']:.0%}

"""
        
        # Pattern recognition
        if 'patterns' in prediction['predictions']:
            patterns = prediction['predictions']['patterns']
            message += f"""🎨 *Pattern Recognition:*
  Signal: {patterns['signal']}
  Patterns found: {patterns['patterns_detected']}
  BUY signals: {patterns['buy_count']}
  SELL signals: {patterns['sell_count']}
  Weight: {patterns['weight']:.0%}

"""
        
        # Technical indicators
        if 'technical' in prediction['predictions']:
            tech = prediction['predictions']['technical']
            message += f"""📊 *Technical Analysis:*
  Signal: {tech['signal']}
  Confidence: {tech['confidence']:.0%}
  Weight: {tech['weight']:.0%}

"""
        
        # Recommendation
        message += "💡 *Recommendation:*\n"
        
        if signal in ['STRONG_BUY', 'BUY']:
            message += "  🟢 Покупка рекомендуется\n"
            if confidence > 0.7:
                message += "  ✅ Высокая уверенность\n"
        elif signal in ['STRONG_SELL', 'SELL']:
            message += "  🔴 Продажа рекомендуется\n"
            if confidence > 0.7:
                message += "  ✅ Высокая уверенность\n"
        else:
            message += "  ⚪ Ожидайте лучшего момента\n"
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")
        logger.error(f"AI predict command error: {e}")


async def ai_train_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обучить LSTM модель"""
    agent = context.bot_data['agent']
    
    if not hasattr(agent, 'intelligent_ai') or agent.intelligent_ai is None:
        await update.message.reply_text("⚠️ IntelligentAI не инициализирован")
        return
    
    try:
        await update.message.reply_text("🤖 Начинаю обучение LSTM модели...\n⏰ Это займет ~1-2 минуты")
        
        # Get training data (1000 candles)
        symbol = 'BTC/USDT'
        ohlcv = agent.exchange.fetch_ohlcv(symbol, timeframe='1h', limit=1000)
        import pandas as pd
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Train LSTM
        result = agent.intelligent_ai.train_lstm(df, epochs=20, batch_size=32)
        
        if 'error' in result:
            await update.message.reply_text(f"❌ Ошибка обучения: {result['error']}")
            return
        
        # Format message
        message = f"""🤖 *LSTM TRAINING COMPLETE*

✅ Model trained successfully!

📊 Training Stats:
  • Training samples: {result['train_samples']}
  • Test samples: {result['test_samples']}
  • Epochs: {result['epochs']}
  • Final train loss: {result['final_train_loss']:.6f}
  • Final test loss: {result['final_test_loss']:.6f}

💾 Model saved to: {result['model_path']}

ℹ️ Теперь используйте /ai_predict для предсказаний
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")
        logger.error(f"AI train command error: {e}")


async def ai_patterns_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показать обнаруженные паттерны"""
    agent = context.bot_data['agent']
    
    if not hasattr(agent, 'intelligent_ai') or agent.intelligent_ai is None:
        await update.message.reply_text("⚠️ IntelligentAI не инициализирован")
        return
    
    try:
        # Get market data
        symbol = 'BTC/USDT'
        ohlcv = agent.exchange.fetch_ohlcv(symbol, timeframe='1h', limit=200)
        import pandas as pd
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Detect patterns
        patterns = agent.intelligent_ai.pattern_recognizer.detect_patterns(df)
        
        if 'error' in patterns:
            await update.message.reply_text(f"⚠️ {patterns['error']}")
            return
        
        if len(patterns) == 0:
            await update.message.reply_text("📊 Паттерны не обнаружены в текущих данных")
            return
        
        # Format message
        message = f"""🎨 *DETECTED PATTERNS*

📊 Found {len(patterns)} pattern(s):

"""
        
        for pattern_name, pattern_data in patterns.items():
            if isinstance(pattern_data, dict):
                signal = pattern_data.get('signal', 'N/A')
                confidence = pattern_data.get('confidence', 0)
                description = pattern_data.get('description', '')
                
                if signal == 'BUY':
                    emoji = "🟢"
                elif signal == 'SELL':
                    emoji = "🔴"
                else:
                    emoji = "⚪"
                
                message += f"""{emoji} *{pattern_name.replace('_', ' ').title()}*
  Signal: {signal}
  Confidence: {confidence:.0%}
  {description}

"""
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")


# ============================================
# RISK MANAGEMENT COMMANDS
# ============================================

async def risk_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show comprehensive risk analysis"""
    try:
        if agent.risk_manager is None:
            await update.message.reply_text("❌ Risk Manager недоступен")
            return
        
        # Get current market data
        df = await agent.exchange.fetch_ohlcv(agent.symbol, '1h', limit=500)
        df = pd.DataFrame(df, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        current_price = df['close'].iloc[-1]
        
        # Get Kelly position size
        kelly_size = agent.risk_manager.get_kelly_position_size(agent.symbol, current_price)
        kelly_pct = kelly_size / agent.risk_manager.current_balance
        
        # Calculate VaR
        var_hist = agent.risk_manager.calculate_portfolio_var(df, confidence=0.95, method='historical')
        
        # Calculate ATR-based stop-loss
        sl_long = agent.risk_manager.calculate_atr_stop_loss(df, current_price, 'long')
        tp_long = agent.risk_manager.calculate_atr_take_profit(df, current_price, 'long', risk_reward_ratio=2.0)
        
        # Get portfolio metrics
        metrics = agent.risk_manager.get_portfolio_metrics(df)
        
        # Get status
        status = agent.risk_manager.get_status()
        
        message = f"""💼 *ADVANCED RISK ANALYSIS*

📊 *Kelly Criterion Position Sizing*
• Optimal Size: ${kelly_size:.2f} ({kelly_pct:.1%})
• Max Risk per Trade: {status['max_risk_per_trade']:.1%}
• Kelly Fraction: {status['kelly_fraction']:.0%}

📉 *Value at Risk (95% confidence)*
• 1-day VaR: {var_hist['var_1day_pct']:.2%} (${var_hist['var_1day_usd']:.2f})
• 1-week VaR: {var_hist['var_1week_pct']:.2%} (${var_hist['var_1week_usd']:.2f})
• 1-month VaR: {var_hist['var_1month_pct']:.2%} (${var_hist['var_1month_usd']:.2f})
• Risk Level: {var_hist['interpretation']}

🎯 *ATR-based Stop-Loss/Take-Profit*
• Current Price: ${current_price:.2f}
• Stop-Loss: ${sl_long:.2f} ({(sl_long-current_price)/current_price:.2%})
• Take-Profit: ${tp_long:.2f} ({(tp_long-current_price)/current_price:.2%})
• Risk/Reward Ratio: 1:2

📈 *Portfolio Metrics*
• Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
• Sortino Ratio: {metrics['sortino_ratio']:.2f}
• Max Drawdown: {metrics['max_drawdown']:.2%}
• Annual Volatility: {metrics['volatility_annual']:.2%}
• Risk Level: {metrics['risk_level']}

💰 *Account Status*
• Balance: ${status['current_balance']:.2f}
• Total Trades: {status['total_trades']}
• Win Rate: {status['win_rate']:.1%}
• Total PnL: ${status['total_pnl']:.2f}
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")


async def var_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show detailed Value at Risk analysis"""
    try:
        if agent.risk_manager is None:
            await update.message.reply_text("❌ Risk Manager недоступен")
            return
        
        # Get market data
        df = await agent.exchange.fetch_ohlcv(agent.symbol, '1h', limit=500)
        df = pd.DataFrame(df, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Calculate VaR with both methods
        var_hist_95 = agent.risk_manager.calculate_portfolio_var(df, 0.95, 'historical')
        var_param_95 = agent.risk_manager.calculate_portfolio_var(df, 0.95, 'parametric')
        var_hist_99 = agent.risk_manager.calculate_portfolio_var(df, 0.99, 'historical')
        
        message = f"""📊 *VALUE AT RISK (VaR) ANALYSIS*

🔍 *Historical VaR (95% confidence)*
• 1 день: {var_hist_95['var_1day_pct']:.2%} ≈ ${var_hist_95['var_1day_usd']:.2f}
• 1 неделя: {var_hist_95['var_1week_pct']:.2%} ≈ ${var_hist_95['var_1week_usd']:.2f}
• 1 месяц: {var_hist_95['var_1month_pct']:.2%} ≈ ${var_hist_95['var_1month_usd']:.2f}
• Risk: {var_hist_95['interpretation']}

📐 *Parametric VaR (95% confidence)*
• 1 день: {var_param_95['var_1day_pct']:.2%} ≈ ${var_param_95['var_1day_usd']:.2f}
• 1 неделя: {var_param_95['var_1week_pct']:.2%} ≈ ${var_param_95['var_1week_usd']:.2f}
• 1 месяц: {var_param_95['var_1month_pct']:.2%} ≈ ${var_param_95['var_1month_usd']:.2f}

⚠️ *Conservative VaR (99% confidence)*
• 1 день: {var_hist_99['var_1day_pct']:.2%} ≈ ${var_hist_99['var_1day_usd']:.2f}
• 1 неделя: {var_hist_99['var_1week_pct']:.2%} ≈ ${var_hist_99['var_1week_usd']:.2f}
• 1 месяц: {var_hist_99['var_1month_pct']:.2%} ≈ ${var_hist_99['var_1month_usd']:.2f}

💡 *Что это значит?*
VaR показывает максимальный убыток, который может произойти с заданной вероятностью.
Например, 95% VaR = 0.84% означает, что с вероятностью 95% вы не потеряете больше 0.84% за день.
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")


async def kelly_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show Kelly Criterion analysis"""
    try:
        if agent.risk_manager is None:
            await update.message.reply_text("❌ Risk Manager недоступен")
            return
        
        # Get current price
        ticker = await agent.exchange.fetch_ticker(agent.symbol)
        current_price = ticker['last']
        
        # Get Kelly position size
        kelly_size = agent.risk_manager.get_kelly_position_size(agent.symbol, current_price)
        kelly_pct = kelly_size / agent.risk_manager.current_balance
        
        # Get status for win/loss stats
        status = agent.risk_manager.get_status()
        
        # Calculate stats from trade history
        if len(agent.risk_manager.trade_history) >= 10:
            wins = [t['pnl'] for t in agent.risk_manager.trade_history if t['pnl'] > 0]
            losses = [abs(t['pnl']) for t in agent.risk_manager.trade_history if t['pnl'] < 0]
            
            win_rate = len(wins) / len(agent.risk_manager.trade_history)
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            
            stats_available = True
        else:
            win_rate = status['win_rate']
            avg_win = 0
            avg_loss = 0
            stats_available = False
        
        message = f"""📊 *KELLY CRITERION ANALYSIS*

🎯 *Рекомендуемый размер позиции*
• Optimal Size: ${kelly_size:.2f}
• Percentage: {kelly_pct:.1%} от баланса
• Current Balance: ${status['current_balance']:.2f}

📈 *Статистика трейдинга*
• Total Trades: {status['total_trades']}
• Win Rate: {win_rate:.1%}
• Winning Trades: {status['winning_trades']}
• Losing Trades: {status['losing_trades']}
"""
        
        if stats_available:
            message += f"""
💰 *Средние результаты*
• Avg Win: ${avg_win:.2f}
• Avg Loss: ${avg_loss:.2f}
• Win/Loss Ratio: {avg_win/avg_loss if avg_loss > 0 else 0:.2f}
"""
        
        message += f"""
⚙️ *Настройки Kelly*
• Kelly Fraction: {status['kelly_fraction']:.0%} (консервативный)
• Max Risk: {status['max_risk_per_trade']:.1%}

💡 *Объяснение:*
Kelly Criterion рассчитывает оптимальный размер позиции на основе вашей истории трейдинга.
Использует fractional Kelly (25%) для более консервативного подхода.
"""
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")

        logger.error(f"AI patterns command error: {e}")

# ========================================
# MAIN
# ========================================

# --- Main Function to Run the Bot ---
def main() -> None:
    import sys
    import traceback
    
    # Python 3.14 fix: create event loop if doesn't exist
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    
    # Exception handler
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = handle_exception
    
    try:
        agent = TradingAgent()
        logger.info("Trading Agent initialized successfully")

        # Create the Application and pass your bot's token.
        logger.info("Creating Telegram Application...")
        application = Application.builder().token(agent.telegram_bot_token).build()
        logger.info("Application created")

        # Store the agent instance in bot_data for access in handlers
        application.bot_data['agent'] = agent
        logger.info("Agent stored in bot_data")

        # Register command handlers
        logger.info("Registering handlers...")
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("status", status_command))
        application.add_handler(CommandHandler("positions", positions_command))
        application.add_handler(CommandHandler("portfolio", portfolio_command))
        application.add_handler(CommandHandler("history", history_command))
        application.add_handler(CommandHandler("analyze", analyze_command))
        
        # 🛡️ Safety commands
        application.add_handler(CommandHandler("safety", safety_command))
        application.add_handler(CommandHandler("emergency_stop", emergency_stop_command))
        application.add_handler(CommandHandler("pause", pause_command))
        application.add_handler(CommandHandler("resume", resume_command))
        
        # 📊 Performance Analysis commands (Phase 3)
        application.add_handler(CommandHandler("performance", performance_command))
        application.add_handler(CommandHandler("analytics", analytics_command))
        application.add_handler(CommandHandler("recommendations", recommendations_command))
        application.add_handler(CommandHandler("filters", filters_command))
        application.add_handler(CommandHandler("optimize", optimize_command))
        
        # 🧠 Adaptive Learning commands (Phase 4)
        application.add_handler(CommandHandler("adaptive_status", adaptive_status_command))
        application.add_handler(CommandHandler("train_model", train_model_command))
        application.add_handler(CommandHandler("adaptive_predict", adaptive_predict_command))
        application.add_handler(CommandHandler("apply_adaptive", apply_adaptive_command))
        application.add_handler(CommandHandler("evaluate_adaptive", evaluate_adaptive_command))
        
        # 📊 Market Regime commands (Phase 5)
        application.add_handler(CommandHandler("regime", regime_command))
        application.add_handler(CommandHandler("regime_fit", regime_fit_command))
        application.add_handler(CommandHandler("regime_history", regime_history_command))
        application.add_handler(CommandHandler("regime_stats", regime_stats_command))
        
        # 💭 Sentiment Analysis commands (Phase 6)
        application.add_handler(CommandHandler("sentiment", sentiment_command))
        application.add_handler(CommandHandler("fear_greed", fear_greed_command))
        application.add_handler(CommandHandler("sentiment_trend", sentiment_trend_command))
        
        # 🤖 Intelligent AI commands (Phase 7)
        application.add_handler(CommandHandler("ai_predict", ai_predict_command))
        application.add_handler(CommandHandler("ai_train", ai_train_command))
        application.add_handler(CommandHandler("ai_patterns", ai_patterns_command))
        
        # 💼 Risk Management commands (Phase 8)
        application.add_handler(CommandHandler("risk", risk_command))
        application.add_handler(CommandHandler("var", var_command))
        application.add_handler(CommandHandler("kelly", kelly_command))
        
        # 🤖 AUTO_TRADE commands
        application.add_handler(CommandHandler("auto_status", auto_trade_status_command))
        application.add_handler(CommandHandler("auto_toggle", auto_trade_toggle_command))
        application.add_handler(CommandHandler("auto_emergency", auto_trade_emergency_command))
        application.add_handler(CommandHandler("auto_aggressive", auto_trade_aggressive_command))
        
        application.add_handler(MessageHandler(filters.Regex(r'^/approve_.*'), approve_trade_command))
        application.add_handler(MessageHandler(filters.Regex(r'^/reject_.*'), reject_trade_command))
        
        # Add callback query handlers for inline buttons
        application.add_handler(CallbackQueryHandler(approve_trade_command, pattern=r'^approve_.*'))
        application.add_handler(CallbackQueryHandler(reject_trade_command, pattern=r'^reject_.*'))
        logger.info("✅ Button handlers registered")

        # --- Automatic Market Analysis Every 5 Minutes ---
        import threading
        import time
        
        def auto_scan_loop():
            """Автоматическое сканирование каждые 5 минут"""
            time.sleep(30)  # Подождать 30 секунд после старта
            while True:
                try:
                    logger.info("🔄 Auto-scan: начинаю сканирование рынка...")
                    agent.analyze_all_markets()
                    logger.info("✅ Auto-scan: сканирование завершено")
                except Exception as e:
                    logger.error(f"❌ Auto-scan error: {e}", exc_info=True)
                
                # Следующее сканирование через 5 минут
                logger.info("⏰ Следующее сканирование через 5 минут")
                time.sleep(300)  # 5 минут = 300 секунд
        
        def position_monitor_loop():
            """Мониторинг позиций каждые 60 секунд"""
            time.sleep(60)  # Подождать 1 минуту после старта
            while True:
                try:
                    agent.check_and_close_positions()
                except Exception as e:
                    logger.error(f"❌ Position monitor error: {e}", exc_info=True)
                
                time.sleep(60)  # Проверка каждую минуту
        
        # Запуск автосканирования в фоновом потоке
        scan_thread = threading.Thread(target=auto_scan_loop, daemon=True)
        scan_thread.start()
        logger.info("✅ Автосканирование запущено! Каждые 5 минут.")
        
        # Запуск мониторинга позиций
        monitor_thread = threading.Thread(target=position_monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("✅ Мониторинг позиций запущен! Проверка каждую минуту.")
        
        # Run the bot until the user presses Ctrl-C  
        logger.info("Starting Telegram bot polling...")
        logger.info("🤖 Bot is running with auto-scan every 5 minutes")
        
        # Запуск бота (python-telegram-bot v22 сам управляет event loop)
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )
        
        logger.info("Bot polling stopped")

        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (Ctrl-C)")
    except Exception as e:
        logger.critical(f"Fatal error in main: {e}", exc_info=True)
        raise
    finally:
        logger.info("Telegram bot stopped.")

if __name__ == '__main__':
    # Setup logging with file output for 24/7 operation
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler('bot.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise
