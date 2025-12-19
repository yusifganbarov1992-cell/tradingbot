"""
Автономный торговый модуль - AUTO_TRADE режим
Позволяет боту торговать самостоятельно без ручного подтверждения

Функционал:
- Автоматическое выполнение сделок
- Умная логика с градацией уверенности
- Emergency controls через Telegram
- Whitelist/blacklist монет
- Hourly limits (защита от overtrading)
- Risk-based auto-approval

Источник: Адаптировано из freqtrade auto-trading logic
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import asyncio

logger = logging.getLogger(__name__)


class AutonomousTrader:
    """
    Автономный трейдер - принимает решения БЕЗ человека
    """
    
    def __init__(
        self,
        auto_trade_enabled: bool = False,
        min_confidence: float = 7.0,
        max_trades_per_hour: int = 3,
        max_concurrent_positions: int = 5,
        emergency_stop_loss_pct: float = 0.05,
        whitelist: list = None,
        blacklist: list = None
    ):
        """
        Args:
            auto_trade_enabled: Включен ли автономный режим
            min_confidence: Минимальная уверенность AI (0-10)
            max_trades_per_hour: Макс. сделок в час
            max_concurrent_positions: Макс. открытых позиций
            emergency_stop_loss_pct: Emergency stop на % просадке
            whitelist: Список разрешенных монет (None = все)
            blacklist: Список запрещенных монет
        """
        self.enabled = auto_trade_enabled
        self.min_confidence = min_confidence
        self.max_trades_per_hour = max_trades_per_hour
        self.max_concurrent_positions = max_concurrent_positions
        self.emergency_stop_loss_pct = emergency_stop_loss_pct
        
        # Whitelist/Blacklist
        self.whitelist = whitelist or []  # Пустой = все разрешены
        self.blacklist = blacklist or ['LUNA', 'FTT']  # Скам-коины
        
        # Tracking
        self.trades_this_hour = []  # Timestamps выполненных сделок
        self.last_trade_time = None
        self.aggressive_mode = False  # Можно переключать динамически
        
        # Emergency
        self.emergency_paused = False
        
        logger.info(f"🤖 AutonomousTrader initialized:")
        logger.info(f"   - Enabled: {self.enabled}")
        logger.info(f"   - Min confidence: {self.min_confidence}/10")
        logger.info(f"   - Max trades/hour: {self.max_trades_per_hour}")
        logger.info(f"   - Whitelist: {len(self.whitelist)} symbols" if self.whitelist else "   - Whitelist: ALL")
        logger.info(f"   - Blacklist: {self.blacklist}")
    
    def should_execute_auto(
        self,
        signal_data: Dict,
        active_positions: Dict,
        balance: float
    ) -> Tuple[bool, str]:
        """
        Главная логика: выполнять ли сделку автоматически?
        
        Args:
            signal_data: Данные сигнала (symbol, confidence, price, etc.)
            active_positions: Открытые позиции
            balance: Текущий баланс
            
        Returns:
            (should_execute: bool, reason: str)
        """
        # 1. Проверка - включен ли AUTO_TRADE?
        if not self.enabled:
            return False, "AUTO_TRADE disabled"
        
        # 2. Emergency pause
        if self.emergency_paused:
            return False, "Emergency pause activated"
        
        symbol = signal_data.get('symbol')
        confidence = signal_data.get('ai_confidence', 0)
        usdt_amount = signal_data.get('usdt_amount', 0)
        
        # 3. Проверка whitelist
        if self.whitelist and symbol not in self.whitelist:
            return False, f"{symbol} not in whitelist"
        
        # 4. Проверка blacklist
        if symbol in self.blacklist:
            return False, f"{symbol} in blacklist"
        
        # 5. Проверка confidence
        if confidence < self.min_confidence:
            return False, f"Confidence {confidence} < {self.min_confidence}"
        
        # 6. Проверка hourly limit
        if not self._check_hourly_limit():
            return False, f"Hourly limit reached ({self.max_trades_per_hour})"
        
        # 7. Проверка макс. позиций
        if len(active_positions) >= self.max_concurrent_positions:
            return False, f"Max positions reached ({self.max_concurrent_positions})"
        
        # 8. Проверка дубликата
        if symbol in active_positions:
            return False, f"Position already open for {symbol}"
        
        # 9. Проверка баланса
        if balance < usdt_amount * 1.1:  # +10% запас
            return False, f"Insufficient balance: ${balance:.2f} < ${usdt_amount * 1.1:.2f}"
        
        # 10. Умная логика на основе уверенности
        if confidence >= 9.0:
            # Очень высокая уверенность - выполняем
            reason = f"HIGH confidence {confidence}/10 - AUTO EXECUTE"
        elif confidence >= 8.0 and self.aggressive_mode:
            # Высокая уверенность + aggressive mode
            reason = f"GOOD confidence {confidence}/10 + aggressive mode - AUTO EXECUTE"
        elif confidence >= self.min_confidence:
            # Средняя уверенность - выполняем если не превышен hourly limit
            trades_count = len([t for t in self.trades_this_hour if t > datetime.now() - timedelta(hours=1)])
            if trades_count < self.max_trades_per_hour - 1:  # Оставляем 1 слот для высоких confidence
                reason = f"NORMAL confidence {confidence}/10 - AUTO EXECUTE"
            else:
                return False, f"Saving hourly limit for higher confidence signals"
        else:
            return False, f"Confidence {confidence}/10 not sufficient"
        
        # ✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!
        logger.info(f"🤖 AUTO EXECUTE APPROVED: {reason}")
        return True, reason
    
    def _check_hourly_limit(self) -> bool:
        """Проверка лимита сделок в час"""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Очистка старых
        self.trades_this_hour = [t for t in self.trades_this_hour if t > hour_ago]
        
        # Проверка лимита
        return len(self.trades_this_hour) < self.max_trades_per_hour
    
    def record_trade(self):
        """Записать выполненную сделку"""
        self.trades_this_hour.append(datetime.now())
        self.last_trade_time = datetime.now()
        logger.info(f"🤖 Trade recorded. Total this hour: {len(self.trades_this_hour)}/{self.max_trades_per_hour}")
    
    def set_aggressive(self, aggressive: bool):
        """
        Переключить агрессивность
        
        Args:
            aggressive: True = более рискованная торговля
        """
        self.aggressive_mode = aggressive
        logger.info(f"🤖 Aggressive mode: {'ON' if aggressive else 'OFF'}")
    
    def emergency_stop(self, reason: str = "Manual"):
        """
        EMERGENCY STOP - остановка всей торговли
        
        Args:
            reason: Причина остановки
        """
        self.emergency_paused = True
        logger.critical(f"🚨 EMERGENCY STOP ACTIVATED: {reason}")
    
    def resume_trading(self):
        """Возобновить торговлю после emergency stop"""
        self.emergency_paused = False
        logger.info("✅ Trading resumed after emergency stop")
    
    def get_status(self) -> Dict:
        """Получить текущий статус"""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        trades_this_hour_count = len([t for t in self.trades_this_hour if t > hour_ago])
        
        return {
            'enabled': self.enabled,
            'emergency_paused': self.emergency_paused,
            'aggressive_mode': self.aggressive_mode,
            'trades_this_hour': trades_this_hour_count,
            'max_trades_per_hour': self.max_trades_per_hour,
            'last_trade': self.last_trade_time.isoformat() if self.last_trade_time else None,
            'min_confidence': self.min_confidence,
            'whitelist_count': len(self.whitelist),
            'blacklist_count': len(self.blacklist)
        }
    
    def adjust_confidence_threshold(self, new_threshold: float):
        """
        Динамически изменить порог уверенности
        
        Args:
            new_threshold: Новый порог (0-10)
        """
        old = self.min_confidence
        self.min_confidence = max(5.0, min(9.0, new_threshold))  # Clamp 5-9
        logger.info(f"🤖 Confidence threshold adjusted: {old} → {self.min_confidence}")
    
    def add_to_whitelist(self, symbol: str):
        """Добавить монету в whitelist"""
        if symbol not in self.whitelist:
            self.whitelist.append(symbol)
            logger.info(f"✅ Added {symbol} to whitelist")
    
    def add_to_blacklist(self, symbol: str):
        """Добавить монету в blacklist"""
        if symbol not in self.blacklist:
            self.blacklist.append(symbol)
            logger.info(f"🚫 Added {symbol} to blacklist")
    
    def remove_from_blacklist(self, symbol: str):
        """Убрать монету из blacklist"""
        if symbol in self.blacklist:
            self.blacklist.remove(symbol)
            logger.info(f"✅ Removed {symbol} from blacklist")
    
    async def send_auto_trade_notification(
        self,
        bot_token: str,
        chat_id: str,
        signal_data: Dict,
        reason: str
    ):
        """
        Отправить уведомление о выполненной автоматической сделке
        
        Args:
            bot_token: Telegram bot token
            chat_id: Chat ID
            signal_data: Данные сигнала
            reason: Причина выполнения
        """
        from telegram import Bot
        
        bot = Bot(token=bot_token)
        
        message = f"""
🤖 **АВТОМАТИЧЕСКАЯ СДЕЛКА ВЫПОЛНЕНА**

📊 Монета: {signal_data.get('symbol')}
💰 Цена: ${signal_data.get('price', 0):.2f}
📦 Размер: {signal_data.get('crypto_amount', 0):.6f} (~${signal_data.get('usdt_amount', 0):.2f})
📈 Сигнал: {signal_data.get('signal')}
🤖 AI Уверенность: {signal_data.get('ai_confidence', 0)}/10

✅ Причина: {reason}

⏰ Время: {datetime.now().strftime('%H:%M:%S')}
"""
        
        try:
            await bot.send_message(chat_id=chat_id, text=message)
            logger.info(f"📤 Auto-trade notification sent for {signal_data.get('symbol')}")
        except Exception as e:
            logger.error(f"Failed to send auto-trade notification: {e}")


# ========================================
# INTEGRATION EXAMPLE для trading_bot.py
# ========================================

"""
# В __init__ TradingAgent:
from modules.autonomous_trader import AutonomousTrader

self.autonomous = AutonomousTrader(
    auto_trade_enabled=os.getenv('AUTO_TRADE', 'false').lower() == 'true',
    min_confidence=7.0,
    max_trades_per_hour=3,
    max_concurrent_positions=5,
    whitelist=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],  # Только эти
    blacklist=['LUNA', 'FTT']  # Никогда эти
)

# В analyze_market_symbol(), после получения signal_data:
if signal_data:
    # Проверка - выполнять автоматически или спрашивать?
    should_auto, reason = self.autonomous.should_execute_auto(
        signal_data=signal_data,
        active_positions=self.active_positions,
        balance=balance
    )
    
    if should_auto:
        # АВТОМАТИЧЕСКОЕ ВЫПОЛНЕНИЕ
        logger.info(f"🤖 AUTO TRADE: {reason}")
        
        # Выполнить сделку БЕЗ подтверждения
        success = self.execute_trade_directly(signal_data)
        
        if success:
            # Записать в tracker
            self.autonomous.record_trade()
            
            # Отправить уведомление (не ждем подтверждения!)
            asyncio.run(self.autonomous.send_auto_trade_notification(
                bot_token=self.telegram_bot_token,
                chat_id=self.operator_chat_id,
                signal_data=signal_data,
                reason=reason
            ))
    else:
        # РУЧНОЕ ПОДТВЕРЖДЕНИЕ (как сейчас)
        logger.info(f"⏸️  Manual confirmation required: {reason}")
        self.send_signal_to_telegram(signal_data)

# Telegram команды для управления:
async def auto_trade_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agent = context.bot_data['agent']
    status = agent.autonomous.get_status()
    
    message = f'''
🤖 **AUTO TRADE STATUS**

{'✅ ENABLED' if status['enabled'] else '❌ DISABLED'}
{'🚨 EMERGENCY PAUSED' if status['emergency_paused'] else ''}
{'⚡ AGGRESSIVE MODE' if status['aggressive_mode'] else '🛡️ CONSERVATIVE MODE'}

📊 Trades this hour: {status['trades_this_hour']}/{status['max_trades_per_hour']}
🎯 Min confidence: {status['min_confidence']}/10
📝 Whitelist: {status['whitelist_count']} symbols
🚫 Blacklist: {status['blacklist_count']} symbols
⏰ Last trade: {status['last_trade'] or 'Never'}
'''
    await update.message.reply_text(message)

async def auto_trade_toggle_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agent = context.bot_data['agent']
    agent.autonomous.enabled = not agent.autonomous.enabled
    
    status = '✅ ENABLED' if agent.autonomous.enabled else '❌ DISABLED'
    await update.message.reply_text(f"🤖 AUTO TRADE: {status}")

async def auto_trade_emergency_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agent = context.bot_data['agent']
    agent.autonomous.emergency_stop(reason="Manual telegram command")
    await update.message.reply_text("🚨 EMERGENCY STOP ACTIVATED!")

# Регистрация команд:
application.add_handler(CommandHandler("auto_status", auto_trade_status_command))
application.add_handler(CommandHandler("auto_toggle", auto_trade_toggle_command))
application.add_handler(CommandHandler("auto_emergency", auto_trade_emergency_command))
"""
