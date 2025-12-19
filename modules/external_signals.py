"""
External Signal Integration Module
Monitors external signal sources and validates them with AI

Supported sources:
1. Telegram Channels (via Telethon)
2. TradingView Webhooks
3. Custom API endpoints

Usage:
    Add to .env:
    SIGNAL_CHANNELS=@crypto_signals,@binance_killers
    VALIDATE_EXTERNAL_SIGNALS=true
    MIN_EXTERNAL_CONFIDENCE=6.0
"""

import os
import re
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExternalSignal:
    """External signal data structure"""
    source: str  # telegram, tradingview, api
    channel: str  # channel name or webhook id
    symbol: str  # Trading pair (BTC/USDT)
    signal_type: str  # BUY, SELL, LONG, SHORT
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    confidence: float  # 0-10
    raw_message: str
    timestamp: datetime
    validated: bool = False
    validation_reason: str = ""


class SignalParser:
    """Parse signals from various formats"""
    
    # Common patterns in signal messages
    PATTERNS = {
        'symbol': [
            r'#?([A-Z]{2,10})[/\-]?(USDT|BTC|ETH|BUSD)',
            r'(BTC|ETH|BNB|SOL|XRP|ADA|DOGE|DOT|MATIC|LINK|AVAX|UNI|ATOM|LTC)',
        ],
        'signal': [
            r'(BUY|SELL|LONG|SHORT)',
            r'(покупка|продажа|лонг|шорт)',
            r'(📈|📉|🟢|🔴)',
        ],
        'entry': [
            r'entry[:\s]*\$?(\d+\.?\d*)',
            r'вход[:\s]*\$?(\d+\.?\d*)',
            r'price[:\s]*\$?(\d+\.?\d*)',
        ],
        'stop_loss': [
            r'(?:sl|stop|stop[- ]?loss)[:\s]*\$?(\d+\.?\d*)',
            r'стоп[:\s]*\$?(\d+\.?\d*)',
        ],
        'take_profit': [
            r'(?:tp|target|take[- ]?profit)[:\s]*\$?(\d+\.?\d*)',
            r'(?:tp1|target1)[:\s]*\$?(\d+\.?\d*)',
            r'цель[:\s]*\$?(\d+\.?\d*)',
        ],
    }
    
    @classmethod
    def parse_message(cls, message: str, source: str = "unknown") -> Optional[ExternalSignal]:
        """Parse a signal message into structured format"""
        message_lower = message.lower()
        
        # Extract symbol
        symbol = None
        for pattern in cls.PATTERNS['symbol']:
            match = re.search(pattern, message.upper())
            if match:
                if match.lastindex == 2:
                    symbol = f"{match.group(1)}/{match.group(2)}"
                else:
                    symbol = f"{match.group(1)}/USDT"
                break
        
        if not symbol:
            return None
        
        # Extract signal type
        signal_type = None
        for pattern in cls.PATTERNS['signal']:
            match = re.search(pattern, message.upper())
            if match:
                sig = match.group(1).upper()
                if sig in ['BUY', 'LONG', '📈', '🟢', 'ПОКУПКА', 'ЛОНГ']:
                    signal_type = 'BUY'
                elif sig in ['SELL', 'SHORT', '📉', '🔴', 'ПРОДАЖА', 'ШОРТ']:
                    signal_type = 'SELL'
                break
        
        if not signal_type:
            return None
        
        # Extract prices
        entry_price = cls._extract_price(message, cls.PATTERNS['entry'])
        stop_loss = cls._extract_price(message, cls.PATTERNS['stop_loss'])
        take_profit = cls._extract_price(message, cls.PATTERNS['take_profit'])
        
        return ExternalSignal(
            source=source,
            channel="",
            symbol=symbol,
            signal_type=signal_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=5.0,  # Default, will be validated by AI
            raw_message=message[:500],
            timestamp=datetime.now()
        )
    
    @staticmethod
    def _extract_price(text: str, patterns: List[str]) -> Optional[float]:
        """Extract price from text using patterns"""
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass
        return None


class ExternalSignalMonitor:
    """
    Monitor external signal sources and integrate with bot
    """
    
    def __init__(
        self,
        validate_signals: bool = True,
        min_confidence: float = 6.0,
        channels: List[str] = None
    ):
        self.validate_signals = validate_signals
        self.min_confidence = min_confidence
        self.channels = channels or []
        self.parser = SignalParser()
        self.signals_queue: List[ExternalSignal] = []
        self.processed_signals: Dict[str, ExternalSignal] = {}
        
        logger.info(f"📡 ExternalSignalMonitor initialized")
        logger.info(f"   - Channels: {len(self.channels)}")
        logger.info(f"   - Validation: {validate_signals}")
        logger.info(f"   - Min confidence: {min_confidence}")
    
    def add_signal(self, message: str, source: str = "telegram", channel: str = "") -> Optional[ExternalSignal]:
        """
        Add a signal from external source
        
        Args:
            message: Raw signal message
            source: Signal source (telegram, tradingview, api)
            channel: Channel/source identifier
            
        Returns:
            Parsed signal or None
        """
        signal = self.parser.parse_message(message, source)
        
        if signal:
            signal.channel = channel
            self.signals_queue.append(signal)
            logger.info(f"📡 External signal received: {signal.signal_type} {signal.symbol} from {channel}")
            return signal
        
        return None
    
    async def validate_signal(self, signal: ExternalSignal, market_analyzer) -> Tuple[bool, str, float]:
        """
        Validate external signal with our AI
        
        Args:
            signal: External signal to validate
            market_analyzer: AI analysis function
            
        Returns:
            (is_valid, reason, confidence)
        """
        try:
            # Get our AI analysis for this symbol
            ai_result = await market_analyzer(signal.symbol)
            
            if not ai_result:
                return False, "AI analysis failed", 0.0
            
            our_signal = ai_result.get('signal', 'HOLD')
            our_confidence = ai_result.get('confidence', 0)
            our_reason = ai_result.get('reason', 'No reason')
            
            # Compare signals
            signals_match = (
                (signal.signal_type == 'BUY' and our_signal in ['BUY', 'STRONG_BUY']) or
                (signal.signal_type == 'SELL' and our_signal in ['SELL', 'STRONG_SELL'])
            )
            
            if signals_match and our_confidence >= self.min_confidence:
                reason = f"✅ AI confirms: {our_reason}"
                return True, reason, our_confidence
            elif signals_match:
                reason = f"⚠️ AI agrees but low confidence ({our_confidence}/10): {our_reason}"
                return False, reason, our_confidence
            else:
                reason = f"❌ AI disagrees ({our_signal} vs {signal.signal_type}): {our_reason}"
                return False, reason, our_confidence
                
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False, f"Error: {str(e)}", 0.0
    
    def get_pending_signals(self) -> List[ExternalSignal]:
        """Get all pending (unvalidated) signals"""
        return [s for s in self.signals_queue if not s.validated]
    
    def get_validated_signals(self, min_confidence: float = None) -> List[ExternalSignal]:
        """Get validated signals above confidence threshold"""
        threshold = min_confidence or self.min_confidence
        return [s for s in self.signals_queue 
                if s.validated and s.confidence >= threshold]
    
    def clear_old_signals(self, hours: int = 24):
        """Remove signals older than specified hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        self.signals_queue = [s for s in self.signals_queue 
                             if s.timestamp > cutoff]


class TradingViewWebhook:
    """
    Handle TradingView webhook signals
    
    TradingView Alert format:
    {
        "symbol": "BTCUSDT",
        "action": "buy",
        "price": 87000,
        "sl": 85000,
        "tp": 92000,
        "secret": "your_secret"
    }
    """
    
    def __init__(self, secret: str):
        self.secret = secret
    
    def parse_webhook(self, data: dict) -> Optional[ExternalSignal]:
        """Parse TradingView webhook data"""
        # Verify secret
        if data.get('secret') != self.secret:
            logger.warning("Invalid TradingView webhook secret")
            return None
        
        symbol = data.get('symbol', '')
        if not '/' in symbol:
            symbol = f"{symbol[:len(symbol)-4]}/{symbol[-4:]}"  # BTCUSDT -> BTC/USDT
        
        action = data.get('action', '').upper()
        if action not in ['BUY', 'SELL', 'LONG', 'SHORT']:
            return None
        
        signal_type = 'BUY' if action in ['BUY', 'LONG'] else 'SELL'
        
        return ExternalSignal(
            source='tradingview',
            channel='webhook',
            symbol=symbol,
            signal_type=signal_type,
            entry_price=data.get('price'),
            stop_loss=data.get('sl'),
            take_profit=data.get('tp'),
            confidence=7.0,  # TradingView signals get higher default confidence
            raw_message=str(data),
            timestamp=datetime.now()
        )


# Example usage
if __name__ == "__main__":
    # Test signal parsing
    test_messages = [
        "🚀 #BTC/USDT BUY Signal\nEntry: $87000\nSL: $85000\nTP: $92000",
        "📈 ETH LONG\nВход: 2900\nСтоп: 2800\nЦель: 3200",
        "🔴 SELL SOL/USDT @ 180\nStop Loss: 190\nTake Profit: 160",
    ]
    
    parser = SignalParser()
    
    for msg in test_messages:
        signal = parser.parse_message(msg, "test")
        if signal:
            print(f"Parsed: {signal.signal_type} {signal.symbol}")
            print(f"  Entry: {signal.entry_price}, SL: {signal.stop_loss}, TP: {signal.take_profit}")
        else:
            print(f"Failed to parse: {msg[:50]}...")
