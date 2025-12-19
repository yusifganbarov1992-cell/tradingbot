# 📚 API Documentation

## NexusTrader AI - Complete API Reference

---

## Table of Contents

1. [Telegram Bot Commands](#telegram-bot-commands)
2. [Trading Agent API](#trading-agent-api)
3. [Database API](#database-api)
4. [Autonomous Trader API](#autonomous-trader-api)
5. [Dashboard API](#dashboard-api)
6. [External APIs](#external-apis)

---

## Telegram Bot Commands

### User Commands

| Command | Description | Parameters |
|---------|-------------|------------|
| `/start` | Initialize bot, show welcome message | None |
| `/help` | Show all available commands | None |
| `/status` | Current bot status and statistics | None |
| `/balance` | Check Binance account balance | None |
| `/analyze <symbol>` | Analyze a cryptocurrency | `symbol` - e.g., BTC, ETH |
| `/positions` | View open positions | None |
| `/history` | Recent trade history | None |
| `/auto_status` | Auto-trade module status | None |

### Admin Commands

| Command | Description | Access |
|---------|-------------|--------|
| `/emergency_stop` | Stop all trading immediately | Admin only |
| `/auto_toggle` | Enable/disable auto-trading | Admin only |
| `/set_confidence <value>` | Set minimum AI confidence | Admin only |

### Example Usage

```
/analyze BTC
/analyze ETH
/balance
/status
```

---

## Trading Agent API

### Class: `TradingAgent`

Main trading agent class with AI-powered analysis.

```python
from trading_bot import TradingAgent

agent = TradingAgent()
```

#### Methods

##### `analyze_market_symbol(symbol: str) -> Dict`

Analyze a cryptocurrency and generate trading signal.

**Parameters:**
- `symbol` (str): Trading pair, e.g., "BTC/USDT"

**Returns:**
```python
{
    "symbol": "BTC/USDT",
    "signal": "BUY" | "SELL" | "HOLD",
    "ai_confidence": 7.5,  # 0-10 scale
    "price": 88350.00,
    "analysis": "Technical analysis text...",
    "indicators": {
        "rsi": 45.2,
        "macd": "bullish",
        "volume_trend": "increasing"
    }
}
```

##### `execute_trade(signal_data: Dict) -> bool`

Execute a trade based on signal data.

**Parameters:**
- `signal_data` (Dict): Signal from `analyze_market_symbol()`

**Returns:**
- `True` if trade executed successfully
- `False` if trade failed

##### `get_balance() -> Dict`

Get current account balance.

**Returns:**
```python
{
    "USDT": 1000.00,
    "BTC": 0.001,
    "ETH": 0.5
}
```

##### `get_active_positions() -> Dict`

Get all open positions.

**Returns:**
```python
{
    "BTC/USDT": {
        "entry_price": 85000.00,
        "amount": 0.001,
        "pnl_percent": 3.5
    }
}
```

---

## Database API

### Class: `SupabaseDatabase`

Cloud database interface for storing trades and signals.

```python
from database_supabase import SupabaseDatabase

db = SupabaseDatabase()
```

#### Methods

##### `save_trade(trade_data: Dict) -> str`

Save a trade to the database.

**Parameters:**
```python
trade_data = {
    "trade_id": "uuid",
    "symbol": "BTC/USDT",
    "signal": "BUY",
    "entry_price": 88000.00,
    "amount": 0.001,
    "usdt_amount": 88.00,
    "ai_confidence": 8.5,
    "status": "open"
}
```

**Returns:** Trade ID (str)

##### `get_trades(limit: int = 100) -> List[Dict]`

Retrieve trades from database.

**Parameters:**
- `limit` (int): Maximum number of trades to return

**Returns:** List of trade dictionaries

##### `update_trade(trade_id: str, updates: Dict) -> bool`

Update an existing trade.

##### `save_signal(signal_data: Dict) -> str`

Save a trading signal.

##### `get_emergency_stop() -> bool`

Check if emergency stop is activated.

##### `save_emergency_stop(status: bool) -> None`

Set emergency stop status.

---

## Autonomous Trader API

### Class: `AutonomousTrader`

Handles automatic trade execution without human confirmation.

```python
from modules.autonomous_trader import AutonomousTrader

auto_trader = AutonomousTrader(
    auto_trade_enabled=True,
    min_confidence=7.0,
    max_trades_per_hour=3,
    max_concurrent_positions=5
)
```

#### Methods

##### `should_execute_auto(signal_data, active_positions, balance) -> Tuple[bool, str]`

Determine if a trade should be executed automatically.

**Parameters:**
- `signal_data` (Dict): Trading signal
- `active_positions` (Dict): Current open positions
- `balance` (float): Available balance

**Returns:**
- `(True, "reason")` - Execute automatically
- `(False, "reason")` - Don't execute, reason why

##### `record_trade() -> None`

Record that a trade was executed (for hourly limit tracking).

##### `emergency_stop(reason: str) -> None`

Activate emergency stop.

##### `resume_trading() -> None`

Resume trading after emergency stop.

##### `get_status() -> Dict`

Get current autonomous trader status.

**Returns:**
```python
{
    "enabled": True,
    "emergency_paused": False,
    "trades_this_hour": 1,
    "max_trades_per_hour": 3,
    "min_confidence": 7.0
}
```

---

## Dashboard API

### Data Provider

```python
from dashboard.data_provider import get_data_provider

provider = get_data_provider()
```

#### Methods

##### `get_portfolio_summary() -> Dict`

```python
{
    "balance": 10000.00,
    "total_trades": 15,
    "win_rate": 65.5,
    "total_pnl": 250.00
}
```

##### `get_recent_trades(limit: int = 50) -> List[Dict]`

##### `get_signals(limit: int = 50) -> List[Dict]`

##### `get_performance_metrics() -> Dict`

```python
{
    "sharpe_ratio": 1.5,
    "max_drawdown": -5.2,
    "profit_factor": 1.8,
    "avg_trade_duration": "4h 30m"
}
```

---

## External APIs

### Binance API (via ccxt)

```python
import ccxt

exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET_KEY'),
    'enableRateLimit': True
})

# Get ticker
ticker = exchange.fetch_ticker('BTC/USDT')

# Get balance
balance = exchange.fetch_balance()

# Create order
order = exchange.create_market_buy_order('BTC/USDT', 0.001)
```

### OpenAI API

```python
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a trading analyst."},
        {"role": "user", "content": "Analyze BTC/USDT"}
    ],
    temperature=0.7,
    max_tokens=1000
)
```

### Telegram API

```python
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler

application = Application.builder().token(BOT_TOKEN).build()
application.add_handler(CommandHandler("start", start_command))
```

---

## Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| `E001` | Insufficient balance | Add funds to account |
| `E002` | API rate limit | Wait and retry |
| `E003` | Invalid symbol | Check symbol format |
| `E004` | Emergency stop active | Resume trading first |
| `E005` | Max positions reached | Close positions first |

---

## Rate Limits

| Service | Limit | Window |
|---------|-------|--------|
| Binance | 1200 requests | 1 minute |
| OpenAI | Based on tier | Per minute |
| Telegram | 30 messages | 1 second |
| Internal trades | 3 trades | 1 hour |

---

## Webhooks (Future)

```
POST /api/webhook/signal
POST /api/webhook/alert
GET  /api/health
GET  /api/status
```

---

*Last updated: December 2024*
