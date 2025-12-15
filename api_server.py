"""
FastAPI server for Frontend-Backend communication
Runs alongside the Telegram bot to provide REST API for the web dashboard
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from typing import List, Dict, Optional
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="NexusTrader API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared state (in production, use Redis or database)
bot_state = {
    "status": "running",
    "current_symbol": "BTC/USDT",
    "signals": [],
    "pending_trades": {},
    "market_data": {}
}

class TradeApproval(BaseModel):
    trade_id: str
    approved: bool

class BotConfig(BaseModel):
    symbol: Optional[str] = None
    risk_level: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "NexusTrader API", "status": "running"}

@app.get("/api/status")
async def get_bot_status():
    """Get current bot status"""
    return {
        "status": bot_state["status"],
        "symbol": bot_state["current_symbol"],
        "active_trades": len(bot_state["pending_trades"]),
        "total_signals": len(bot_state["signals"])
    }

@app.get("/api/signals")
async def get_signals():
    """Get recent trading signals"""
    return {"signals": bot_state["signals"][-50:]}  # Last 50 signals

@app.get("/api/pending-trades")
async def get_pending_trades():
    """Get trades awaiting approval"""
    return {"trades": list(bot_state["pending_trades"].values())}

@app.post("/api/approve-trade")
async def approve_trade(approval: TradeApproval):
    """Approve or reject a pending trade"""
    if approval.trade_id not in bot_state["pending_trades"]:
        raise HTTPException(status_code=404, detail="Trade not found")
    
    trade = bot_state["pending_trades"][approval.trade_id]
    
    if approval.approved:
        # In production, this would trigger actual order placement
        trade["status"] = "approved"
        bot_state["signals"].append({
            "type": trade["side"],
            "symbol": trade["symbol"],
            "price": trade["price"],
            "status": "executed",
            "timestamp": trade["timestamp"]
        })
    else:
        trade["status"] = "rejected"
    
    del bot_state["pending_trades"][approval.trade_id]
    
    return {"message": f"Trade {approval.trade_id} {'approved' if approval.approved else 'rejected'}"}

@app.post("/api/config")
async def update_config(config: BotConfig):
    """Update bot configuration"""
    if config.symbol:
        bot_state["current_symbol"] = config.symbol
    if config.risk_level:
        bot_state["risk_level"] = config.risk_level
    
    return {"message": "Configuration updated", "config": config}

@app.get("/api/market-data/{symbol}")
async def get_market_data(symbol: str):
    """Get current market data for symbol"""
    # In production, fetch from exchange
    return {
        "symbol": symbol,
        "price": bot_state["market_data"].get(symbol, {}).get("price", 0),
        "change_24h": bot_state["market_data"].get(symbol, {}).get("change", 0),
        "volume": bot_state["market_data"].get(symbol, {}).get("volume", 0)
    }

# Helper function to update state from trading bot
def update_bot_state(key: str, value):
    """Update shared state (called from trading_bot.py)"""
    bot_state[key] = value

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
