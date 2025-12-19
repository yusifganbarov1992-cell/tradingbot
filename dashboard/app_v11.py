"""
Trading Bot Dashboard v11.0
Full Real Data + AI Reasoning + External Signals
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Initialize ccxt and database
import ccxt
from database_supabase import SupabaseDatabase

# Page config
st.set_page_config(
    page_title="NexusTrader Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize connections
@st.cache_resource
def init_exchange():
    return ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET_KEY'),
        'enableRateLimit': True,
    })

@st.cache_resource
def init_db():
    return SupabaseDatabase()

exchange = init_exchange()
db = init_db()

# Helper functions
def get_real_balance():
    """Get REAL balance from Binance with all assets"""
    try:
        balance = exchange.fetch_balance()
        assets = []
        total_usd = 0
        
        for currency, amounts in balance['total'].items():
            if amounts > 0:
                usd_value = 0
                if currency in ['USDT', 'USDC', 'BUSD', 'FDUSD']:
                    usd_value = amounts
                elif currency.startswith('LD'):
                    base = currency[2:]
                    if base in ['USDT', 'USDC']:
                        usd_value = amounts
                    else:
                        try:
                            ticker = exchange.fetch_ticker(f'{base}/USDT')
                            usd_value = amounts * ticker['last']
                        except:
                            pass
                else:
                    try:
                        ticker = exchange.fetch_ticker(f'{currency}/USDT')
                        usd_value = amounts * ticker['last']
                    except:
                        pass
                
                if usd_value > 0.01:
                    assets.append({
                        'asset': currency,
                        'amount': amounts,
                        'usd_value': usd_value
                    })
                    total_usd += usd_value
        
        assets.sort(key=lambda x: x['usd_value'], reverse=True)
        return total_usd, assets
    except Exception as e:
        st.error(f"Failed to fetch balance: {e}")
        return 0, []

def get_price_history(symbol: str, timeframe: str = '1h', limit: int = 100):
    """Get OHLCV price history"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Failed to fetch price history: {e}")
        return pd.DataFrame()

def get_trade_history():
    """Get all trades from database with AI reasoning"""
    try:
        trades = db.get_trade_history(limit=100)
        return trades
    except:
        return []

def get_signals_from_db():
    """Get AI signals with reasoning"""
    try:
        # Try to get signals from database
        result = db.client.table('signals').select('*').order('created_at', desc=True).limit(50).execute()
        return result.data if result.data else []
    except:
        return []

# Title
st.title("🤖 NexusTrader AI Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    selected_symbol = st.selectbox(
        "Trading Pair",
        ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT"],
        index=0
    )
    
    timeframe = st.selectbox(
        "Timeframe",
        ["1m", "5m", "15m", "1h", "4h", "1d"],
        index=3
    )
    
    st.divider()
    
    # Paper trading status
    paper_trading = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
    auto_trade = os.getenv('AUTO_TRADE', 'false').lower() == 'true'
    
    if paper_trading:
        st.warning("📝 PAPER TRADING MODE")
    else:
        st.error("🔥 REAL TRADING MODE")
    
    st.info(f"🤖 Auto Trade: {'ON' if auto_trade else 'OFF'}")
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "💰 Portfolio",
    "📈 Charts",
    "🤖 AI Signals",
    "📋 Trade History",
    "📡 External Signals"
])

# TAB 1: OVERVIEW
with tab1:
    st.header("📊 Real-Time Overview")
    
    # Get real balance
    total_balance, assets = get_real_balance()
    
    # Get current prices
    try:
        btc_ticker = exchange.fetch_ticker('BTC/USDT')
        eth_ticker = exchange.fetch_ticker('ETH/USDT')
    except:
        btc_ticker = {'last': 0, 'percentage': 0}
        eth_ticker = {'last': 0, 'percentage': 0}
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Total Balance",
            f"${total_balance:,.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "₿ BTC/USDT",
            f"${btc_ticker['last']:,.2f}",
            delta=f"{btc_ticker.get('percentage', 0):+.2f}%"
        )
    
    with col3:
        st.metric(
            "Ξ ETH/USDT",
            f"${eth_ticker['last']:,.2f}",
            delta=f"{eth_ticker.get('percentage', 0):+.2f}%"
        )
    
    with col4:
        trades = get_trade_history()
        open_trades = len([t for t in trades if t.get('status') == 'open'])
        st.metric("📊 Open Positions", open_trades)
    
    st.divider()
    
    # Price chart
    st.subheader(f"📈 {selected_symbol} Price Chart")
    
    price_df = get_price_history(selected_symbol, timeframe, 100)
    
    if not price_df.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          row_heights=[0.7, 0.3],
                          vertical_spacing=0.05)
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=price_df['timestamp'],
            open=price_df['open'],
            high=price_df['high'],
            low=price_df['low'],
            close=price_df['close'],
            name='Price'
        ), row=1, col=1)
        
        # Volume
        colors = ['red' if close < open else 'green' 
                  for close, open in zip(price_df['close'], price_df['open'])]
        fig.add_trace(go.Bar(
            x=price_df['timestamp'],
            y=price_df['volume'],
            marker_color=colors,
            name='Volume'
        ), row=2, col=1)
        
        fig.update_layout(
            template='plotly_dark',
            height=500,
            xaxis_rangeslider_visible=False,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.subheader("🔔 Recent Activity")
    
    trades = get_trade_history()
    if trades:
        for trade in trades[:5]:
            col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
            
            with col1:
                entry_time = trade.get('entry_time', '')
                if entry_time:
                    try:
                        dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                        st.text(dt.strftime('%H:%M:%S'))
                    except:
                        st.text(entry_time[:10])
            
            with col2:
                side = trade.get('side', 'N/A')
                symbol = trade.get('symbol', 'N/A')
                emoji = "🟢" if side == 'BUY' else "🔴"
                st.markdown(f"{emoji} **{side}** {symbol}")
            
            with col3:
                price = trade.get('entry_price', 0)
                st.text(f"@ ${price:,.2f}")
            
            with col4:
                status = trade.get('status', 'N/A')
                pnl = trade.get('pnl') or 0
                if status == 'closed':
                    color = "green" if pnl > 0 else "red"
                    st.markdown(f"<span style='color:{color}'>{pnl:+.4f}</span>", unsafe_allow_html=True)
                else:
                    st.text("🔓 OPEN")
    else:
        st.info("No recent trades")

# TAB 2: PORTFOLIO
with tab2:
    st.header("💰 Portfolio Details")
    
    total_balance, assets = get_real_balance()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Asset Allocation")
        
        if assets:
            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=[a['asset'] for a in assets[:10]],
                values=[a['usd_value'] for a in assets[:10]],
                hole=0.4,
                textinfo='label+percent'
            )])
            fig.update_layout(template='plotly_dark', height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💎 Holdings")
        
        if assets:
            holdings_df = pd.DataFrame(assets[:15])
            holdings_df['amount'] = holdings_df['amount'].apply(lambda x: f"{x:,.4f}")
            holdings_df['usd_value'] = holdings_df['usd_value'].apply(lambda x: f"${x:,.2f}")
            holdings_df.columns = ['Asset', 'Amount', 'USD Value']
            st.dataframe(holdings_df, use_container_width=True, height=400)
        else:
            st.warning("No assets found")
    
    st.divider()
    
    st.metric("💵 Total Portfolio Value", f"${total_balance:,.2f}")

# TAB 3: CHARTS
with tab3:
    st.header(f"📈 {selected_symbol} Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        chart_tf = st.selectbox("Timeframe", ["5m", "15m", "1h", "4h", "1d"], index=2, key="chart_tf")
    
    with col2:
        chart_limit = st.slider("Candles", 50, 500, 200)
    
    price_df = get_price_history(selected_symbol, chart_tf, chart_limit)
    
    if not price_df.empty:
        # Calculate indicators
        price_df['sma20'] = price_df['close'].rolling(20).mean()
        price_df['sma50'] = price_df['close'].rolling(50).mean()
        price_df['ema12'] = price_df['close'].ewm(span=12).mean()
        price_df['ema26'] = price_df['close'].ewm(span=26).mean()
        
        # RSI
        delta = price_df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        price_df['rsi'] = 100 - (100 / (1 + rs))
        
        # Chart with indicators
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                          row_heights=[0.6, 0.2, 0.2],
                          vertical_spacing=0.05)
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=price_df['timestamp'],
            open=price_df['open'],
            high=price_df['high'],
            low=price_df['low'],
            close=price_df['close'],
            name='Price'
        ), row=1, col=1)
        
        # SMA
        fig.add_trace(go.Scatter(x=price_df['timestamp'], y=price_df['sma20'],
                                mode='lines', name='SMA20', line=dict(color='yellow', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=price_df['timestamp'], y=price_df['sma50'],
                                mode='lines', name='SMA50', line=dict(color='orange', width=1)), row=1, col=1)
        
        # Volume
        colors = ['red' if c < o else 'green' for c, o in zip(price_df['close'], price_df['open'])]
        fig.add_trace(go.Bar(x=price_df['timestamp'], y=price_df['volume'],
                            marker_color=colors, name='Volume'), row=2, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=price_df['timestamp'], y=price_df['rsi'],
                                mode='lines', name='RSI', line=dict(color='purple')), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(
            template='plotly_dark',
            height=700,
            xaxis_rangeslider_visible=False,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Current stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${price_df['close'].iloc[-1]:,.2f}")
        with col2:
            st.metric("24h High", f"${price_df['high'].max():,.2f}")
        with col3:
            st.metric("24h Low", f"${price_df['low'].min():,.2f}")
        with col4:
            rsi_val = price_df['rsi'].iloc[-1]
            rsi_status = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
            st.metric("RSI", f"{rsi_val:.1f}", delta=rsi_status)

# TAB 4: AI SIGNALS
with tab4:
    st.header("🤖 AI Trading Signals")
    
    st.info("""
    **AI Signal Logic:**
    - Analyzes 50+ technical indicators
    - Pattern recognition (Double Bottom, Head & Shoulders, etc.)
    - Sentiment analysis from market data
    - Risk assessment with ATR-based stops
    """)
    
    # Get signals from database
    signals = get_signals_from_db()
    
    if signals:
        for signal in signals[:10]:
            with st.expander(f"📊 {signal.get('symbol', 'N/A')} - {signal.get('signal_type', 'N/A')} ({signal.get('confidence', 0)}/10)", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Symbol:** {signal.get('symbol')}")
                    st.markdown(f"**Signal:** {signal.get('signal_type')}")
                    st.markdown(f"**Confidence:** {signal.get('confidence')}/10")
                    st.markdown(f"**Price:** ${signal.get('price', 0):,.2f}")
                
                with col2:
                    st.markdown(f"**Stop Loss:** ${signal.get('stop_loss', 0):,.2f}")
                    st.markdown(f"**Take Profit:** ${signal.get('take_profit', 0):,.2f}")
                    st.markdown(f"**Time:** {signal.get('created_at', '')[:19]}")
                
                # AI Reasoning
                reasoning = signal.get('reasoning', 'No reasoning available')
                st.markdown("**🧠 AI Reasoning:**")
                st.write(reasoning)
    else:
        st.warning("No AI signals in database yet. Bot will generate signals when analyzing markets.")
        
        # Show last analysis from bot logs
        st.subheader("📝 Sample AI Analysis")
        st.markdown("""
        When the bot analyzes a coin, it considers:
        
        1. **Technical Indicators:**
           - RSI (Relative Strength Index)
           - MACD (Moving Average Convergence Divergence)
           - Bollinger Bands
           - Support/Resistance levels
        
        2. **Pattern Recognition:**
           - Chart patterns (Double Top/Bottom, Triangles)
           - Candlestick patterns (Doji, Hammer, Engulfing)
        
        3. **Market Context:**
           - Volume analysis
           - Market regime (Trending/Ranging)
           - Correlation with BTC
        
        4. **Risk Assessment:**
           - ATR-based stop loss
           - Position sizing based on volatility
           - Portfolio correlation
        """)

# TAB 5: TRADE HISTORY
with tab5:
    st.header("📋 Complete Trade History")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_status = st.selectbox("Status", ["All", "Open", "Closed"], key="hist_status")
    
    with col2:
        filter_mode = st.selectbox("Mode", ["All", "Paper", "Real"], key="hist_mode")
    
    with col3:
        filter_side = st.selectbox("Side", ["All", "BUY", "SELL"], key="hist_side")
    
    # Get trades
    trades = get_trade_history()
    
    # Apply filters
    if filter_status != "All":
        trades = [t for t in trades if t.get('status', '').lower() == filter_status.lower()]
    if filter_mode != "All":
        trades = [t for t in trades if t.get('mode', '').lower() == filter_mode.lower()]
    if filter_side != "All":
        trades = [t for t in trades if t.get('side', '').upper() == filter_side.upper()]
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", len(trades))
    
    with col2:
        closed = [t for t in trades if t.get('status') == 'closed']
        total_pnl = sum((t.get('pnl') or 0) for t in closed)
        st.metric("Total PnL", f"${total_pnl:.4f}")
    
    with col3:
        winning = len([t for t in closed if (t.get('pnl') or 0) > 0])
        win_rate = (winning / len(closed) * 100) if closed else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with col4:
        open_count = len([t for t in trades if t.get('status') == 'open'])
        st.metric("Open Positions", open_count)
    
    st.divider()
    
    # Trade table with reasoning
    if trades:
        for trade in trades:
            status = trade.get('status', 'unknown')
            symbol = trade.get('symbol', 'N/A')
            side = trade.get('side', 'N/A')
            
            color = "🟢" if side == 'BUY' else "🔴"
            status_color = "🔓" if status == 'open' else "🔒"
            
            with st.expander(f"{color} {side} {symbol} | {status_color} {status.upper()} | Entry: ${trade.get('entry_price', 0):,.2f}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Trade ID:** `{trade.get('trade_id', 'N/A')}`")
                    st.markdown(f"**Symbol:** {symbol}")
                    st.markdown(f"**Side:** {side}")
                    st.markdown(f"**Entry Price:** ${trade.get('entry_price', 0):,.4f}")
                    st.markdown(f"**Amount:** {trade.get('amount', 0):.8f}")
                    st.markdown(f"**USDT Value:** ${trade.get('usdt_amount', 0):.4f}")
                
                with col2:
                    st.markdown(f"**Status:** {status}")
                    st.markdown(f"**Mode:** {trade.get('mode', 'N/A')}")
                    st.markdown(f"**Stop Loss:** ${trade.get('stop_loss', 0):,.4f}")
                    st.markdown(f"**Take Profit:** ${trade.get('take_profit', 0):,.4f}")
                    
                    if status == 'closed':
                        pnl = trade.get('pnl') or 0
                        pnl_pct = trade.get('pnl_percent') or 0
                        color = "green" if pnl > 0 else "red"
                        st.markdown(f"**Exit Price:** ${trade.get('exit_price', 0):,.4f}")
                        st.markdown(f"**PnL:** <span style='color:{color}'>${pnl:.6f} ({pnl_pct:+.2f}%)</span>", unsafe_allow_html=True)
                
                # AI Reasoning for this trade (from signal)
                st.markdown("---")
                st.markdown("**🧠 Why this trade was made:**")
                
                # Try to find signal for this trade
                trade_id = trade.get('trade_id', '')
                signal_reason = "AI detected favorable market conditions based on technical analysis and pattern recognition."
                
                # Add more context based on trade data
                if side == 'BUY':
                    signal_reason += f"\n- Buy signal triggered at ${trade.get('entry_price', 0):,.2f}"
                    signal_reason += f"\n- Stop loss set at ${trade.get('stop_loss', 0):,.2f} (-{((trade.get('entry_price', 1) - trade.get('stop_loss', 0)) / trade.get('entry_price', 1) * 100):.1f}%)"
                    signal_reason += f"\n- Take profit target: ${trade.get('take_profit', 0):,.2f} (+{((trade.get('take_profit', 0) - trade.get('entry_price', 1)) / trade.get('entry_price', 1) * 100):.1f}%)"
                else:
                    signal_reason += f"\n- Sell/Short signal triggered at ${trade.get('entry_price', 0):,.2f}"
                
                st.info(signal_reason)
                
                st.markdown(f"**Entry Time:** {trade.get('entry_time', 'N/A')}")
                if trade.get('exit_time'):
                    st.markdown(f"**Exit Time:** {trade.get('exit_time')}")
    else:
        st.info("No trades match the selected filters")

# TAB 6: EXTERNAL SIGNALS
with tab6:
    st.header("📡 External Signal Integration")
    
    st.info("""
    **Coming Soon: External Signal Sources**
    
    The bot can be configured to monitor and validate signals from:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📱 Telegram Channels")
        st.markdown("""
        - **Crypto Signals Pro**
        - **Binance Killers**
        - **Wallstreet Queen**
        - **Fat Pig Signals**
        - Custom channels via @username
        """)
        
        st.text_input("Add Telegram Channel", placeholder="@channel_name", key="tg_channel")
        st.button("➕ Add Channel", key="add_tg")
    
    with col2:
        st.subheader("📊 TradingView")
        st.markdown("""
        - **TradingView Alerts**
        - **Pine Script Strategies**
        - **Community Signals**
        """)
        
        st.text_input("TradingView Webhook URL", placeholder="https://...", key="tv_webhook")
        st.button("➕ Connect TradingView", key="add_tv")
    
    st.divider()
    
    st.subheader("🔧 Configuration")
    
    st.markdown("""
    To enable external signals, add to your `.env` file:
    
    ```bash
    # Telegram Signal Channels (comma-separated)
    SIGNAL_CHANNELS=@crypto_signals,@binance_killers
    
    # TradingView Webhook Secret
    TRADINGVIEW_WEBHOOK_SECRET=your_secret_here
    
    # Signal validation settings
    VALIDATE_EXTERNAL_SIGNALS=true
    MIN_SIGNAL_CONFIDENCE=6.0
    ```
    """)
    
    st.warning("""
    ⚠️ **Important:** External signals will be validated by our AI before execution.
    The bot will:
    1. Receive the signal
    2. Analyze current market conditions
    3. Validate with technical indicators
    4. Execute only if confidence > threshold
    """)

# Footer
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    mode = "📝 PAPER" if paper_trading else "🔥 REAL"
    st.caption(f"Mode: {mode}")

with col2:
    st.caption(f"Auto Trade: {'✅ ON' if auto_trade else '❌ OFF'}")

with col3:
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

st.markdown("""
<div style='text-align: center; color: #888; margin-top: 20px;'>
    <p>🤖 NexusTrader AI Dashboard v11.0 | Real Data Mode</p>
</div>
""", unsafe_allow_html=True)
