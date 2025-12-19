"""
NexusTrader Dashboard v15 - Production Ready
Full Portfolio Control + Health + Metrics + Backtesting
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import ccxt
from database_supabase import SupabaseDatabase
from modules.portfolio_manager import PortfolioManager, ActionType
from modules.performance_metrics import PerformanceTracker
from modules.health_monitor import HealthMonitor

# === CONFIG ===
st.set_page_config(
    page_title="NexusTrader",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark theme
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stMetric {background: #1a1f2e; padding: 15px; border-radius: 10px; border: 1px solid #30363d;}
    .stMetric label {color: #8b949e !important; font-size: 14px !important;}
    .stMetric [data-testid="stMetricValue"] {color: #fff !important; font-size: 24px !important;}
    .block-container {padding-top: 1.5rem;}
    div[data-testid="stExpander"] {background: #161b22; border: 1px solid #30363d; border-radius: 8px;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {background: #21262d; border-radius: 8px; padding: 10px 20px;}
    .stTabs [aria-selected="true"] {background: #238636;}
</style>
""", unsafe_allow_html=True)

# === CACHE ===
@st.cache_resource
def get_portfolio_manager():
    return PortfolioManager()

@st.cache_resource
def get_db():
    return SupabaseDatabase()

@st.cache_resource
def get_performance_tracker():
    return PerformanceTracker()

@st.cache_data(ttl=60)
def fetch_health_status():
    monitor = HealthMonitor()
    asyncio.run(monitor.run_all_checks())
    return {name: {'status': c.status, 'message': c.message} for name, c in monitor.checks.items()}

@st.cache_data(ttl=60)
def fetch_performance_metrics():
    return get_performance_tracker().get_metrics_json(days=30)

@st.cache_data(ttl=20)
def fetch_portfolio():
    return get_portfolio_manager().get_full_portfolio()

@st.cache_data(ttl=30)
def fetch_recommendations():
    return get_portfolio_manager().analyze_portfolio()

@st.cache_data(ttl=10)
def fetch_ticker(symbol):
    try:
        return get_portfolio_manager().exchange.fetch_ticker(symbol)
    except:
        return {'last': 0, 'percentage': 0}

@st.cache_data(ttl=60)
def fetch_chart(symbol, tf='1h', limit=50):
    try:
        ohlcv = get_portfolio_manager().exchange.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=30)
def fetch_trades():
    try:
        trades = get_db().get_trade_history(limit=50)
        return [t for t in trades if not t.get('trade_id', '').startswith('TEST_')]
    except:
        return []

@st.cache_data(ttl=30)
def fetch_signals():
    try:
        result = get_db().client.table('signals').select('*').order('created_at', desc=True).limit(20).execute()
        return [s for s in (result.data or []) if not s.get('trade_id', '').startswith('TEST_')]
    except:
        return []

# === HEADER ===
col1, col2, col3 = st.columns([4, 1, 1])
with col1:
    st.markdown("# 🤖 NexusTrader")
with col2:
    paper = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
    mode_color = "#f0ad4e" if paper else "#238636"
    mode_text = "PAPER" if paper else "LIVE"
    st.markdown(f"<div style='background:{mode_color};padding:8px 16px;border-radius:20px;text-align:center;font-weight:bold;color:#fff;'>{mode_text}</div>", unsafe_allow_html=True)
with col3:
    if st.button("🔄", help="Refresh"):
        st.cache_data.clear()
        st.rerun()

# === LOAD DATA ===
portfolio = fetch_portfolio()
recommendations = fetch_recommendations()
trades = fetch_trades()
signals = fetch_signals()

# === TOP METRICS ===
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("💰 Total", f"${portfolio['total_usd']:,.2f}")
with col2:
    st.metric("💵 Spot", f"${portfolio['spot_usd']:,.2f}", help="Available for trading")
with col3:
    st.metric("🔒 Earn", f"${portfolio['earn_usd']:,.2f}", help="Earning yield")
with col4:
    btc = fetch_ticker('BTC/USDT')
    st.metric("₿ BTC", f"${btc['last']:,.0f}", f"{btc.get('percentage', 0):+.1f}%")
with col5:
    open_trades = len([t for t in trades if t.get('status') == 'open'])
    st.metric("📊 Positions", open_trades)

# === TABS ===
st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎯 AI Recommendations", "💼 Portfolio", "📈 Trading", "📋 History", "🏥 Health", "📊 Metrics"])

# === TAB 1: AI RECOMMENDATIONS ===
with tab1:
    st.markdown("### 🤖 AI Portfolio Analysis")
    st.caption("Every recommendation is analyzed and justified by AI")
    
    if recommendations:
        # Summary
        action_counts = {}
        for r in recommendations:
            action_counts[r.action.value] = action_counts.get(r.action.value, 0) + 1
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("🔄 To Earn", action_counts.get('to_earn', 0))
        with cols[1]:
            st.metric("💸 From Earn", action_counts.get('from_earn', 0))
        with cols[2]:
            st.metric("📉 Sell", action_counts.get('sell', 0))
        with cols[3]:
            st.metric("✋ Hold", action_counts.get('hold', 0))
        
        st.markdown("---")
        
        # Recommendations list
        for i, rec in enumerate(recommendations[:10]):
            # Color by action type
            if rec.action == ActionType.TO_EARN:
                icon = "🔐"
                color = "#238636"
                action_text = "Move to Earn"
            elif rec.action == ActionType.FROM_EARN:
                icon = "💸"
                color = "#f0ad4e"
                action_text = "Redeem from Earn"
            elif rec.action == ActionType.SELL:
                icon = "📉"
                color = "#f85149"
                action_text = "Sell"
            elif rec.action == ActionType.BUY:
                icon = "📈"
                color = "#3fb950"
                action_text = "Buy"
            else:
                icon = "✋"
                color = "#8b949e"
                action_text = "Hold"
            
            with st.expander(f"{icon} {rec.asset} - {action_text} | Confidence: {rec.confidence}/10"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**🧠 AI Reasoning:**")
                    st.markdown(f"_{rec.reason}_")
                
                with col2:
                    st.markdown(f"**Risk:** {rec.risk_level.upper()}")
                    if rec.expected_apy:
                        st.markdown(f"**Expected APY:** {rec.expected_apy}%")
                    st.markdown(f"**Amount:** {rec.amount:.6f} {rec.asset}")
                    
                    # Execute button
                    if rec.action != ActionType.HOLD:
                        if st.button(f"Execute {action_text}", key=f"exec_{i}"):
                            pm = get_portfolio_manager()
                            success, msg = pm.execute_action(rec)
                            if success:
                                st.success(msg)
                            else:
                                st.error(msg)
    else:
        st.info("No recommendations at this time. Portfolio is optimized!")

# === TAB 2: PORTFOLIO ===
with tab2:
    col1, col2 = st.columns(2)
    
    # === SPOT ===
    with col1:
        st.markdown("### 💵 Spot (Tradeable)")
        
        if portfolio['spot']:
            for asset, data in sorted(portfolio['spot'].items(), key=lambda x: x[1]['usd'], reverse=True):
                pct = (data['usd'] / portfolio['total_usd'] * 100) if portfolio['total_usd'] > 0 else 0
                
                st.markdown(f"""
                <div style="background:#161b22;padding:12px;border-radius:8px;margin:8px 0;border-left:4px solid #3fb950;">
                    <div style="display:flex;justify-content:space-between;">
                        <span style="font-weight:bold;font-size:16px;">{asset}</span>
                        <span style="color:#3fb950;font-weight:bold;">${data['usd']:.2f}</span>
                    </div>
                    <div style="color:#8b949e;font-size:13px;">
                        {data['amount']:.6f} | {pct:.1f}% of portfolio
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No assets in Spot")
        
        st.metric("Spot Total", f"${portfolio['spot_usd']:.2f}")
    
    # === EARN ===
    with col2:
        st.markdown("### 🔒 Earn (Yielding)")
        
        if portfolio['earn']:
            # Sort by value
            earn_sorted = sorted(portfolio['earn'].items(), key=lambda x: x[1]['usd'], reverse=True)
            
            for asset, data in earn_sorted[:15]:  # Top 15
                apy = data.get('apy', 0)
                daily = (data['usd'] * apy / 100) / 365
                
                st.markdown(f"""
                <div style="background:#161b22;padding:12px;border-radius:8px;margin:8px 0;border-left:4px solid #f0ad4e;">
                    <div style="display:flex;justify-content:space-between;">
                        <span style="font-weight:bold;font-size:16px;">{asset}</span>
                        <span style="color:#f0ad4e;font-weight:bold;">${data['usd']:.2f}</span>
                    </div>
                    <div style="color:#8b949e;font-size:13px;">
                        {data['amount']:.6f} | APY: {apy}% | ${daily:.4f}/day
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if len(earn_sorted) > 15:
                st.caption(f"+ {len(earn_sorted) - 15} more assets...")
        else:
            st.info("No assets in Earn")
        
        st.metric("Earn Total", f"${portfolio['earn_usd']:.2f}")
    
    # Allocation chart
    st.markdown("---")
    st.markdown("### 📊 Allocation")
    
    # Prepare data for pie chart
    all_assets = []
    for asset, data in portfolio['spot'].items():
        all_assets.append({'asset': f"{asset} (Spot)", 'usd': data['usd'], 'type': 'Spot'})
    for asset, data in list(portfolio['earn'].items())[:10]:
        all_assets.append({'asset': f"{asset} (Earn)", 'usd': data['usd'], 'type': 'Earn'})
    
    if all_assets:
        all_assets.sort(key=lambda x: x['usd'], reverse=True)
        top_assets = all_assets[:8]
        
        fig = go.Figure(data=[go.Pie(
            labels=[a['asset'] for a in top_assets],
            values=[a['usd'] for a in top_assets],
            hole=0.4,
            marker_colors=['#3fb950' if a['type'] == 'Spot' else '#f0ad4e' for a in top_assets]
        )])
        fig.update_layout(
            template='plotly_dark',
            height=300,
            showlegend=True,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='#0e1117',
            legend=dict(orientation="h", y=-0.1)
        )
        st.plotly_chart(fig, use_container_width=True)

# === TAB 3: TRADING ===
with tab3:
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("### 📈 Chart")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            symbol = st.selectbox("", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ARB/USDT"], label_visibility="collapsed")
        with c2:
            tf = st.selectbox("", ["15m", "1h", "4h", "1d"], index=1, label_visibility="collapsed", key="tf")
        
        df = fetch_chart(symbol, tf)
        
        if not df.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=df['ts'],
                open=df['o'], high=df['h'],
                low=df['l'], close=df['c'],
                increasing_line_color='#3fb950',
                decreasing_line_color='#f85149'
            )])
            fig.update_layout(
                template='plotly_dark',
                height=350,
                xaxis_rangeslider_visible=False,
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117',
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🤖 AI Signals")
        
        if signals:
            for s in signals[:5]:
                sig_type = s.get('signal_type', 'N/A')
                conf = s.get('ai_confidence', 0)
                reason = s.get('ai_reason', 'No reason')
                
                color = "#3fb950" if sig_type in ['BUY', 'LONG'] else "#f85149"
                icon = "🟢" if sig_type in ['BUY', 'LONG'] else "🔴"
                
                with st.expander(f"{icon} {s.get('symbol', 'N/A')} - {sig_type} ({conf}/10)"):
                    st.markdown(f"**Price:** ${s.get('price', 0):,.4f}")
                    st.markdown(f"**🧠 Reason:** _{reason}_")
        else:
            st.info("Waiting for signals...")
    
    # Open positions
    st.markdown("---")
    st.markdown("### 📊 Open Positions")
    
    open_trades = [t for t in trades if t.get('status') == 'open']
    
    if open_trades:
        cols = st.columns(len(open_trades) if len(open_trades) <= 4 else 4)
        for i, t in enumerate(open_trades[:4]):
            with cols[i]:
                symbol = t.get('symbol', 'N/A')
                side = t.get('side', 'N/A')
                entry = t.get('entry_price', 0)
                
                try:
                    current = fetch_ticker(symbol)['last']
                    pnl_pct = ((current - entry) / entry * 100) if side == 'BUY' else ((entry - current) / entry * 100)
                except:
                    current = entry
                    pnl_pct = 0
                
                color = "#3fb950" if pnl_pct >= 0 else "#f85149"
                
                st.markdown(f"""
                <div style="background:#161b22;padding:15px;border-radius:10px;text-align:center;">
                    <div style="font-size:14px;color:#8b949e;">{symbol}</div>
                    <div style="font-size:12px;margin:5px 0;">{side} @ ${entry:.4f}</div>
                    <div style="font-size:24px;font-weight:bold;color:{color};">{pnl_pct:+.2f}%</div>
                    <div style="font-size:12px;color:#8b949e;">Now: ${current:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No open positions")

# === TAB 4: HISTORY ===
with tab4:
    st.markdown("### 📋 Trade History")
    
    closed_trades = [t for t in trades if t.get('status') == 'closed']
    
    if closed_trades:
        # Stats
        total_pnl = sum((t.get('pnl') or 0) for t in closed_trades)
        wins = len([t for t in closed_trades if (t.get('pnl') or 0) > 0])
        wr = (wins / len(closed_trades) * 100) if closed_trades else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            color = "#3fb950" if total_pnl >= 0 else "#f85149"
            st.metric("Total P&L", f"${total_pnl:.4f}")
        with col2:
            st.metric("Win Rate", f"{wr:.0f}%")
        with col3:
            st.metric("Trades", len(closed_trades))
        
        st.markdown("---")
        
        for t in closed_trades[:10]:
            pnl = t.get('pnl') or 0
            pnl_pct = t.get('pnl_percent') or 0
            color = "#3fb950" if pnl >= 0 else "#f85149"
            icon = "✅" if pnl >= 0 else "❌"
            
            with st.expander(f"{icon} {t.get('symbol')} | {t.get('side')} | {pnl_pct:+.2f}%"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Entry:** ${t.get('entry_price', 0):.4f}")
                    st.markdown(f"**Exit:** ${t.get('exit_price', 0):.4f}")
                with c2:
                    st.markdown(f"**P&L:** <span style='color:{color}'>${pnl:.6f}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Mode:** {t.get('mode', 'N/A')}")
    else:
        st.info("No closed trades yet")

# === TAB 5: HEALTH ===
with tab5:
    st.markdown("### 🏥 System Health Monitor")
    st.caption("Real-time status of all bot components")
    
    health = fetch_health_status()
    
    # Overall status
    status = health.get('status', 'unknown')
    status_colors = {'healthy': '#3fb950', 'warning': '#f0ad4e', 'critical': '#f85149', 'unknown': '#8b949e'}
    status_icons = {'healthy': '✅', 'warning': '⚠️', 'critical': '🚨', 'unknown': '❓'}
    
    st.markdown(f"""
    <div style="background:{status_colors.get(status, '#8b949e')};padding:20px;border-radius:10px;text-align:center;margin-bottom:20px;">
        <span style="font-size:48px;">{status_icons.get(status, '❓')}</span>
        <h2 style="margin:10px 0;color:white;">{status.upper()}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Component details
    st.markdown("### Component Status")
    
    details = health.get('details', {})
    cols = st.columns(3)
    
    component_icons = {
        'exchange': '📊', 'balance': '💰', 'database': '🗄️',
        'positions': '📈', 'daily_pnl': '💵'
    }
    
    for i, (component, data) in enumerate(details.items()):
        with cols[i % 3]:
            comp_status = data.get('status', 'unknown')
            icon = component_icons.get(component, '🔧')
            
            if comp_status == 'ok':
                bg = '#238636'
                status_text = '✅ OK'
            elif comp_status == 'warning':
                bg = '#f0ad4e'
                status_text = '⚠️ Warning'
            else:
                bg = '#f85149'
                status_text = '❌ Error'
            
            message = data.get('message', '')[:50]
            
            st.markdown(f"""
            <div style="background:#161b22;padding:15px;border-radius:8px;margin:10px 0;border-left:4px solid {bg};">
                <div style="font-size:20px;">{icon} {component.replace('_', ' ').title()}</div>
                <div style="color:{bg};font-weight:bold;margin:5px 0;">{status_text}</div>
                <div style="color:#8b949e;font-size:12px;">{message}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Last check time
    st.markdown("---")
    last_check = health.get('timestamp', 'Unknown')
    st.caption(f"Last check: {last_check}")

# === TAB 6: METRICS ===
with tab6:
    st.markdown("### 📊 Performance Metrics")
    st.caption("Trading statistics and performance analysis")
    
    metrics = fetch_performance_metrics()
    
    if metrics:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_pnl = metrics.get('total_pnl', 0)
            pnl_color = "#3fb950" if total_pnl >= 0 else "#f85149"
            st.markdown(f"""
            <div style="background:#161b22;padding:20px;border-radius:10px;text-align:center;">
                <div style="color:#8b949e;font-size:14px;">Total P&L</div>
                <div style="color:{pnl_color};font-size:28px;font-weight:bold;">${total_pnl:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            win_rate = metrics.get('win_rate', 0)
            wr_color = "#3fb950" if win_rate >= 50 else "#f85149"
            st.markdown(f"""
            <div style="background:#161b22;padding:20px;border-radius:10px;text-align:center;">
                <div style="color:#8b949e;font-size:14px;">Win Rate</div>
                <div style="color:{wr_color};font-size:28px;font-weight:bold;">{win_rate:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            sharpe = metrics.get('sharpe_ratio', 0)
            sharpe_color = "#3fb950" if sharpe >= 1 else "#f0ad4e" if sharpe >= 0 else "#f85149"
            st.markdown(f"""
            <div style="background:#161b22;padding:20px;border-radius:10px;text-align:center;">
                <div style="color:#8b949e;font-size:14px;">Sharpe Ratio</div>
                <div style="color:{sharpe_color};font-size:28px;font-weight:bold;">{sharpe:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            max_dd = metrics.get('max_drawdown', 0)
            dd_color = "#3fb950" if max_dd > -5 else "#f0ad4e" if max_dd > -10 else "#f85149"
            st.markdown(f"""
            <div style="background:#161b22;padding:20px;border-radius:10px;text-align:center;">
                <div style="color:#8b949e;font-size:14px;">Max Drawdown</div>
                <div style="color:{dd_color};font-size:28px;font-weight:bold;">{max_dd:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional stats
        st.markdown("---")
        st.markdown("### 📈 Detailed Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background:#161b22;padding:20px;border-radius:10px;">
                <h4 style="color:#58a6ff;margin-bottom:15px;">📊 Trade Stats</h4>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Total Trades:** {metrics.get('total_trades', 0)}")
            st.markdown(f"**Winning Trades:** {metrics.get('winning_trades', 0)}")
            st.markdown(f"**Losing Trades:** {metrics.get('losing_trades', 0)}")
            st.markdown(f"**Avg Win:** ${metrics.get('average_win', 0):.4f}")
            st.markdown(f"**Avg Loss:** ${metrics.get('average_loss', 0):.4f}")
            st.markdown(f"**Profit Factor:** {metrics.get('profit_factor', 0):.2f}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background:#161b22;padding:20px;border-radius:10px;">
                <h4 style="color:#58a6ff;margin-bottom:15px;">🔥 Streaks</h4>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Current Streak:** {metrics.get('current_streak', 0)}")
            st.markdown(f"**Max Win Streak:** {metrics.get('max_win_streak', 0)}")
            st.markdown(f"**Max Loss Streak:** {metrics.get('max_loss_streak', 0)}")
            st.markdown(f"**Best Trade:** ${metrics.get('best_trade', 0):.4f}")
            st.markdown(f"**Worst Trade:** ${metrics.get('worst_trade', 0):.4f}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Daily performance chart
        st.markdown("---")
        st.markdown("### 📅 Daily Performance")
        
        daily = metrics.get('daily_pnl', {})
        if daily:
            dates = list(daily.keys())[-14:]  # Last 14 days
            values = [daily[d] for d in dates]
            
            fig = go.Figure(data=[go.Bar(
                x=dates,
                y=values,
                marker_color=['#3fb950' if v >= 0 else '#f85149' for v in values]
            )])
            fig.update_layout(
                template='plotly_dark',
                height=250,
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117',
                xaxis_title="Date",
                yaxis_title="P&L ($)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No daily data yet. Start trading to see performance!")
    else:
        st.info("No performance data available yet. Metrics will appear after trades.")

# === FOOTER ===
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
with col2:
    auto = os.getenv('AUTO_TRADE', 'false').lower() == 'true'
    st.caption(f"Auto-Trade: {'✅ ON' if auto else '❌ OFF'}")
with col3:
    st.caption("NexusTrader v15 | Production Ready 🚀")
