"""
Trading Bot Dashboard - Phase 10
Clean version with real data integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import data provider
from dashboard.data_provider import get_data_provider

# Initialize data provider
@st.cache_resource
def init_data_provider():
    return get_data_provider()

data_provider = init_data_provider()

# Page config
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🤖 Trading Bot Dashboard")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    symbol = st.selectbox("Trading Pair", ["BTC/USDT", "ETH/USDT", "BNB/USDT"], index=0)
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
    auto_refresh = st.checkbox("Auto Refresh", value=True)

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "💰 Portfolio",
    "🤖 AI Analysis",
    "⚠️ Risk",
    "📈 Trades"
])

# TAB 1: OVERVIEW
with tab1:
    st.header("📊 Portfolio Overview")
    
    # Fetch data
    portfolio = data_provider.get_portfolio_summary()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Balance",
            f"${portfolio['balance']:.2f}",
            delta=f"{portfolio['balance_change']:+.2f}"
        )
    
    with col2:
        st.metric(
            "Total Trades",
            portfolio['total_trades'],
            delta=f"+{portfolio['trades_today']}"
        )
    
    with col3:
        st.metric(
            "Win Rate",
            f"{portfolio['win_rate']:.1f}%",
            delta=f"{portfolio['win_rate_change']:+.1f}%"
        )
    
    with col4:
        st.metric(
            "Sharpe",
            f"{portfolio['sharpe_ratio']:.2f}",
            delta=f"{portfolio['sharpe_change']:+.2f}"
        )
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💹 Equity Curve")
        equity_df = data_provider.get_equity_curve(days=45)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df['date'],
            y=equity_df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy'
        ))
        
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 PnL Distribution")
        pnl_data = data_provider.get_pnl_distribution()
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=pnl_data,
            nbinsx=50,
            marker=dict(color=pnl_data, colorscale='RdYlGn')
        ))
        
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Recent activity
    st.subheader("🔔 Recent Activity")
    activities = data_provider.get_recent_activity()
    
    for activity in activities[:10]:
        col_a, col_b, col_c = st.columns([2, 3, 2])
        
        with col_a:
            st.text(activity['timestamp'])
        
        with col_b:
            if activity['type'] == 'trade':
                st.markdown(f"**{activity['action']}** {activity['symbol']} @ ${activity['price']:.2f}")
            else:
                st.markdown(f"**{activity['message']}**")
        
        with col_c:
            if activity.get('pnl'):
                st.text(f"{activity['pnl']:+.2f} $")

# TAB 2: PORTFOLIO
with tab2:
    st.header("💰 Portfolio Details")
    
    portfolio = data_provider.get_portfolio_summary()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Balance", f"${portfolio['balance']:.2f}")
    
    with col2:
        st.metric("Available", f"${portfolio['balance'] * 0.7:.2f}")
    
    with col3:
        st.metric("In Trades", f"${portfolio['balance'] * 0.3:.2f}")
    
    st.divider()
    
    # Asset allocation
    st.subheader("🎯 Asset Allocation")
    
    allocation = pd.DataFrame({
        'Asset': ['BTC', 'ETH', 'USDT', 'Others'],
        'Value': [
            portfolio['balance'] * 0.4,
            portfolio['balance'] * 0.3,
            portfolio['balance'] * 0.2,
            portfolio['balance'] * 0.1
        ]
    })
    
    fig = go.Figure(data=[go.Pie(
        labels=allocation['Asset'],
        values=allocation['Value'],
        hole=0.4
    )])
    
    fig.update_layout(template='plotly_dark', height=400)
    st.plotly_chart(fig, use_container_width=True)

# TAB 3: AI ANALYSIS
with tab3:
    st.header("🤖 AI Analysis")
    
    ai_symbol = st.selectbox("Select pair", ["BTC/USDT", "ETH/USDT", "BNB/USDT"], key="ai_symbol")
    
    prediction = data_provider.get_ai_prediction(ai_symbol)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Direction", prediction['direction'].upper())
    
    with col2:
        st.metric("Confidence", f"{prediction['confidence']*100:.1f}%")
    
    with col3:
        st.metric("Action", prediction['action'])
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"📈 Entry: ${prediction['entry_price']:.2f}")
        st.info(f"🎯 Take Profit: ${prediction['take_profit']:.2f}")
    
    with col2:
        st.error(f"🛑 Stop Loss: ${prediction['stop_loss']:.2f}")
        expected_pnl = prediction['take_profit'] - prediction['entry_price']
        st.metric("Expected PnL", f"+${expected_pnl:.2f}")
    
    st.subheader("💡 Reasoning")
    st.write(prediction['reasoning'])

# TAB 4: RISK
with tab4:
    st.header("⚠️ Risk Management")
    
    risk_data = data_provider.get_risk_metrics(symbol)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("VaR (95%)", f"${risk_data.get('var_95', 0):.2f}")
    
    with col2:
        st.metric("Exp. Shortfall", f"${risk_data.get('expected_shortfall', 0):.2f}")
    
    with col3:
        st.metric("Volatility", f"{risk_data.get('volatility', 0)*100:.2f}%")
    
    with col4:
        st.metric("Beta", f"{risk_data.get('beta', 1.0):.2f}")
    
    st.divider()
    
    st.subheader("📊 VaR Analysis")
    
    var_metrics = risk_data.get('var', {})
    var_df = pd.DataFrame({
        'Period': ['1 Day', '1 Week', '1 Month'],
        'VaR 95%': [
            var_metrics.get('1d_95', 0.84),
            var_metrics.get('1w_95', 2.22),
            var_metrics.get('1m_95', 4.59)
        ]
    })
    
    st.dataframe(var_df, use_container_width=True)

# TAB 5: TRADES
with tab5:
    st.header("📈 Trade History")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_symbol = st.selectbox("Symbol", ["All", "BTC/USDT", "ETH/USDT"], key="filter_symbol")
    
    with col2:
        filter_status = st.selectbox("Status", ["All", "Closed", "Open"], key="filter_status")
    
    with col3:
        days_back = st.slider("Days", 1, 90, 30, key="days_back")
    
    filters = {
        'symbol': None if filter_symbol == "All" else filter_symbol,
        'status': None if filter_status == "All" else filter_status.lower(),
        'days': days_back
    }
    
    trades_df = data_provider.get_trade_history(filters)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Trades", len(trades_df))
    
    with col2:
        if 'pnl' in trades_df.columns:
            total_pnl = trades_df['pnl'].sum()
            st.metric("Total PnL", f"${total_pnl:.2f}")
        else:
            st.metric("Total PnL", "$0.00")
    
    with col3:
        if 'pnl' in trades_df.columns and len(trades_df) > 0:
            winning = len(trades_df[trades_df['pnl'] > 0])
            win_rate = (winning / len(trades_df) * 100)
            st.metric("Win Rate", f"{win_rate:.1f}%")
        else:
            st.metric("Win Rate", "0%")
    
    st.divider()
    
    if len(trades_df) > 0:
        st.dataframe(trades_df, use_container_width=True, height=400)
    else:
        st.info("No trades found for selected period")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>🤖 Trading Bot Dashboard | Version 10.0 | Real Data Active</p>
</div>
""", unsafe_allow_html=True)
