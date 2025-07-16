import streamlit as st
import pandas as pd
import os
import json

st.set_page_config(page_title="Forex AI Signals Dashboard", layout="wide")
st.title("📈 Forex & Crypto AI Signals Dashboard")

SIGNALS_CSV = 'logs/signals.csv'
EQUITY_CURVE_CSV = 'logs/equity_curve.csv'
ANALYTICS_JSON = 'logs/analytics_summary.json'

# --- Load latest signals ---
def load_signals():
    if os.path.exists(SIGNALS_CSV):
        df = pd.read_csv(SIGNALS_CSV)
        df = df.sort_values('time', ascending=False)
        return df
    else:
        return pd.DataFrame()

def load_equity_curve():
    if os.path.exists(EQUITY_CURVE_CSV):
        df = pd.read_csv(EQUITY_CURVE_CSV)
        return df
    else:
        return pd.DataFrame()

# --- Advanced Analytics ---
def load_analytics():
    if os.path.exists(ANALYTICS_JSON):
        with open(ANALYTICS_JSON, 'r') as f:
            return json.load(f)
    return None

# --- Sidebar ---
st.sidebar.header("Options")
def rerun():
    if hasattr(st, 'experimental_rerun'):
        st.experimental_rerun()
    elif hasattr(st, 'rerun'):
        st.rerun()
    else:
        pass
if st.sidebar.button("Refresh"):
    rerun()

# --- Main content ---
signals_df = load_signals()
if not signals_df.empty:
    st.subheader("Latest Signals")
    st.dataframe(signals_df.head(10), use_container_width=True)
    st.metric("Last Signal Confidence", f"{signals_df.iloc[0]['confidence']:.2%}")
    st.metric("Last Signal Pair", signals_df.iloc[0]['pair'])
    st.metric("Last Signal Type", signals_df.iloc[0]['signal'])
else:
    st.info("No signals found. Waiting for new signals...")

# --- Equity Curve ---
equity_df = load_equity_curve()
if not equity_df.empty:
    st.subheader("Equity Curve")
    st.line_chart(equity_df['equity'])
else:
    st.info("No equity curve data found. Run backtest to generate.")

# --- Metrics ---
if not signals_df.empty:
    st.subheader("Performance Metrics (Recent)")
    win_rate = (signals_df['profit'] > 0).mean() if 'profit' in signals_df else None
    total_signals = len(signals_df)
    avg_conf = signals_df['confidence'].mean()
    st.metric("Total Signals", total_signals)
    if win_rate is not None:
        st.metric("Win Rate", f"{win_rate:.2%}")
    st.metric("Avg Confidence", f"{avg_conf:.2%}")
    if 'profit' in signals_df:
        st.metric("Total PnL", f"{signals_df['profit'].sum():.2f}") 

# --- Advanced Analytics Section ---
analytics = load_analytics()
if analytics:
    st.subheader('Advanced Analytics Summary')
    col1, col2, col3 = st.columns(3)
    col1.metric('Total Signals', analytics['total_signals'])
    col1.metric('Win Rate', f"{analytics['win_rate']:.2%}")
    col1.metric('Avg R:R', f"{analytics['avg_rr']:.2f}" if analytics['avg_rr'] is not None else 'N/A')
    col2.metric('Expectancy', f"{analytics['expectancy']:.2f}" if analytics['expectancy'] is not None else 'N/A')
    col2.metric('Max Win Streak', analytics['max_win_streak'])
    col2.metric('Max Loss Streak', analytics['max_loss_streak'])
    col3.metric('Max Drawdown', f"{analytics['max_drawdown']:.2f}")
    col3.metric('Profit Factor', f"{analytics['profit_factor']:.2f}")
    col3.metric('Sharpe Ratio', f"{analytics['sharpe_ratio']:.2f}")
    col3.metric('Final Equity', f"{analytics['final_equity']:.2f}")
    st.markdown('**Per-Pair Stats:**')
    st.json(analytics['per_pair_stats'])
    if analytics.get('equity_curve'):
        st.subheader('Analytics Equity Curve')
        st.line_chart(analytics['equity_curve']) 