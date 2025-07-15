import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Forex AI Signals Dashboard", layout="wide")
st.title("📈 Forex & Crypto AI Signals Dashboard")

SIGNALS_CSV = 'logs/signals.csv'
EQUITY_CURVE_CSV = 'logs/equity_curve.csv'

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