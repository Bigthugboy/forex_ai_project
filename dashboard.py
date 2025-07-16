import streamlit as st
import pandas as pd
import os
from datetime import datetime

# --- Branding and Config ---
PROJECT_NAME = "forex_signal_generator"
LOGO_TEXT = "FS"
COLOR_PRIMARY = "#111111"
COLOR_ACCENT = "#00FFB0"
SIGNALS_CSV = "logs/signals.csv"
ANALYTICS_JSON = "logs/analytics_summary.json"

# --- Page Config ---
st.set_page_config(
    page_title=f"{PROJECT_NAME} Dashboard",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': f"{PROJECT_NAME} - AI-powered Forex & Crypto Signal Generator"
    }
)

# --- Custom CSS for black theme and logo ---
st.markdown(f"""
    <style>
    .stApp {{ background-color: {COLOR_PRIMARY}; color: #fff; }}
    .css-18e3th9 {{ background-color: {COLOR_PRIMARY} !important; }}
    .css-1d391kg {{ background-color: {COLOR_PRIMARY} !important; }}
    .css-1v0mbdj p {{ color: #fff; }}
    .logo-circle {{
        display: flex; align-items: center; justify-content: center;
        background: {COLOR_ACCENT}; color: #111; border-radius: 50%;
        width: 48px; height: 48px; font-size: 2rem; font-weight: bold;
        margin-right: 0.5rem;
    }}
    .brand-title {{ font-size: 2rem; font-weight: bold; letter-spacing: 2px; color: #fff; }}
    .signal-table th, .signal-table td {{ color: #fff !important; }}
    .emoji-message {{ font-size: 2.5rem; text-align: center; margin-top: 2rem; }}
    .class-message {{ font-size: 1.2rem; text-align: center; margin-bottom: 2rem; color: #ccc; }}
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Branding ---
st.sidebar.markdown(f"""
<div style='display: flex; align-items: center;'>
    <div class='logo-circle'>{LOGO_TEXT}</div>
    <span class='brand-title'>{PROJECT_NAME}</span>
</div>
---
""", unsafe_allow_html=True)
st.sidebar.markdown("""
- [GitHub](https://github.com/your-repo)
- [Docs](https://github.com/your-repo/docs)
---
**Disclaimer:** This is not financial advice. Use at your own risk.
""")

# --- Main Title ---
st.markdown(f"<div style='display: flex; align-items: center;'><div class='logo-circle'>{LOGO_TEXT}</div><span class='brand-title'>{PROJECT_NAME} Dashboard</span></div>", unsafe_allow_html=True)
st.markdown("---")

# --- Load Data ---
def load_signals():
    if os.path.exists(SIGNALS_CSV):
        df = pd.read_csv(SIGNALS_CSV)
        if not df.empty:
            df = df.sort_values("time", ascending=False)
        return df
    return pd.DataFrame()

def load_analytics():
    if os.path.exists(ANALYTICS_JSON):
        import json
        with open(ANALYTICS_JSON, "r") as f:
            return json.load(f)
    return None

signals_df = load_signals()
analytics = load_analytics()

# --- Last Updated ---
st.markdown(f"<span style='color: #aaa;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>", unsafe_allow_html=True)

# --- Signal Table ---
st.subheader("📈 Latest Signals")
if not signals_df.empty:
    # Show only the most relevant columns
    show_cols = [
        'time', 'pair', 'signal', 'trade_type', 'confidence', 'entry', 'stop_loss',
        'take_profit_1', 'take_profit_2', 'take_profit_3', 'position_size', 'outcome'
    ]
    show_cols = [c for c in show_cols if c in signals_df.columns]
    st.dataframe(
        signals_df[show_cols].head(20),
        use_container_width=True,
        hide_index=True,
        column_config={
            'confidence': st.column_config.ProgressColumn("Confidence", format="%.2f", min_value=0, max_value=1, help="Model confidence (0-1)")
        }
    )
else:
    st.markdown("<div class='emoji-message'>😴</div>", unsafe_allow_html=True)
    st.markdown("<div class='class-message'>No signals available right now. The AI is still studying the market. Please check back soon!</div>", unsafe_allow_html=True)

# --- Analytics Section ---
st.subheader("📊 Advanced Analytics")
if analytics:
    st.markdown(f"**Total signals:** {analytics.get('total_signals', 'N/A')}")
    st.markdown(f"**Win rate:** {analytics.get('win_rate', 0):.2%}")
    st.markdown(f"**Avg R:R:** {analytics.get('avg_rr', 'N/A')}")
    st.markdown(f"**Expectancy:** {analytics.get('expectancy', 'N/A')}")
    st.markdown(f"**Max win streak:** {analytics.get('max_win_streak', 'N/A')}")
    st.markdown(f"**Max loss streak:** {analytics.get('max_loss_streak', 'N/A')}")
    st.markdown(f"**Final equity:** {analytics.get('final_equity', 'N/A')}")
    # Equity curve plot
    if 'equity_curve' in analytics:
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()
        ax.plot(analytics['equity_curve'], color=COLOR_ACCENT)
        ax.set_title("Equity Curve")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Equity ($)")
        st.pyplot(fig)
else:
    st.markdown("<div class='emoji-message'>🤖</div>", unsafe_allow_html=True)
    st.markdown("<div class='class-message'>No analytics available yet. Waiting for more signals and outcomes!</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: #888;'>© {datetime.now().year} {PROJECT_NAME} | Powered by AI | <a href='https://github.com/your-repo' style='color: #00FFB0;'>GitHub</a></div>", unsafe_allow_html=True) 