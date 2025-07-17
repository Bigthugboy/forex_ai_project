try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import pandas as pd
from data.preprocess import preprocess_features
from models.train_model import train_signal_model
from models.predict import predict_signal
from signals.signal_generator import generate_signal_output
from config import Config
from data.fetch_market import get_price_data, SYMBOL_MAP
from data.fetch_news import get_news_sentiment
from datetime import datetime, timedelta
import numpy as np
from utils.logger import get_logger
import os

logger = get_logger('backtest')

PAIRS = ['USDJPY', 'BTCUSD', 'USDCHF', 'NZDJPY']

# --- Enhanced Backtest Parameters ---
DEFAULTS = {
    'slippage_pct': {
        'FX': 0.0001,   # 0.01%
        'CRYPTO': 0.0005  # 0.05%
    },
    'spread': {
        'USDJPY': 0.02, 'USDCHF': 0.02, 'NZDJPY': 0.03,  # in price units
        'BTCJPY': 50.0, 'BTCUSD': 2.0
    },
    'commission': {
        'FX': 3.0,      # $3 per trade
        'CRYPTO': 0.001 # 0.1% per trade
    },
    'initial_capital': 50.0,  # Changed from 10000.0 to 50.0
    'risk_per_trade': 0.01  # 1% of capital per trade
}

EQUITY_CURVE_CSV = 'logs/equity_curve.csv'
SIGNALS_CSV = 'logs/signals.csv'

def get_asset_type(pair):
    if 'BTC' in pair:
        return 'CRYPTO'
    return 'FX'

def backtest_pair(pair, lookback=120):
    logger.info(f"Backtesting {pair}...")
    price_df = get_price_data(pair, interval='1h', lookback=lookback)
    if price_df is None or price_df.empty:
        logger.warning(f"No price data for {pair}. Skipping backtest for this pair.")
        return pd.DataFrame(), [], []
    today = datetime.utcnow().date()
    from_date = (today - timedelta(days=2)).strftime('%Y-%m-%d')
    to_date = today.strftime('%Y-%m-%d')
    sentiment = get_news_sentiment(Config.NEWS_KEYWORDS, from_date, to_date)
    features_df = preprocess_features(price_df, sentiment)
    model, scaler, feature_cols = train_signal_model(features_df)

    # Enhanced backtest params
    asset_type = get_asset_type(pair)
    slippage_pct = DEFAULTS['slippage_pct'][asset_type]
    spread = DEFAULTS['spread'].get(pair, 0.02)
    commission = DEFAULTS['commission'][asset_type]
    initial_capital = DEFAULTS['initial_capital']
    risk_per_trade = DEFAULTS['risk_per_trade']
    capital = initial_capital
    equity_curve = [capital]
    max_equity = capital
    drawdowns = []

    results = []
    for i in range(60, len(features_df) - 1):  # start after indicators warm up
        try:
            test_df = features_df.iloc[:i+1]
            pred = predict_signal(test_df)
            if not pred:
                continue
            signal = generate_signal_output(pair, test_df, pred)
            next_close = features_df.iloc[i+1][[col for col in features_df.columns if col.startswith('Close')][0]]
            entry = signal['entry']
            # --- Slippage ---
            slippage = entry * slippage_pct * np.random.choice([-1, 1])
            entry_adj = entry + slippage
            # --- Spread ---
            if signal['signal'] == 'BUY':
                entry_adj += spread / 2
            else:
                entry_adj -= spread / 2
            # --- Commission ---
            if asset_type == 'FX':
                commission_cost = commission
            else:
                commission_cost = entry_adj * commission  # percent of notional
            # --- Position Sizing ---
            position_size = capital * risk_per_trade / abs(signal['sl'] - entry_adj) if abs(signal['sl'] - entry_adj) > 0 else 1.0
            # --- Profit Calculation ---
            if signal['signal'] == 'BUY':
                profit = (next_close - entry_adj) * position_size - commission_cost
            else:
                profit = (entry_adj - next_close) * position_size - commission_cost
            capital += profit
            equity_curve.append(capital)
            max_equity = max(max_equity, capital)
            drawdown = max_equity - capital
            drawdowns.append(drawdown)
            results.append({
                'time': features_df.index[i],
                'signal': signal['signal'],
                'entry': entry,
                'entry_adj': entry_adj,
                'next_close': next_close,
                'profit': profit,
                'confidence': signal['confidence'],
                'slippage': slippage,
                'spread': spread,
                'commission': commission_cost,
                'position_size': position_size,
                'capital': capital,
                'drawdown': drawdown
            })
            logger.info(f"Trade {i}: {signal['signal']} Entry: {entry_adj:.5f} Profit: {profit:.2f} Capital: {capital:.2f}")
        except Exception as e:
            logger.error(f"Error in trade simulation at index {i}: {e}", exc_info=True)
    return pd.DataFrame(results), equity_curve, drawdowns

def compute_streaks(series):
    max_win_streak = max_loss_streak = 0
    current_win = current_loss = 0
    for val in series:
        if val > 0:
            current_win += 1
            current_loss = 0
        elif val < 0:
            current_loss += 1
            current_win = 0
        else:
            current_win = current_loss = 0
        max_win_streak = max(max_win_streak, current_win)
        max_loss_streak = max(max_loss_streak, current_loss)
    return max_win_streak, max_loss_streak

def summarize_backtest(df, equity_curve, drawdowns, pair=None):
    logger.info(f'Summarizing backtest for {pair if pair else "all pairs"}...')
    if df.empty:
        logger.warning("No trades generated.")
        return
    total_trades = len(df)
    wins = (df['profit'] > 0).sum()
    losses = (df['profit'] <= 0).sum()
    win_rate = wins / total_trades if total_trades else 0
    profit_factor = df[df['profit'] > 0]['profit'].sum() / abs(df[df['profit'] <= 0]['profit'].sum()) if losses else float('inf')
    max_drawdown = max(drawdowns) if drawdowns else 0
    avg_trade = df['profit'].mean()
    avg_win = df[df['profit'] > 0]['profit'].mean() if wins else 0
    avg_loss = df[df['profit'] <= 0]['profit'].mean() if losses else 0
    sharpe = (df['profit'].mean() / df['profit'].std()) * np.sqrt(252) if df['profit'].std() > 0 else 0
    max_win_streak, max_loss_streak = compute_streaks(df['profit'])
    logger.info(f"Summary for {pair if pair else ''}: Total trades: {total_trades}, Win rate: {win_rate:.2%}, Profit factor: {profit_factor:.2f}, Max drawdown: {max_drawdown:.2f}, Final capital: {df['capital'].iloc[-1]:.2f}, Sharpe: {sharpe:.2f}")
    logger.info(f"Avg trade: {avg_trade:.2f}, Avg win: {avg_win:.2f}, Avg loss: {avg_loss:.2f}, Longest win streak: {max_win_streak}, Longest loss streak: {max_loss_streak}")
    print(f"Total trades: {total_trades}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Profit factor: {profit_factor:.2f}")
    print(f"Max drawdown: {max_drawdown:.2f}")
    print(f"Total PnL: {df['profit'].sum():.2f}")
    print(f"Final capital: {df['capital'].iloc[-1]:.2f}")
    print(f"Sharpe ratio: {sharpe:.2f}")
    print(f"Avg trade: {avg_trade:.2f}, Avg win: {avg_win:.2f}, Avg loss: {avg_loss:.2f}")
    print(f"Longest win streak: {max_win_streak}, Longest loss streak: {max_loss_streak}")
    logger.info(f"Equity curve: {equity_curve[:5]} ... {equity_curve[-5:]}")
    logger.info(f"Sample trade costs (slippage, spread, commission):")
    logger.info(df[['slippage', 'spread', 'commission']].head())
    # --- Plot equity curve and drawdown ---
    if plt:
        plt.figure(figsize=(10, 5))
        plt.plot(equity_curve, label='Equity Curve')
        plt.title(f'Equity Curve - {pair}' if pair else 'Equity Curve')
        plt.xlabel('Trade #')
        plt.ylabel('Account Balance')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if pair:
            plt.savefig(f'equity_curve_{pair}.png')
            print(f"Equity curve plot saved as equity_curve_{pair}.png")
        else:
            plt.show()
    # Drawdown plot
    if plt:
        plt.figure(figsize=(10, 3))
        plt.plot(drawdowns, color='red', label='Drawdown')
        plt.title(f'Drawdown - {pair}' if pair else 'Drawdown')
        plt.xlabel('Trade #')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if pair:
            plt.savefig(f'drawdown_{pair}.png')
            print(f"Drawdown plot saved as drawdown_{pair}.png")
        else:
            plt.show()


def main():
    logger.info("Starting backtest process...")
    for pair in PAIRS:
        logger.info(f"Processing pair: {pair}")
        df, equity_curve, drawdowns = backtest_pair(pair, lookback=120)
        summarize_backtest(df, equity_curve, drawdowns, pair=pair)
        # --- Save equity curve ---
        eq_df = pd.DataFrame({'equity': equity_curve})
        os.makedirs(os.path.dirname(EQUITY_CURVE_CSV), exist_ok=True)
        eq_df.to_csv(EQUITY_CURVE_CSV, index=False)
        # --- Save trades/signals ---
        if not df.empty:
            df_out = df.copy()
            df_out['source'] = 'backtest'
            df_out.to_csv(SIGNALS_CSV, index=False)
    logger.info("Backtest process finished.")

if __name__ == "__main__":
    main() 