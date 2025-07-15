import pandas as pd
from datetime import datetime, timedelta
import os
import yfinance as yf
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.fetch_market import SYMBOL_MAP

SIGNALS_CSV = 'logs/signals.csv'
PRICE_DATA_DIR = 'logs/price_snapshots'  # Directory where price data snapshots are stored (or fetch live)

def fetch_price_data(pair, start_time, end_time, interval='1h'):
    symbol = SYMBOL_MAP.get(pair, pair)
    try:
        data = yf.download(symbol, start=start_time, end=end_time, interval=interval, auto_adjust=True)
        if data is None or not hasattr(data, 'columns'):
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [str(col[-1]) for col in data.columns.values]
        standard_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if len(data.columns) == 5:
            data.columns = standard_cols
        data = data.dropna()
        if data.empty:
            return None
        return data
    except Exception as e:
        print(f"[Error] Failed to fetch price data for {pair} ({symbol}): {e}")
        return None

def update_signal_outcomes():
    if not os.path.exists(SIGNALS_CSV):
        print(f"No signals log found at {SIGNALS_CSV}")
        return
    df = pd.read_csv(SIGNALS_CSV)
    updated = False
    for idx, row in df.iterrows():
        if pd.notnull(row.get('outcome')) and str(row['outcome']).strip() != '':
            continue  # Already resolved
        pair = row['pair']
        time_str = str(row['time'])
        try:
            entry_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        except Exception:
            print(f"Skipping row with invalid time: {time_str}")
            continue
        entry = float(row['entry'])
        sl = float(row['stop_loss'])
        tp1 = float(row['take_profit_1'])
        end_time = entry_time + timedelta(hours=48)
        price_df = fetch_price_data(pair, entry_time, end_time)
        if price_df is None:
            print(f"No price data for {pair} from {entry_time} to {end_time}")
            continue
        if isinstance(price_df, pd.DataFrame) and price_df.empty:
            print(f"No price data for {pair} from {entry_time} to {end_time}")
            continue
        outcome = ''
        for i, p in price_df.iterrows():
            if row['signal'] == 'BUY':
                if p['Low'] <= sl:
                    outcome = 'SL'
                    break
                if p['High'] >= tp1:
                    outcome = 'TP'
                    break
            else:  # SELL
                if p['High'] >= sl:
                    outcome = 'SL'
                    break
                if p['Low'] <= tp1:
                    outcome = 'TP'
                    break
        if outcome:
            df.at[idx, 'outcome'] = outcome
            updated = True
            print(f"Signal {row['signal_id']} outcome: {outcome}")
    if updated:
        df.to_csv(SIGNALS_CSV, index=False)
        print("Signal outcomes updated.")
    else:
        print("No new outcomes found.")

def compute_advanced_analytics(signals_csv=SIGNALS_CSV):
    if not os.path.exists(signals_csv):
        print(f"No signals log found at {signals_csv}")
        return
    df = pd.read_csv(signals_csv)
    if df.empty or 'outcome' not in df.columns:
        print("No outcome data to analyze.")
        return
    # Only analyze signals with resolved outcome
    df = df[df['outcome'].isin(['TP', 'SL'])]
    if df.empty:
        print("No resolved signals to analyze.")
        return
    # --- Basic stats ---
    total = len(df)
    wins = (df['outcome'] == 'TP').sum()
    losses = (df['outcome'] == 'SL').sum()
    win_rate = wins / total if total else 0
    avg_conf = df['confidence'].mean() if 'confidence' in df else None
    # --- R:R and expectancy ---
    if all(col in df.columns for col in ['entry', 'stop_loss', 'take_profit_1']):
        rr_list = []
        for _, row in df.iterrows():
            try:
                entry = float(row['entry'])
                stop_loss = float(row['stop_loss'])
                take_profit_1 = float(row['take_profit_1'])
                risk = abs(entry - stop_loss)
                reward = abs(take_profit_1 - entry)
                rr = reward / risk if risk > 0 else None
            except Exception:
                rr = None
            rr_list.append(rr)
        valid_rr = [x for x in rr_list if x is not None]
        avg_rr = sum(valid_rr) / len(valid_rr) if valid_rr else None
    else:
        avg_rr = None
    expectancy = (win_rate * avg_rr - (1 - win_rate)) if avg_rr is not None else None
    # --- Streaks ---
    streaks = []
    current = 0
    last = None
    for o in df['outcome']:
        if o == last:
            current += 1
        else:
            current = 1
        streaks.append(current)
        last = o
    max_win_streak = max([s for s, o in zip(streaks, df['outcome']) if o == 'TP'], default=0)
    max_loss_streak = max([s for s, o in zip(streaks, df['outcome']) if o == 'SL'], default=0)
    # --- Per-pair stats ---
    per_pair = df.groupby('pair')['outcome'].value_counts().unstack().fillna(0)
    # --- Equity curve simulation ---
    initial_equity = 100.0
    equity = [initial_equity]
    for _, row in df.iterrows():
        last = equity[-1]
        try:
            entry = float(row['entry'])
            stop_loss = float(row['stop_loss'])
            take_profit_1 = float(row['take_profit_1'])
            risk = abs(entry - stop_loss)
            reward = abs(take_profit_1 - entry)
        except Exception:
            risk = 0
            reward = 0
        if row['outcome'] == 'TP':
            pnl = reward
        else:
            pnl = -risk
        equity.append(last + pnl)
    # Drawdown calculation
    equity_series = pd.Series(equity)
    running_max = equity_series.cummax()
    drawdowns = running_max - equity_series
    max_drawdown = drawdowns.max() if not drawdowns.empty else 0
    profit_factor = (wins / losses) if losses else float('inf')
    returns = equity_series.diff().dropna()
    sharpe = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0
    # --- Print summary ---
    print("\n===== ADVANCED ANALYTICS =====")
    print(f"Total signals: {total}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Avg R:R: {avg_rr:.2f}" if avg_rr is not None else "Avg R:R: N/A")
    print(f"Expectancy: {expectancy:.2f}" if expectancy is not None else "Expectancy: N/A")
    print(f"Max win streak: {max_win_streak}")
    print(f"Max loss streak: {max_loss_streak}")
    print(f"Avg confidence: {avg_conf:.2f}" if avg_conf is not None else "Avg confidence: N/A")
    print(f"Max drawdown: {max_drawdown:.2f}")
    print(f"Profit factor: {profit_factor:.2f}")
    print(f"Sharpe ratio: {sharpe:.2f}")
    print(f"Final equity: {equity[-1]:.2f}")
    print("\nPer-pair stats:")
    print(per_pair)
    print("\nEquity curve (first 10):", equity[:10], "... (last 10):", equity[-10:])
    print("============================\n")

if __name__ == "__main__":
    update_signal_outcomes()
    compute_advanced_analytics() 