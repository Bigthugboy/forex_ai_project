import pandas as pd
from datetime import datetime, timedelta
import os
import yfinance as yf
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.fetch_market import SYMBOL_MAP
import json
from utils.logger import get_logger
logger = get_logger('outcome_tracker', log_file='logs/outcome_tracker.log')

SIGNALS_CSV = 'logs/signals.csv'
PRICE_DATA_DIR = 'logs/price_snapshots'  # Directory where price data snapshots are stored (or fetch live)

def fetch_price_data(pair, start_time, end_time, interval='1h'):
    logger.info(f'Fetching price data for {pair} from {start_time} to {end_time}...')
    symbol = SYMBOL_MAP.get(pair, pair)
    try:
        data = yf.download(symbol, start=start_time, end=end_time, interval=interval, auto_adjust=True)
        if data is None or not hasattr(data, 'columns'):
            logger.warning(f'No data columns for {pair}')
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [str(col[-1]) for col in data.columns.values]
        standard_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if len(data.columns) == 5:
            data.columns = standard_cols
        data = data.dropna()
        if data.empty:
            logger.warning(f'Fetched data is empty for {pair}')
            return None
        logger.info(f'Fetched {len(data)} rows for {pair}')
        return data
    except Exception as e:
        logger.error(f"[Error] Failed to fetch price data for {pair} ({symbol}): {e}")
        return None

def safe_float(val, fallback):
    try:
        if pd.isna(val):
            return fallback
        return float(val)
    except Exception:
        return fallback

def update_signal_outcomes():
    logger.info('Updating signal outcomes...')
    if not os.path.exists(SIGNALS_CSV):
        logger.warning(f"No signals log found at {SIGNALS_CSV}")
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
            logger.warning(f"Skipping row with invalid time: {time_str}")
            continue
        entry = float(row['entry'])
        sl = float(row['stop_loss'])
        tp1 = float(row['take_profit_1'])
        tp2 = safe_float(row.get('take_profit_2'), tp1)
        tp3 = safe_float(row.get('take_profit_3'), tp2)
        end_time = entry_time + timedelta(hours=48)
        price_df = fetch_price_data(pair, entry_time, end_time)
        if price_df is None:
            logger.warning(f"No price data for {pair} from {entry_time} to {end_time}")
            continue
        if price_df is not None and hasattr(price_df, "empty") and price_df.empty:
            logger.warning(f"No price data for {pair} from {entry_time} to {end_time}")
            continue
        outcome = ''
        for i, p in price_df.iterrows():
            if row['signal'] == 'BUY':
                if p['Low'] <= sl:
                    outcome = 'SL'
                    break
                if p['High'] >= tp3:
                    outcome = 'TP3'
                    break
                if p['High'] >= tp2:
                    outcome = 'TP2'
                    break
                if p['High'] >= tp1:
                    outcome = 'TP1'
                    break
            else:  # SELL
                if p['High'] >= sl:
                    outcome = 'SL'
                    break
                if p['Low'] <= tp3:
                    outcome = 'TP3'
                    break
                if p['Low'] <= tp2:
                    outcome = 'TP2'
                    break
                if p['Low'] <= tp1:
                    outcome = 'TP1'
                    break
        if outcome:
            df.at[idx, 'outcome'] = outcome
            updated = True
            if 'signal_id' in df.columns:
                logger.info(f"Signal {row['signal_id']} outcome: {outcome}")
            else:
                logger.info(f"Signal {row['time']} {row['pair']} outcome: {outcome}")
    if updated:
        df.to_csv(SIGNALS_CSV, index=False)
        logger.info("Signal outcomes updated.")
    else:
        logger.info("No new outcomes found.")

def compute_advanced_analytics(signals_csv=SIGNALS_CSV, export_json_path='logs/analytics_summary.json'):
    logger.info('Computing advanced analytics...')
    if not os.path.exists(signals_csv):
        logger.warning(f"No signals log found at {signals_csv}")
        return
    df = pd.read_csv(signals_csv)
    if df.empty or 'outcome' not in df.columns:
        logger.warning("No outcome data to analyze.")
        return
    # Only analyze signals with resolved outcome
    df = df[df['outcome'].isin(['TP', 'SL'])]
    if df.empty:
        logger.warning("No resolved signals to analyze.")
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
    # Convert per_pair DataFrame to a serializable dict
    per_pair_dict = {str(idx): row.dropna().to_dict() for idx, row in per_pair.iterrows()}
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
    logger.info("\n===== ADVANCED ANALYTICS =====")
    logger.info(f"Total signals: {total}")
    logger.info(f"Win rate: {win_rate:.2%}")
    logger.info(f"Avg R:R: {avg_rr:.2f}" if avg_rr is not None else "Avg R:R: N/A")
    logger.info(f"Expectancy: {expectancy:.2f}" if expectancy is not None else "Expectancy: N/A")
    logger.info(f"Max win streak: {max_win_streak}")
    logger.info(f"Max loss streak: {max_loss_streak}")
    logger.info(f"Avg confidence: {avg_conf:.2f}" if avg_conf is not None else "Avg confidence: N/A")
    logger.info(f"Max drawdown: {max_drawdown:.2f}")
    logger.info(f"Profit factor: {profit_factor:.2f}")
    logger.info(f"Sharpe ratio: {sharpe:.2f}")
    logger.info(f"Final equity: {equity[-1]:.2f}")
    logger.info("\nPer-pair stats:")
    logger.info(per_pair)
    logger.info("\nEquity curve (first 10):", equity[:10], "... (last 10):", equity[-10:])
    logger.info("============================\n")
    # --- Export to JSON ---
    analytics = {
        'total_signals': total,
        'win_rate': win_rate,
        'avg_rr': avg_rr,
        'expectancy': expectancy,
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,
        'avg_confidence': avg_conf,
        'max_drawdown': float(max_drawdown),
        'profit_factor': float(profit_factor),
        'sharpe_ratio': float(sharpe),
        'final_equity': float(equity[-1]),
        'per_pair_stats': per_pair_dict,
        'equity_curve': [float(x) for x in equity],
    }
    os.makedirs(os.path.dirname(export_json_path), exist_ok=True)
    with open(export_json_path, 'w') as f:
        json.dump(analytics, f, indent=2)
    logger.info('Advanced analytics computation complete.')

if __name__ == "__main__":
    update_signal_outcomes()
    compute_advanced_analytics() 