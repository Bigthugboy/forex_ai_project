import yfinance as yf
import pandas as pd
from config import Config

# Central symbol mapping for Yahoo Finance
SYMBOL_MAP = {
    'USDJPY': 'USDJPY=X',
    'USDCHF': 'USDCHF=X',
    'JPYNZD': 'NZDJPY=X',  # Yahoo does not have JPYNZD, use NZDJPY as proxy
    'BTCUSD': 'BTC-USD',
}

def get_price_data(pair, interval='1h', lookback=Config.LOOKBACK_PERIOD):
    """
    Fetch historical price data for a given trading pair.
    Args:
        pair (str): Trading pair, e.g., 'USDJPY' or 'BTCUSD'.
        interval (str): Data interval, e.g., '1h'.
        lookback (int): Number of days to look back.
    Returns:
        pd.DataFrame: Price data with OHLCV, or None if unavailable.
    """
    # Force lookback to at least 90 for robust rolling features
    min_lookback = 90
    if lookback < min_lookback:
        print(f"[DEBUG] Increasing lookback from {lookback} to {min_lookback} for {pair}")
        lookback = min_lookback
    print(f"[DEBUG] Using lookback={lookback} for {pair}")
    symbol = SYMBOL_MAP.get(pair, pair)
    try:
        print(f"[DEBUG] About to fetch data for {pair} (symbol: {symbol}) with yfinance...")
        data = yf.download(symbol, period=f'{lookback}d', interval=interval, auto_adjust=True)
        print(f"[DEBUG] Data fetched for {pair}. Type: {type(data)}, Shape: {getattr(data, 'shape', None)}")
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [str(col[-1]) for col in data.columns.values]
            print(f"[DEBUG] Flattened columns: {data.columns}")
        # Rename columns to standard OHLCV if needed
        standard_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if len(data.columns) == 5:
            data.columns = standard_cols
            print(f"[DEBUG] Renamed columns to standard OHLCV: {data.columns}")
        else:
            print(f"[ERROR] Unexpected number of columns after flattening: {data.columns}")
        if data is None:
            print(f"[DEBUG] Data is None for {pair}.")
            return None
        if hasattr(data, 'empty') and data.empty:
            print(f"[DEBUG] Data is empty for {pair}.")
            return None
        print(f"[DEBUG] Data before dropna for {pair}:\n{data.head(3)}")
        data = data.dropna()
        print(f"[DEBUG] Data after dropna for {pair}:\n{data.head(3)}")
        # --- Check for required columns ---
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            print(f"[DEBUG] Checking column '{col}' in data for {pair}...")
            if col not in data.columns:
                print(f"[ERROR] Required price column missing: {col} for {pair} (symbol: {symbol})")
                raise ValueError(f"[ERROR] Required price column missing: {col} for {pair} (symbol: {symbol})")
            n_nans = data[col].isnull().sum()
            print(f"[DEBUG] NaN count for column '{col}' in {pair}: {n_nans}")
            if n_nans > 0:
                print(f"[ERROR] NaN values found in price column: {col} for {pair} (symbol: {symbol})")
                raise ValueError(f"[ERROR] NaN values found in price column: {col} for {pair} (symbol: {symbol})")
        print(f"[DEBUG] Price data for {pair} (symbol: {symbol}), first 3 rows:")
        print(data.head(3))
        return data
    except Exception as e:
        print(f"[Error] Failed to fetch price data for {pair} (symbol: {symbol}): {e}")
        return None
