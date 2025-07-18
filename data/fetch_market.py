import yfinance as yf
import pandas as pd
from config import Config
from utils.logger import get_logger
import json
from datetime import datetime
import os
import io
logger = get_logger('fetch_market', log_file='logs/fetch_market.log')

# Central symbol mapping for Yahoo Finance
SYMBOL_MAP = {
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'USDJPY': 'USDJPY=X',
    'USDCHF': 'USDCHF=X',
    'NZDJPY': 'NZDJPY=X',
    'BTCUSD': 'BTC-USD',
    # --- Added major pairs ---
    'AUDUSD': 'AUDUSD=X',
    'NZDUSD': 'NZDUSD=X',
    'USDCAD': 'USDCAD=X',
    'EURJPY': 'EURJPY=X',
    'GBPJPY': 'GBPJPY=X',
    'AUDJPY': 'AUDJPY=X',
}

def get_price_data(pair, interval='1h', lookback=Config.LOOKBACK_PERIOD, ttl_hours=1):
    """
    Fetch historical price data for a given trading pair with TTL cache.
    Args:
        pair (str): Trading pair, e.g., 'USDJPY' or 'BTCUSD'.
        interval (str): Data interval, e.g., '1h'.
        lookback (int): Number of days to look back.
        ttl_hours (int): Time-to-live for cache in hours.
    Returns:
        pd.DataFrame: Price data with OHLCV, or None if unavailable.
    """
    # Force lookback to at least 200 for robust rolling features
    min_lookback = 200
    if lookback < min_lookback:
        logger.info(f"[DEBUG] Increasing lookback from {lookback} to {min_lookback} for {pair}")
        lookback = min_lookback
    logger.info(f"[DEBUG] Using lookback={lookback} for {pair}")
    symbol = SYMBOL_MAP.get(pair)
    if symbol is None:
        logger.error(f"No Yahoo symbol mapping for pair: {pair}")
        return None
    logger.info(f"Fetching data for {pair} using Yahoo symbol: {symbol}")
    # Cache setup
    CACHE_DIR = 'logs/price_cache/'
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = f"{pair}_{interval}_{lookback}"
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    now = datetime.now().timestamp()
    # Try cache first with TTL
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            cache_time = cache_data.get('timestamp', 0)
            age_hours = (now - cache_time) / 3600
            if age_hours < ttl_hours:
                logger.info(f"[price_cache] Cache hit for {cache_key}, age: {age_hours:.2f}h < {ttl_hours}h TTL")
                data = pd.read_json(io.StringIO(cache_data['data']), orient='split')
                return data
            else:
                logger.info(f"[price_cache] Cache expired for {cache_key}, age: {age_hours:.2f}h >= {ttl_hours}h TTL. Fetching fresh data.")
        except Exception as e:
            logger.warning(f"[price_cache] Error reading cache for {cache_key}: {e}")
    try:
        print(f"[DEBUG] About to fetch data for {pair} (symbol: {symbol}) with yfinance...")
        data = yf.download(symbol, period=f'{lookback}d', interval=interval, auto_adjust=True)
        print(f"[DEBUG] Data fetched for {pair}. Type: {type(data)}, Shape: {getattr(data, 'shape', None)}")
        # Flatten MultiIndex columns if present
        if data is not None and not data.empty:
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
            logger.error(f"Data is None for {pair} after fetch.")
            return None
        if hasattr(data, 'empty') and data.empty:
            logger.error(f"Data is empty for {pair} after fetch.")
            return None
        print(f"[DEBUG] Data before dropna for {pair}:\n{data.head(3)}")
        data = data.dropna()
        if data is None or data.empty:
            logger.error(f"Data is None or empty for {pair} after dropna.")
            return None
        if len(data) < 100:
            logger.error(f"Not enough data for {pair}: only {len(data)} rows after cleaning")
            return None
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
        # Save to cache with timestamp
        try:
            cache_data = {
                'timestamp': now,
                'data': data.to_json(orient='split')
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            logger.info(f"[price_cache] Saved price data for {cache_key} to {cache_file}")
        except Exception as e:
            logger.warning(f"[price_cache] Error saving cache for {cache_key}: {e}")
        return data
    except Exception as e:
        print(f"[Error] Failed to fetch price data for {pair} (symbol: {symbol}): {e}")
        # If all fail, try last cache (even if expired)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                logger.info(f"[price_cache] Final fallback cache hit for {cache_key}")
                data = pd.read_json(io.StringIO(cache_data['data']), orient='split')
                return data
            except Exception as e:
                logger.warning(f"[price_cache] Error reading fallback cache for {cache_key}: {e}")
        return None
