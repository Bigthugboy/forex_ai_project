import yfinance as yf
import pandas as pd
from config import Config

def get_price_data(pair, interval='1h', lookback=Config.LOOKBACK_PERIOD):
    """
    Fetch historical price data for a given trading pair.
    Args:
        pair (str): Trading pair, e.g., 'USDJPY' or 'BTCUSD'.
        interval (str): Data interval, e.g., '1h'.
        lookback (int): Number of days to look back.
    Returns:
        pd.DataFrame: Price data with OHLCV.
    """
    symbol = pair + '=X' if pair in ['USDJPY'] else pair + '-USD'
    data = yf.download(symbol, period=f'{lookback}d', interval=interval)
    data = data.dropna()
    return data
