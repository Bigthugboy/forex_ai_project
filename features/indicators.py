# features/indicators.py

import pandas as pd
import numpy as np
from ta.trend import MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
# import ta.volume if needed
import pandas_ta as pta

class Indicator:
    def __init__(self, name, func, description=""):
        self.name = name
        self.func = func
        self.description = description

    def compute(self, df):
        return self.func(df)

# --- Indicator implementations ---
def atr_14(df):
    """Calculate 14-period Average True Range (ATR)."""
    high_col = [col for col in df.columns if col.startswith('High')][0]
    low_col = [col for col in df.columns if col.startswith('Low')][0]
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    if len(df) >= 14:
        return AverageTrueRange(high=df[high_col], low=df[low_col], close=df[close_col], window=14).average_true_range()
    else:
        return pd.Series([np.nan]*len(df), index=df.index)

def volatility_20(df):
    """Calculate 20-period rolling standard deviation (volatility)."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    if len(df) >= 20:
        return df[close_col].rolling(window=20).std()
    else:
        return pd.Series([np.nan]*len(df), index=df.index)

def sma_20(df):
    """Calculate 20-period Simple Moving Average (SMA)."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    return df[close_col].rolling(window=20).mean()

def sma_50(df):
    """Calculate 50-period Simple Moving Average (SMA)."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    return df[close_col].rolling(window=50).mean()

def sma_200(df):
    """Calculate 200-period Simple Moving Average (SMA)."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    return df[close_col].rolling(window=200).mean()

def ema_12(df):
    """Calculate 12-period Exponential Moving Average (EMA)."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    return df[close_col].ewm(span=12, adjust=False).mean()

def ema_26(df):
    """Calculate 26-period Exponential Moving Average (EMA)."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    return df[close_col].ewm(span=26, adjust=False).mean()

def rsi_14(df):
    """Calculate 14-period Relative Strength Index (RSI)."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    return RSIIndicator(close=df[close_col], window=14).rsi()

def macd(df):
    """Calculate MACD indicator."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    return MACD(close=df[close_col]).macd()

def macd_signal(df):
    """Calculate MACD signal line."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    return MACD(close=df[close_col]).macd_signal()

def bb_high(df):
    """Calculate Bollinger Bands high."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    return BollingerBands(close=df[close_col], window=20, window_dev=2).bollinger_hband()

def bb_low(df):
    """Calculate Bollinger Bands low."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    return BollingerBands(close=df[close_col], window=20, window_dev=2).bollinger_lband()

def stoch_k(df):
    """Calculate Stochastic %K."""
    high_col = [col for col in df.columns if col.startswith('High')][0]
    low_col = [col for col in df.columns if col.startswith('Low')][0]
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    return StochasticOscillator(high=df[high_col], low=df[low_col], close=df[close_col], window=14, smooth_window=3).stoch()

def stoch_d(df):
    """Calculate Stochastic %D."""
    high_col = [col for col in df.columns if col.startswith('High')][0]
    low_col = [col for col in df.columns if col.startswith('Low')][0]
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    return StochasticOscillator(high=df[high_col], low=df[low_col], close=df[close_col], window=14, smooth_window=3).stoch_signal()

def fib_levels(df):
    """Calculate Fibonacci retracement levels and distances to close."""
    high_col = [col for col in df.columns if col.startswith('High')][0]
    low_col = [col for col in df.columns if col.startswith('Low')][0]
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    N = 20
    rolling_high = df[high_col].rolling(window=N).max()
    rolling_low = df[low_col].rolling(window=N).min()
    fibs = {
        'fib_0': rolling_high,
        'fib_236': rolling_high - (rolling_high - rolling_low) * 0.236,
        'fib_382': rolling_high - (rolling_high - rolling_low) * 0.382,
        'fib_500': rolling_high - (rolling_high - rolling_low) * 0.5,
        'fib_618': rolling_high - (rolling_high - rolling_low) * 0.618,
        'fib_786': rolling_high - (rolling_high - rolling_low) * 0.786,
        'fib_100': rolling_low,
    }
    for level in fibs:
        df[level] = fibs[level]
        df[f'dist_to_{level}'] = df[close_col] - df[level]
    return df

def trendline_slope_high(df):
    """Calculate rolling slope of highs as a proxy for trendline."""
    high_col = [col for col in df.columns if col.startswith('High')][0]
    N = 20
    return df[high_col].rolling(window=N).apply(lambda x: (x[-1] - x[0]) / N, raw=True)

def trendline_slope_low(df):
    """Calculate rolling slope of lows as a proxy for trendline."""
    low_col = [col for col in df.columns if col.startswith('Low')][0]
    N = 20
    return df[low_col].rolling(window=N).apply(lambda x: (x[-1] - x[0]) / N, raw=True)

def trendline_up(df):
    """Binary indicator if trendline slope of highs is positive."""
    slope = trendline_slope_high(df)
    return (slope > 0).astype(int)

def trendline_down(df):
    """Binary indicator if trendline slope of lows is negative."""
    slope = trendline_slope_low(df)
    return (slope < 0).astype(int)

def breakout_high(df):
    """Binary indicator if close breaks above recent high."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    high_col = [col for col in df.columns if col.startswith('High')][0]
    N = 20
    high_rolling_max = pd.Series(df[high_col], index=df.index).rolling(window=N, min_periods=1).max()
    return (df[close_col] > high_rolling_max.shift(1)).astype(int)

def breakout_low(df):
    """Binary indicator if close breaks below recent low."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    low_col = [col for col in df.columns if col.startswith('Low')][0]
    N = 20
    low_rolling_min = pd.Series(df[low_col], index=df.index).rolling(window=N, min_periods=1).min()
    return (df[close_col] < low_rolling_min.shift(1)).astype(int)

def dist_to_high(df):
    """Distance from close to recent high."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    high_col = [col for col in df.columns if col.startswith('High')][0]
    N = 20
    high_rolling_max = pd.Series(df[high_col], index=df.index).rolling(window=N, min_periods=1).max()
    return df[close_col] - high_rolling_max.shift(1)

def dist_to_low(df):
    """Distance from close to recent low."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    low_col = [col for col in df.columns if col.startswith('Low')][0]
    N = 20
    low_rolling_min = pd.Series(df[low_col], index=df.index).rolling(window=N, min_periods=1).min()
    return df[close_col] - low_rolling_min.shift(1)

def vol_spike_atr(df):
    """Binary indicator for ATR volatility spike."""
    if 'atr_14' in df.columns:
        N = 20
        return (df['atr_14'] > pd.Series(df['atr_14']).rolling(window=N).mean().shift(1) * 1.5).astype(int)
    else:
        return pd.Series([0]*len(df), index=df.index)

def vol_spike_std(df):
    """Binary indicator for standard deviation volatility spike."""
    if 'volatility_20' in df.columns:
        N = 20
        return (df['volatility_20'] > pd.Series(df['volatility_20']).rolling(window=N).mean().shift(1) * 1.5).astype(int)
    else:
        return pd.Series([0]*len(df), index=df.index)

def is_trending(df):
    """Binary indicator if price is trending."""
    if 'breakout_high' in df.columns and 'breakout_low' in df.columns:
        return ((df['breakout_high'] == 1) | (df['breakout_low'] == 1)).astype(int)
    else:
        return pd.Series([0]*len(df), index=df.index)

def is_ranging(df):
    """Binary indicator if price is ranging."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    high_col = [col for col in df.columns if col.startswith('High')][0]
    low_col = [col for col in df.columns if col.startswith('Low')][0]
    N = 20
    dist_to_high = df[close_col] - pd.Series(df[high_col], index=df.index).rolling(window=N, min_periods=1).max().shift(1)
    dist_to_low = df[close_col] - pd.Series(df[low_col], index=df.index).rolling(window=N, min_periods=1).min().shift(1)
    avg_range = (df[high_col] - df[low_col]).rolling(window=N).mean() * 0.2
    return ((dist_to_high.abs() < avg_range) & (dist_to_low.abs() < avg_range)).astype(int)

def wyckoff_phases(df):
    """Detect Wyckoff phases and one-hot encode them."""
    # This is a stub; actual logic should be migrated from detect_wyckoff_phase
    return pd.DataFrame()

def supply_zone(df):
    """Detect supply (resistance) zones."""
    high_col = [col for col in df.columns if col.startswith('High')][0]
    window = 20
    threshold = 0.002
    highs = df[high_col].rolling(window=window, min_periods=1).max()
    return (np.abs(df[high_col] - highs) < threshold * highs).astype(int)

def demand_zone(df):
    """Detect demand (support) zones."""
    low_col = [col for col in df.columns if col.startswith('Low')][0]
    window = 20
    threshold = 0.002
    lows = df[low_col].rolling(window=window, min_periods=1).min()
    return (np.abs(df[low_col] - lows) < threshold * lows).astype(int)

def price_vs_sma20_4h(df):
    """Distance of price from 20-period SMA on 4h timeframe."""
    # This is a stub; actual logic should be implemented for multi-timeframe
    return pd.Series([0]*len(df), index=df.index)

def price_vs_sma50_4h(df):
    """Distance of price from 50-period SMA on 4h timeframe."""
    # This is a stub; actual logic should be implemented for multi-timeframe
    return pd.Series([0]*len(df), index=df.index)

def trend_strength_4h(df):
    """Trend strength indicator on 4h timeframe."""
    # This is a stub; actual logic should be implemented for multi-timeframe
    return pd.Series([0]*len(df), index=df.index)

def news_sentiment(df):
    """News sentiment score as a feature."""
    if 'news_sentiment' in df.columns:
        return df['news_sentiment']
    else:
        return pd.Series([0]*len(df), index=df.index)

# --- Registry ---
INDICATORS = [
    Indicator('atr_14', atr_14, 'Average True Range (14)'),
    Indicator('volatility_20', volatility_20, '20-period rolling standard deviation'),
    Indicator('sma_20', sma_20, '20-period Simple Moving Average'),
    Indicator('sma_50', sma_50, '50-period Simple Moving Average'),
    Indicator('sma_200', sma_200, '200-period Simple Moving Average'),
    Indicator('ema_12', ema_12, '12-period Exponential Moving Average'),
    Indicator('ema_26', ema_26, '26-period Exponential Moving Average'),
    Indicator('rsi_14', rsi_14, '14-period Relative Strength Index'),
    Indicator('macd', macd, 'MACD indicator'),
    Indicator('macd_signal', macd_signal, 'MACD signal line'),
    Indicator('bb_high', bb_high, 'Bollinger Bands high'),
    Indicator('bb_low', bb_low, 'Bollinger Bands low'),
    Indicator('stoch_k', stoch_k, 'Stochastic %K'),
    Indicator('stoch_d', stoch_d, 'Stochastic %D'),
    Indicator('fib_levels', fib_levels, 'Fibonacci retracement levels and distances'),
    Indicator('trendline_slope_high', trendline_slope_high, 'Rolling slope of highs'),
    Indicator('trendline_slope_low', trendline_slope_low, 'Rolling slope of lows'),
    Indicator('trendline_up', trendline_up, 'Trendline up binary'),
    Indicator('trendline_down', trendline_down, 'Trendline down binary'),
    Indicator('breakout_high', breakout_high, 'Breakout above high'),
    Indicator('breakout_low', breakout_low, 'Breakout below low'),
    Indicator('dist_to_high', dist_to_high, 'Distance to high'),
    Indicator('dist_to_low', dist_to_low, 'Distance to low'),
    Indicator('vol_spike_atr', vol_spike_atr, 'ATR volatility spike'),
    Indicator('vol_spike_std', vol_spike_std, 'Std dev volatility spike'),
    Indicator('is_trending', is_trending, 'Trending binary'),
    Indicator('is_ranging', is_ranging, 'Ranging binary'),
    Indicator('wyckoff_phases', wyckoff_phases, 'Wyckoff phases'),
    Indicator('supply_zone', supply_zone, 'Supply zone'),
    Indicator('demand_zone', demand_zone, 'Demand zone'),
    Indicator('price_vs_sma20_4h', price_vs_sma20_4h, 'Price vs SMA20 4h'),
    Indicator('price_vs_sma50_4h', price_vs_sma50_4h, 'Price vs SMA50 4h'),
    Indicator('trend_strength_4h', trend_strength_4h, 'Trend strength 4h'),
    Indicator('news_sentiment', news_sentiment, 'News sentiment score'),
] 