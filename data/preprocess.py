import pandas as pd
import ta.trend
import ta.momentum
import ta.volatility
import ta.volume
from candlestick import candlestick

import unittest
import numpy as np

# --- Advanced Data Cleaning Helpers ---
def clean_missing_and_outliers(df):
    """
    Impute missing values and handle outliers in the DataFrame.
    - Forward fill, then mean imputation for missing values.
    - Outliers (z-score > 3) replaced with median.
    Logs actions taken.
    """
    df_clean = df.copy()
    # Impute missing values
    n_missing_before = df_clean.isnull().sum().sum()
    if n_missing_before > 0:
        print(f"[Data Cleaning] Missing values before: {n_missing_before}")
        df_clean = df_clean.ffill().bfill()
        n_missing_after_ffill = df_clean.isnull().sum().sum()
        if n_missing_after_ffill > 0:
            for col in df_clean.columns:
                if df_clean[col].isnull().any():
                    mean_val = df_clean[col].mean()
                    df_clean[col] = df_clean[col].fillna(mean_val)
        n_missing_after = df_clean.isnull().sum().sum()
        print(f"[Data Cleaning] Missing values after imputation: {n_missing_after}")
    # Outlier detection (z-score method)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    z_scores = np.abs((df_clean[numeric_cols] - df_clean[numeric_cols].mean()) / df_clean[numeric_cols].std(ddof=0))
    outlier_mask = (z_scores > 3)
    n_outliers = outlier_mask.sum().sum()
    if n_outliers > 0:
        print(f"[Data Cleaning] Outliers detected: {n_outliers}")
        for col in numeric_cols:
            median_val = df_clean[col].median()
            df_clean.loc[outlier_mask[col], col] = median_val
        print(f"[Data Cleaning] Outliers replaced with median.")
    return df_clean

def detect_candlestick_patterns(df, open_col, close_col, high_col, low_col):
    """
    Detect common candlestick patterns and add as binary features.
    Patterns: doji, hammer, bullish engulfing, bearish engulfing, shooting star
    """
    df = df.copy()
    # Doji: Open and Close are very close
    df['doji'] = (abs(df[open_col] - df[close_col]) <= (df[high_col] - df[low_col]) * 0.1).astype(int)
    # Hammer: Small body, long lower shadow
    df['hammer'] = (
        (df[close_col] > df[open_col]) &
        ((df[open_col] - df[low_col]) > 2 * abs(df[close_col] - df[open_col])) &
        ((df[high_col] - df[close_col]) < abs(df[close_col] - df[open_col]))
    ).astype(int)
    # Bullish Engulfing: Current body engulfs previous body, bullish
    df['bullish_engulfing'] = (
        (df[close_col].shift(1) < df[open_col].shift(1)) &
        (df[close_col] > df[open_col]) &
        (df[close_col] > df[open_col].shift(1)) &
        (df[open_col] < df[close_col].shift(1))
    ).astype(int)
    # Bearish Engulfing: Current body engulfs previous body, bearish
    df['bearish_engulfing'] = (
        (df[close_col].shift(1) > df[open_col].shift(1)) &
        (df[close_col] < df[open_col]) &
        (df[open_col] > df[close_col].shift(1)) &
        (df[close_col] < df[open_col].shift(1))
    ).astype(int)
    # Shooting Star: Small body, long upper shadow
    df['shooting_star'] = (
        (df[open_col] > df[close_col]) &
        ((df[high_col] - df[open_col]) > 2 * abs(df[close_col] - df[open_col])) &
        ((df[open_col] - df[low_col]) < abs(df[close_col] - df[open_col]))
    ).astype(int)
    return df

def preprocess_features(price_df, sentiment_score):
    """
    Add technical indicators and news sentiment to the price DataFrame.
    Args:
        price_df (pd.DataFrame): Price data with OHLCV.
        sentiment_score (float): News sentiment score.
    Returns:
        pd.DataFrame: DataFrame with features.
    """
    # Flatten MultiIndex columns if present
    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = ['_'.join([str(i) for i in col if i]) for col in price_df.columns.values]
    df = price_df.copy()
    # --- Advanced Data Cleaning ---
    df = clean_missing_and_outliers(df)
    # Use the correct column names for the pair
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    open_col = [col for col in df.columns if col.startswith('Open')][0]
    high_col = [col for col in df.columns if col.startswith('High')][0]
    low_col = [col for col in df.columns if col.startswith('Low')][0]
    volume_col = [col for col in df.columns if col.startswith('Volume')][0]
    
    # SMA
    df['sma_20'] = ta.trend.sma_indicator(close=df[close_col], window=20)
    df['sma_50'] = ta.trend.sma_indicator(close=df[close_col], window=50)
    df['sma_200'] = ta.trend.sma_indicator(close=df[close_col], window=200)
    
    # EMA
    df['ema_12'] = ta.trend.ema_indicator(close=df[close_col], window=12)
    df['ema_26'] = ta.trend.ema_indicator(close=df[close_col], window=26)
    
    # RSI
    df['rsi_14'] = ta.momentum.rsi(close=df[close_col], window=14)
    
    # MACD
    macd = ta.trend.macd(close=df[close_col], window_slow=26, window_fast=12)
    macd_signal = ta.trend.macd_signal(close=df[close_col], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df[close_col], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high=df[high_col], low=df[low_col], close=df[close_col], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # ATR
    df['atr_14'] = ta.volatility.average_true_range(high=df[high_col], low=df[low_col], close=df[close_col], window=14)
    
    # Advanced indicators
    df['adx_14'] = ta.trend.adx(high=df[high_col], low=df[low_col], close=df[close_col], window=14)
    df['cci_20'] = ta.trend.cci(high=df[high_col], low=df[low_col], close=df[close_col], window=20)
    df['obv'] = ta.volume.on_balance_volume(close=df[close_col], volume=df[volume_col])
    df['williams_r'] = ta.momentum.williams_r(high=df[high_col], low=df[low_col], close=df[close_col], lbp=14)
    df['volatility_20'] = df[close_col].rolling(window=20).std()
    
    # Add news sentiment as a feature (same value for all rows in this window)
    df['news_sentiment'] = sentiment_score
    
    # Add candlestick pattern features (update to use correct columns)
    df = detect_candlestick_patterns(df, open_col, close_col, high_col, low_col)
    
    # Advanced candlestick patterns from candlestick library
    patterns = ['doji', 'hammer', 'shooting_star', 'engulfing', 'morning_star', 'evening_star', 'three_white_soldiers', 'three_black_crows', 'hanging_man', 'inverted_hammer', 'piercing_pattern', 'dark_cloud_cover']
    for pattern in patterns:
        try:
            func = getattr(candlestick, pattern)
            df[pattern] = func(df)
        except Exception:
            df[pattern] = 0
    
    # Drop rows with NaN (from indicator calculation)
    df = df.dropna()
    return df

class TestPreprocessFeatures(unittest.TestCase):
    def test_preprocess_features(self):
        # Create mock price data
        data = {
            'Open': np.linspace(100, 120, 240),
            'High': np.linspace(101, 121, 240),
            'Low': np.linspace(99, 119, 240),
            'Close': np.linspace(100, 120, 240),
            'Volume': np.random.randint(1000, 2000, 240)
        }
        price_df = pd.DataFrame(data)
        sentiment_score = 0.5
        features_df = preprocess_features(price_df, sentiment_score)
        # Check that all expected columns exist
        expected_cols = [
            'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26', 'rsi_14',
            'macd', 'macd_signal', 'bb_high', 'bb_low', 'stoch_k', 'stoch_d', 'atr_14', 'news_sentiment',
            'doji', 'hammer', 'bullish_engulfing', 'bearish_engulfing', 'shooting_star'
        ]
        for col in expected_cols:
            self.assertIn(col, features_df.columns)
        # Check that sentiment is set correctly
        self.assertTrue((features_df['news_sentiment'] == sentiment_score).all())
        # Check no NaNs
        self.assertFalse(features_df.isnull().any().any())

if __name__ == "__main__":
    unittest.main()
