import pandas as pd
import ta.trend
import ta.momentum
import ta.volume
import Pattern
import pandas_ta as pta  # NEW: pandas-ta for advanced price action

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
    Add technical indicators, price action, and news sentiment to the price DataFrame.
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
    # Ensure all price columns are present and non-null
    for col in [open_col, high_col, low_col, close_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required price column: {col}")
        if df[col].isnull().any():
            df = df[df[col].notnull()]
    # --- ATR (Average True Range) ---
    from ta.volatility import AverageTrueRange
    if len(df) >= 14:
        df['atr_14'] = AverageTrueRange(high=df[high_col], low=df[low_col], close=df[close_col], window=14).average_true_range()
        print('\n[DEBUG] After ATR calculation:')
        print(df.head(3))
        print('[DEBUG] Columns:', df.columns)
        print(f"[DEBUG] ATR_14: min={df['atr_14'].min()}, max={df['atr_14'].max()}, mean={df['atr_14'].mean()}, NaNs={df['atr_14'].isnull().sum()}")
        if df['atr_14'].isnull().sum() > 0:
            print('[WARNING] NaN values found in atr_14')
    else:
        print('[WARNING] Not enough rows for ATR calculation, skipping atr_14')
    # --- Volatility (20-period rolling std) ---
    if len(df) >= 20:
        df['volatility_20'] = df[close_col].rolling(window=20).std()
        print('\n[DEBUG] After volatility calculation:')
        print(df.head(3))
        print('[DEBUG] Columns:', df.columns)
        print(f"[DEBUG] volatility_20: min={df['volatility_20'].min()}, max={df['volatility_20'].max()}, mean={df['volatility_20'].mean()}, NaNs={df['volatility_20'].isnull().sum()}")
        if df['volatility_20'].isnull().sum() > 0:
            print('[WARNING] NaN values found in volatility_20')
    else:
        print('[WARNING] Not enough rows for volatility calculation, skipping volatility_20')
    # --- Candlestick Patterns ---
    try:
        df = pta.cdl_pattern(df, open=open_col, high=high_col, low=low_col, close=close_col)
        print('\n[DEBUG] After candlestick pattern calculation:')
        print(df.head(3))
        print('[DEBUG] Columns:', df.columns)
    except Exception as e:
        print(f"[Warning] Could not compute candlestick patterns: {e}")
    # --- After all features ---
    print('\n[DEBUG] After all feature engineering:')
    print(df.head(3))
    print('[DEBUG] Columns:', df.columns)
    # Drop rows with any NaNs after all features are created
    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    print(f'[DEBUG] Dropped {n_before - n_after} rows with NaNs after feature engineering. Remaining rows: {n_after}')
    # --- Check for missing required columns ---
    required_cols = [open_col, high_col, low_col, close_col, 'atr_14', 'volatility_20']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[ERROR] Required column missing after feature engineering: {col}")
        if df[col].isnull().sum() > 0:
            raise ValueError(f"[ERROR] NaN values found in required column: {col}")
    # --- Custom Fibonacci retracement levels (rolling window) ---
    N = 20
    rolling_high = df[high_col].rolling(window=N).max()
    rolling_low = df[low_col].rolling(window=N).min()
    df['fib_0'] = rolling_high
    df['fib_236'] = rolling_high - (rolling_high - rolling_low) * 0.236
    df['fib_382'] = rolling_high - (rolling_high - rolling_low) * 0.382
    df['fib_500'] = rolling_high - (rolling_high - rolling_low) * 0.5
    df['fib_618'] = rolling_high - (rolling_high - rolling_low) * 0.618
    df['fib_786'] = rolling_high - (rolling_high - rolling_low) * 0.786
    df['fib_100'] = rolling_low
    for level in ['fib_0', 'fib_236', 'fib_382', 'fib_500', 'fib_618', 'fib_786', 'fib_100']:
        df[f'dist_to_{level}'] = df[close_col] - df[level]
    # --- Trendline detection (basic proxy) ---
    # Calculate rolling min/max slope as a proxy for trendlines
    N = 20
    df['trendline_slope_high'] = df[high_col].rolling(window=N).apply(lambda x: (x[-1] - x[0]) / N, raw=True)
    df['trendline_slope_low'] = df[low_col].rolling(window=N).apply(lambda x: (x[-1] - x[0]) / N, raw=True)
    df['trendline_up'] = (df['trendline_slope_high'] > 0).astype(int)
    df['trendline_down'] = (df['trendline_slope_low'] < 0).astype(int)
    # --- Keep your existing custom price action features ---
    # (Breakouts, support/resistance, pin bar, etc. as before)
    # 1. Breakout detection (close above last N highs/lows)
    N = 20
    # For shift errors, ensure rolling and shift are called on pandas Series, not numpy arrays
    high_rolling_max = df[high_col].rolling(window=N, min_periods=1).max()
    low_rolling_min = df[low_col].rolling(window=N, min_periods=1).min()
    high_rolling_max = high_rolling_max.reindex(df.index)
    low_rolling_min = low_rolling_min.reindex(df.index)
    df['breakout_high'] = (df[close_col] > high_rolling_max.shift(1)).astype(int)
    df['breakout_low'] = (df[close_col] < low_rolling_min.shift(1)).astype(int)
    # 2. Support/resistance proximity (distance to recent high/low)
    df['dist_to_high'] = df[close_col] - high_rolling_max.shift(1)
    df['dist_to_low'] = df[close_col] - low_rolling_min.shift(1)
    # 3. Pin bar (long wick, small body)
    body = abs(df[close_col] - df[open_col])
    upper_wick = df[high_col] - df[[close_col, open_col]].max(axis=1)
    lower_wick = df[[close_col, open_col]].min(axis=1) - df[low_col]
    df['pin_bar'] = ((body < (df[high_col] - df[low_col]) * 0.3) & ((upper_wick > 2 * body) | (lower_wick > 2 * body))).astype(int)
    # 4. Engulfing bar (bullish/bearish)
    df['bullish_engulfing_bar'] = ((df[close_col] > df[open_col]) & (df[close_col].shift(1) < df[open_col].shift(1)) & (df[close_col] > df[open_col].shift(1)) & (df[open_col] < df[close_col].shift(1))).astype(int)
    df['bearish_engulfing_bar'] = ((df[close_col] < df[open_col]) & (df[close_col].shift(1) > df[open_col].shift(1)) & (df[close_col] < df[open_col].shift(1)) & (df[open_col] > df[close_col].shift(1))).astype(int)
    # 5. Inside bar (current high/low within previous bar)
    df['inside_bar'] = ((df[high_col] < df[high_col].shift(1)) & (df[low_col] > df[low_col].shift(1))).astype(int)
    # 6. Volatility spike (ATR or std dev > recent average)
    if 'atr_14' in df.columns:
        df['vol_spike_atr'] = (df['atr_14'] > pd.Series(df['atr_14']).rolling(window=N).mean().shift(1) * 1.5).astype(int)
    else:
        print('[Warning] ATR_14 not found, skipping vol_spike_atr calculation.')
    if 'volatility_20' in df.columns:
        df['vol_spike_std'] = (df['volatility_20'] > pd.Series(df['volatility_20']).rolling(window=N).mean().shift(1) * 1.5).astype(int)
    else:
        print('[Warning] volatility_20 not found, skipping vol_spike_std calculation.')
    # 7. Range vs. trend detection (is price in range or trending)
    # Simple: if price is near both recent high and low, it's ranging; if breaking out, it's trending
    df['is_trending'] = ((df['breakout_high'] == 1) | (df['breakout_low'] == 1)).astype(int)
    df['is_ranging'] = (((df['dist_to_high'].abs() < (df[high_col] - df[low_col]).rolling(window=N).mean() * 0.2) & (df['dist_to_low'].abs() < (df[high_col] - df[low_col]).rolling(window=N).mean() * 0.2))).astype(int)
    
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
        self.assertFalse(features_df.isnull().values.any())

if __name__ == "__main__":
    unittest.main()
