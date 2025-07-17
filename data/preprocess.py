import pandas as pd
import ta.trend
import ta.momentum
import ta.volume
import pandas_ta as pta  # NEW: pandas-ta for advanced price action

import unittest
import numpy as np
import os
from utils.logger import get_logger
from config import Config
logger = get_logger('preprocess', log_file='logs/preprocess.log')

# --- Advanced Data Cleaning Helpers ---
def clean_missing_and_outliers(df):
    logger.info('Starting data cleaning (missing values, outliers)...')
    df_clean = df.copy()
    # Impute missing values
    n_missing_before = df_clean.isnull().sum().sum()
    if n_missing_before > 0:
        logger.info(f'Missing values before: {n_missing_before}')
        df_clean = df_clean.ffill().bfill()
        n_missing_after_ffill = df_clean.isnull().sum().sum()
        if n_missing_after_ffill > 0:
            for col in df_clean.columns:
                if df_clean[col].isnull().any():
                    mean_val = df_clean[col].mean()
                    df_clean[col] = df_clean[col].fillna(mean_val)
        n_missing_after = df_clean.isnull().sum().sum()
        logger.info(f'Missing values after imputation: {n_missing_after}')
    # Outlier detection (z-score method)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    z_scores = np.abs((df_clean[numeric_cols] - df_clean[numeric_cols].mean()) / df_clean[numeric_cols].std(ddof=0))
    outlier_mask = (z_scores > 3)
    n_outliers = outlier_mask.sum().sum()
    if n_outliers > 0:
        logger.info(f'Outliers detected: {n_outliers}')
        for col in numeric_cols:
            median_val = df_clean[col].median()
            df_clean.loc[outlier_mask[col], col] = median_val
        logger.info('Outliers replaced with median.')
    logger.info('Data cleaning complete.')
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

def detect_supply_demand_zones(df, high_col, low_col, window=20, threshold=0.002):
    """Detect supply (resistance) and demand (support) zones based on swing highs/lows clustering."""
    import numpy as np
    highs = df[high_col].rolling(window=window, min_periods=1).max()
    lows = df[low_col].rolling(window=window, min_periods=1).min()
    # Zones: if price is within threshold of rolling high/low, mark as zone
    df['supply_zone'] = (np.abs(df[high_col] - highs) < threshold * highs).astype(int)
    df['demand_zone'] = (np.abs(df[low_col] - lows) < threshold * lows).astype(int)
    logger.info('Model is mapping supply (resistance) and demand (support) zones...')
    logger.info(f"Detected {df['supply_zone'].sum()} supply and {df['demand_zone'].sum()} demand zone points in window={window}.")
    return df

def detect_wyckoff_phase(df, close_col, volume_col, window=40):
    """Detect Wyckoff cycle phase: accumulation, markup, distribution, markdown."""
    import numpy as np
    # Simple proxy: use rolling min/max and volume spikes
    close = df[close_col]
    volume = df[volume_col]
    phase = []
    for i in range(len(df)):
        if i < window:
            phase.append('unknown')
            continue
        win_close = close.iloc[i-window:i]
        win_vol = volume.iloc[i-window:i]
        price_range = win_close.max() - win_close.min()
        vol_mean = win_vol.mean()
        vol_now = volume.iloc[i]
        if price_range < 0.005 * win_close.mean() and vol_now > 1.5 * vol_mean:
            phase.append('accumulation')
        elif price_range > 0.02 * win_close.mean() and vol_now > 1.5 * vol_mean:
            phase.append('distribution')
        elif win_close.iloc[-1] > win_close.iloc[0]:
            phase.append('markup')
        else:
            phase.append('markdown')
    df['wyckoff_phase'] = phase
    logger.info('Model is detecting Wyckoff cycle phase (accumulation, markup, distribution, markdown)...')
    logger.info(f"Wyckoff phase counts: {df['wyckoff_phase'].value_counts().to_dict()}")
    return df

def detect_trend_range_regime(df, close_col, atr_col, window=20, atr_threshold=0.001):
    """Classify regime as 'trend' or 'range' using ATR and moving average slope."""
    import numpy as np
    ma = df[close_col].rolling(window=window).mean()
    slope = ma.diff(window)
    atr = df[atr_col]
    regime = []
    for i in range(len(df)):
        if i < window:
            regime.append('unknown')
            continue
        if abs(slope.iloc[i]) > atr_threshold and atr.iloc[i] > atr_threshold:
            regime.append('trend')
        else:
            regime.append('range')
    df['market_regime'] = regime
    logger.info('Model is classifying market regime (trend vs. range)...')
    logger.info(f"Regime counts: {df['market_regime'].value_counts().to_dict()}")
    return df

def detect_head_shoulders(df, close_col, window=20):
    """Detect head & shoulders and inverse head & shoulders patterns."""
    import numpy as np
    pattern = [0] * len(df)
    inv_pattern = [0] * len(df)
    for i in range(window, len(df)):
        win = df[close_col].iloc[i-window:i].values
        if len(win) < window:
            continue
        # Simple proxy: look for 3 peaks/troughs with the middle one highest/lowest
        mid = window // 2
        left = win[:mid]
        right = win[mid+1:]
        if win[mid] == max(win) and win[mid] > max(left) and win[mid] > max(right):
            pattern[i] = 1  # Head & Shoulders
        if win[mid] == min(win) and win[mid] < min(left) and win[mid] < min(right):
            inv_pattern[i] = 1  # Inverse Head & Shoulders
    df['head_shoulders'] = pattern
    df['inv_head_shoulders'] = inv_pattern
    logger.info(f"Detected {sum(pattern)} head & shoulders and {sum(inv_pattern)} inverse head & shoulders patterns.")
    return df

def detect_double_top_bottom(df, close_col, window=20, threshold=0.001):
    """Detect double top and double bottom patterns."""
    import numpy as np
    double_top = [0] * len(df)
    double_bottom = [0] * len(df)
    for i in range(window, len(df)):
        win = df[close_col].iloc[i-window:i].values
        if len(win) < window:
            continue
        peaks = (win == max(win)).nonzero()[0]
        troughs = (win == min(win)).nonzero()[0]
        if len(peaks) >= 2 and abs(peaks[0] - peaks[1]) > 2 and abs(win[peaks[0]] - win[peaks[1]]) < threshold * win[peaks[0]]:
            double_top[i] = 1
        if len(troughs) >= 2 and abs(troughs[0] - troughs[1]) > 2 and abs(win[troughs[0]] - win[troughs[1]]) < threshold * win[troughs[0]]:
            double_bottom[i] = 1
    df['double_top'] = double_top
    df['double_bottom'] = double_bottom
    logger.info(f"Detected {sum(double_top)} double tops and {sum(double_bottom)} double bottoms.")
    return df

def detect_wedges(df, high_col, low_col, window=20):
    """Detect rising and falling wedge patterns."""
    import numpy as np
    rising_wedge = [0] * len(df)
    falling_wedge = [0] * len(df)
    for i in range(window, len(df)):
        highs = df[high_col].iloc[i-window:i].values
        lows = df[low_col].iloc[i-window:i].values
        if len(highs) < window or len(lows) < window:
            continue
        # Rising wedge: highs and lows both trending up, but highs rising slower
        if np.polyfit(range(window), highs, 1)[0] > 0 and np.polyfit(range(window), lows, 1)[0] > 0:
            if np.polyfit(range(window), highs, 1)[0] < np.polyfit(range(window), lows, 1)[0]:
                rising_wedge[i] = 1
        # Falling wedge: highs and lows both trending down, but lows falling slower
        if np.polyfit(range(window), highs, 1)[0] < 0 and np.polyfit(range(window), lows, 1)[0] < 0:
            if np.polyfit(range(window), highs, 1)[0] > np.polyfit(range(window), lows, 1)[0]:
                falling_wedge[i] = 1
    df['rising_wedge'] = rising_wedge
    df['falling_wedge'] = falling_wedge
    logger.info(f"Detected {sum(rising_wedge)} rising wedges and {sum(falling_wedge)} falling wedges.")
    return df

def detect_fakeouts(df, close_col, high_col, low_col, window=20, threshold=0.001):
    """Detect fake-outs (false breakouts) above/below recent highs/lows."""
    import numpy as np
    fakeout_up = [0] * len(df)
    fakeout_down = [0] * len(df)
    for i in range(window, len(df)):
        win_close = df[close_col].iloc[i-window:i].values
        win_high = df[high_col].iloc[i-window:i].values
        win_low = df[low_col].iloc[i-window:i].values
        if len(win_close) < window:
            continue
        # Fakeout up: close breaks above previous high, then closes back below
        if win_close[-2] > max(win_high[:-2]) and win_close[-1] < max(win_high[:-2]):
            fakeout_up[i] = 1
        # Fakeout down: close breaks below previous low, then closes back above
        if win_close[-2] < min(win_low[:-2]) and win_close[-1] > min(win_low[:-2]):
            fakeout_down[i] = 1
    df['fakeout_up'] = fakeout_up
    df['fakeout_down'] = fakeout_down
    logger.info(f"Detected {sum(fakeout_up)} fakeout ups and {sum(fakeout_down)} fakeout downs.")
    return df

def preprocess_features(price_df, sentiment_score, use_multi_timeframe=True):
    logger.info('Model is analyzing technical indicators...')
    
    # Add multi-timeframe analysis if enabled
    if use_multi_timeframe:
        try:
            from data.multi_timeframe import get_multi_timeframe_features
            # Extract symbol from price_df (assuming it's in the data somewhere)
            # For now, we'll use the first symbol from config
            symbol = Config.TRADING_PAIRS[0] if Config.TRADING_PAIRS else 'USDJPY'
            logger.info(f'Adding multi-timeframe features for {symbol}')
            
            # Get multi-timeframe features
            mtf_features = get_multi_timeframe_features(symbol, lookback_days=90)
            if mtf_features is not None and not mtf_features.empty:
                # Merge multi-timeframe features with base features
                # We'll add them after the base features are created
                logger.info(f'Multi-timeframe features shape: {mtf_features.shape}')
            else:
                logger.warning('Could not fetch multi-timeframe features, continuing with base features only')
                mtf_features = None
        except Exception as e:
            logger.warning(f'Error in multi-timeframe analysis: {e}, continuing with base features only')
            mtf_features = None
    else:
        mtf_features = None
    # Flatten MultiIndex columns if present
    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = ['_'.join([str(i) for i in col if i]) for col in price_df.columns.values]
    df = price_df.copy()
    # --- Advanced Data Cleaning ---
    try:
        df = clean_missing_and_outliers(df)
        logger.info('Missing/outlier cleaning done.')
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
            logger.info('\n[DEBUG] After ATR calculation:')
            logger.info(f"{df.head(3)}")
            logger.info(f"[DEBUG] Columns: {df.columns}")
            logger.info(f"[DEBUG] ATR_14: min={df['atr_14'].min()}, max={df['atr_14'].max()}, mean={df['atr_14'].mean()}, NaNs={df['atr_14'].isnull().sum()}")
            if df['atr_14'].isnull().sum() > 0:
                logger.warning('[WARNING] NaN values found in atr_14')
        else:
            logger.warning('[WARNING] Not enough rows for ATR calculation, skipping atr_14')
        # --- Volatility (20-period rolling std) ---
        if len(df) >= 20:
            df['volatility_20'] = df[close_col].rolling(window=20).std()
            logger.info('\n[DEBUG] After volatility calculation:')
            logger.info(f"{df.head(3)}")
            logger.info(f"[DEBUG] Columns: {df.columns}")
            logger.info(f"[DEBUG] volatility_20: min={df['volatility_20'].min()}, max={df['volatility_20'].max()}, mean={df['volatility_20'].mean()}, NaNs={df['volatility_20'].isnull().sum()}")
            if df['volatility_20'].isnull().sum() > 0:
                logger.warning('[WARNING] NaN values found in volatility_20')
        else:
            logger.warning('[WARNING] Not enough rows for volatility calculation, skipping volatility_20')
        # --- Candlestick Patterns ---
        try:
            df = pta.cdl_pattern(df, open=open_col, high=high_col, low=low_col, close=close_col)
            logger.info('\n[DEBUG] After candlestick pattern calculation:')
            logger.info(f"{df.head(3)}")
            logger.info(f"[DEBUG] Columns: {df.columns}")
        except Exception as e:
            logger.warning(f"[Warning] Could not compute candlestick patterns: {e}")
        logger.info('Model is studying candlestick patterns...')
        # --- After all features ---
        logger.info('\n[DEBUG] After all feature engineering:')
        logger.info(f"{df.head(3)}")
        logger.info(f"[DEBUG] Columns: {df.columns}")
        # Drop rows with any NaNs after all features are created
        n_before = len(df)
        df = df.dropna()
        n_after = len(df)
        logger.info(f'[DEBUG] Dropped {n_before - n_after} rows with NaNs after feature engineering. Remaining rows: {n_after}')
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
        high_col_series = pd.Series(df[high_col], index=df.index)
        low_col_series = pd.Series(df[low_col], index=df.index)
        high_rolling_max = high_col_series.rolling(window=N, min_periods=1).max()
        low_rolling_min = low_col_series.rolling(window=N, min_periods=1).min()
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
            logger.warning('[Warning] ATR_14 not found, skipping vol_spike_atr calculation.')
        if 'volatility_20' in df.columns:
            df['vol_spike_std'] = (df['volatility_20'] > pd.Series(df['volatility_20']).rolling(window=N).mean().shift(1) * 1.5).astype(int)
        else:
            logger.warning('[Warning] volatility_20 not found, skipping vol_spike_std calculation.')
        # 7. Range vs. trend detection (is price in range or trending)
        # Simple: if price is near both recent high and low, it's ranging; if breaking out, it's trending
        df['is_trending'] = ((df['breakout_high'] == 1) | (df['breakout_low'] == 1)).astype(int)
        df['is_ranging'] = (((df['dist_to_high'].abs() < (df[high_col] - df[low_col]).rolling(window=N).mean() * 0.2) & (df['dist_to_low'].abs() < (df[high_col] - df[low_col]).rolling(window=N).mean() * 0.2))).astype(int)
        
        logger.info('Model is extracting price action features...')
        # --- Advanced Market Structure Detection ---
        close_col = [col for col in price_df.columns if col.startswith('Close')][0]
        high_col = [col for col in price_df.columns if col.startswith('High')][0]
        low_col = [col for col in price_df.columns if col.startswith('Low')][0]
        volume_col = [col for col in price_df.columns if col.startswith('Volume')][0]
        atr_col = 'atr_14' if 'atr_14' in price_df.columns else None
        df = price_df.copy()
        if atr_col:
            df = detect_trend_range_regime(df, close_col, atr_col)
        df = detect_supply_demand_zones(df, high_col, low_col)
        df = detect_wyckoff_phase(df, close_col, volume_col)
        # --- Wyckoff phase one-hot encoding ---
        if 'wyckoff_phase' in df.columns:
            wyckoff_dummies = pd.get_dummies(df['wyckoff_phase'], prefix='wyckoff')
            df = pd.concat([df, wyckoff_dummies], axis=1)
            df.drop('wyckoff_phase', axis=1, inplace=True)
            # Ensure all possible wyckoff one-hot columns are present
            wyckoff_phases = [
                'wyckoff_accumulation',
                'wyckoff_distribution',
                'wyckoff_markdown',
                'wyckoff_markup',
                'wyckoff_unknown'
            ]
            for col in wyckoff_phases:
                if col not in df.columns:
                    df[col] = 0
        logger.info('Advanced market structure features added.')
        # --- Advanced Price Action Patterns ---
        df = detect_head_shoulders(df, close_col)
        df = detect_double_top_bottom(df, close_col)
        df = detect_wedges(df, high_col, low_col)
        df = detect_fakeouts(df, close_col, high_col, low_col)
        logger.info('Advanced price action pattern features added.')
        logger.info('Feature engineering complete.')
        # After all indicators/features are computed:
        missing_before = df.isna().sum().sum()
        logger.info(f"[preprocess_features] Total missing values before fill: {missing_before}")
        df = df.ffill().bfill().fillna(0)
        missing_after = df.isna().sum().sum()
        logger.info(f"[preprocess_features] Total missing values after fill: {missing_after}")
        
        # Add multi-timeframe features if available
        if mtf_features is not None and not mtf_features.empty:
            try:
                logger.info('Integrating multi-timeframe features...')
                # Align indices
                common_index = df.index.intersection(mtf_features.index)
                if len(common_index) > 0:
                    df_aligned = df.loc[common_index]
                    mtf_aligned = mtf_features.loc[common_index]
                    
                    # Add multi-timeframe features (excluding price columns to avoid duplicates)
                    mtf_feature_cols = [col for col in mtf_aligned.columns 
                                      if not col.startswith(('Open', 'High', 'Low', 'Close', 'Volume'))]
                    
                    for col in mtf_feature_cols:
                        if col not in df_aligned.columns:
                            df_aligned[col] = mtf_aligned[col]
                    
                    df = df_aligned
                    logger.info(f'Added {len(mtf_feature_cols)} multi-timeframe features')
                else:
                    logger.warning('No common index found for multi-timeframe integration')
            except Exception as e:
                logger.error(f'Error integrating multi-timeframe features: {e}')
        
        return df
    except Exception as e:
        logger.error(f'Error in preprocess_features: {e}', exc_info=True)
        raise

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
