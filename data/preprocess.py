import pandas as pd
import ta.trend
import ta.momentum
import ta.volume
import pandas_ta as pta  # NEW: pandas-ta for advanced price action
import ta.volatility

import unittest
import numpy as np
import os
from utils.logger import get_logger
from config import Config
import sys
from features.indicators import INDICATORS
from features.patterns import PATTERNS
from sklearn.preprocessing import LabelEncoder

indicator_registry = {ind.name: ind.func for ind in INDICATORS}
pattern_registry = {pat.name: pat.func for pat in PATTERNS}
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

# --- All technical indicator and candlestick pattern feature engineering is now handled via the registries in features/indicators.py and features/patterns.py. ---

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

def detect_market_regime(df, close_col, high_col, low_col, atr_col, window=20):
    """Composite regime detection: uptrend, downtrend, range, consolidation, choppy."""
    ma = df[close_col].rolling(window=window).mean()
    slope = ma.diff(window)
    atr = df[atr_col]
    # Calculate ADX
    adx = ta.trend.adx(df[high_col], df[low_col], df[close_col], window=window)
    # Calculate Bollinger Band width
    bb_high = ma + 2 * df[close_col].rolling(window=window).std()
    bb_low = ma - 2 * df[close_col].rolling(window=window).std()
    bb_width = (bb_high - bb_low) / ma
    regime = []
    for i in range(len(df)):
        if i < window:
            regime.append('range')
            continue
        # Uptrend
        if adx.iloc[i] > 20 and slope.iloc[i] > 0.001:
            regime.append('uptrend')
        # Downtrend
        elif adx.iloc[i] > 20 and slope.iloc[i] < -0.001:
            regime.append('downtrend')
        # Consolidation
        elif bb_width.iloc[i] < 0.01 and atr.iloc[i] < 0.001:
            regime.append('consolidation')
        # Choppy
        elif atr.iloc[i] > 0.002 and adx.iloc[i] < 15:
            regime.append('choppy')
        # Default to range
        else:
            regime.append('range')
    df['market_regime'] = regime
    logger.info('Model is classifying market regime (uptrend, downtrend, range, consolidation, choppy)...')
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

def preprocess_features(price_df, sentiment_score, use_multi_timeframe=True):
    logger.info('Model is analyzing technical indicators...')
    
    # Add multi-timeframe analysis if enabled
    if use_multi_timeframe:
        try:
            from data.multi_timeframe import get_multi_timeframe_features
            symbol = Config.TRADING_PAIRS[0] if Config.TRADING_PAIRS else 'USDJPY'
            logger.info(f'Adding multi-timeframe features for {symbol}')
            mtf_features = get_multi_timeframe_features(symbol, lookback_days=90)
            if mtf_features is not None and not mtf_features.empty:
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
    try:
        df = clean_missing_and_outliers(df)
        logger.info('Missing/outlier cleaning done.')
        # Use the correct column names for the pair
        close_col = [col for col in df.columns if col.startswith('Close')][0]
        open_col = [col for col in df.columns if col.startswith('Open')][0]
        high_col = [col for col in df.columns if col.startswith('High')][0]
        low_col = [col for col in df.columns if col.startswith('Low')][0]
        volume_col = [col for col in df.columns if col.startswith('Volume')][0]
        for col in [open_col, high_col, low_col, close_col]:
            if col not in df.columns:
                raise ValueError(f"Missing required price column: {col}")
            if df[col].isnull().any():
                df = df[df[col].notnull()]
        # --- Registry-based Indicator Calculation ---
        logger.info('Calculating indicators from registry...')
        for name, indicator in indicator_registry.items():
            try:
                result = indicator(df)
                if isinstance(result, pd.Series):
                    df[name] = result
                elif isinstance(result, pd.DataFrame):
                    for col in result.columns:
                        df[col] = result[col]
                else:
                    logger.warning(f"Indicator {name} returned unsupported type: {type(result)}")
            except Exception as e:
                logger.warning(f"Indicator {name} failed: {e}")
        # --- Registry-based Pattern Calculation ---
        logger.info('Calculating patterns from registry...')
        for name, pattern in pattern_registry.items():
            try:
                result = pattern(df)
                if isinstance(result, pd.Series):
                    df[name] = result
                elif isinstance(result, pd.DataFrame):
                    for col in result.columns:
                        df[col] = result[col]
                else:
                    logger.warning(f"Pattern {name} returned unsupported type: {type(result)}")
            except Exception as e:
                logger.warning(f"Pattern {name} failed: {e}")
        # --- Market regime detection (always present) ---
        atr_col = [col for col in df.columns if 'atr' in col.lower() and '_1h' not in col and '_4h' not in col and '_1d' not in col][0]
        df = detect_market_regime(df, close_col, high_col, low_col, atr_col, window=20)
        # --- Multi-timeframe features integration (unchanged) ---
        if mtf_features is not None and not mtf_features.empty:
            try:
                logger.info('Integrating multi-timeframe features...')
                common_index = df.index.intersection(mtf_features.index)
                if len(common_index) > 0:
                    df_aligned = df.loc[common_index]
                    mtf_aligned = mtf_features.loc[common_index]
                    mtf_feature_cols = [col for col in mtf_aligned.columns 
                                      if not col.startswith(('Open', 'High', 'Low', 'Close', 'Volume'))]
                    # --- Refactor: Use pd.concat for bulk addition ---
                    mtf_to_add = mtf_aligned[mtf_feature_cols]
                    df = pd.concat([df_aligned, mtf_to_add], axis=1)
                    logger.info(f'Added {len(mtf_feature_cols)} multi-timeframe features')
                    logger.info(f'[DEBUG] Columns after multi-timeframe integration: {list(df.columns)}')
                else:
                    logger.warning('No common index found for multi-timeframe integration')
            except Exception as e:
                logger.error(f'Error integrating multi-timeframe features: {e}')
        # --- Ensure all expected features are present ---
        all_expected = list(indicator_registry.keys()) + list(pattern_registry.keys())
        # Add multi-timeframe expected features if needed (unchanged)
        for col in all_expected:
            if col not in df.columns:
                df[col] = 0
        # --- After all feature engineering, always add news_sentiment column ---
        df['news_sentiment'] = sentiment_score
        logger.info(f"[DEBUG] Final columns in preprocess_features: {list(df.columns)}")
        # --- Encode categorical features as numeric ---
        categorical_cols = []
        for col in df.columns:
            if col in ['regime_label', 'market_regime', 'wyckoff_phase']:
                categorical_cols.append(col)
            elif isinstance(df[col], pd.Series) and pd.api.types.is_object_dtype(df[col]):
                categorical_cols.append(col)
        if categorical_cols:
            for col in categorical_cols:
                # Use one-hot encoding for regime_label and market_regime, integer for wyckoff_phase
                if col in ['regime_label', 'market_regime']:
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                else:
                    # For wyckoff_phase and other object columns, use integer encoding
                    df[col] = pd.factorize(df[col])[0]
        logger.info(f"[DEBUG] Encoded categorical columns: {categorical_cols}")
        # Fill any remaining NaNs
        df = df.ffill().bfill().fillna(0)
        return df
    except Exception as e:
        logger.error(f'Error in preprocess_features: {e}', exc_info=True)
        raise

# Remove legacy pattern/indicator detection functions and test class
