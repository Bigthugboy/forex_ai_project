import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from config import Config
from utils.logger import get_logger
from data.fetch_market import get_price_data

logger = get_logger('multi_timeframe', log_file='logs/multi_timeframe.log')

def detect_structure_trend(df, high_col, low_col, window=10):
    """
    Detect market structure: uptrend (higher highs/lows), downtrend (lower highs/lows), or range.
    Returns a Series with values: 'uptrend', 'downtrend', 'range'.
    """
    highs = df[high_col].rolling(window=window, min_periods=window).max()
    lows = df[low_col].rolling(window=window, min_periods=window).min()
    trend = []
    for i in range(len(df)):
        if i < window:
            trend.append('range')
            continue
        # Compare current high/low to previous window
        prev_high = highs.iloc[i-window]
        prev_low = lows.iloc[i-window]
        curr_high = highs.iloc[i]
        curr_low = lows.iloc[i]
        if curr_high > prev_high and curr_low > prev_low:
            trend.append('uptrend')
        elif curr_high < prev_high and curr_low < prev_low:
            trend.append('downtrend')
        else:
            trend.append('range')
    return pd.Series(trend, index=df.index)

class MultiTimeframeData:
    def __init__(self):
        self.timeframes = {
            '1h': '1h',    # 1 hour
            '4h': '4h'     # 4 hours
        }
        self.data_cache = {}
        
    def get_multi_timeframe_data(self, symbol, lookback_days=60):
        """
        Fetch data for all timeframes for a given symbol
        """
        logger.info(f"Fetching multi-timeframe data for {symbol}")
        
        multi_tf_data = {}
        
        for tf_name, tf_interval in self.timeframes.items():
            try:
                logger.info(f"Fetching {tf_name} data for {symbol}")
                data = get_price_data(symbol, interval=tf_interval, lookback=lookback_days)
                
                if data is not None and not data.empty:
                    multi_tf_data[tf_name] = data
                    logger.info(f"Successfully fetched {len(data)} rows of {tf_name} data for {symbol}")
                else:
                    logger.warning(f"No {tf_name} data received for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error fetching {tf_name} data for {symbol}: {e}")
                
        # Also fetch D1 data for structure analysis
        try:
            logger.info(f"Fetching D1 data for {symbol}")
            d1_data = get_price_data(symbol, interval='1d', lookback=lookback_days)
            if d1_data is not None and not d1_data.empty:
                multi_tf_data['1d'] = d1_data
                logger.info(f"Successfully fetched {len(d1_data)} rows of D1 data for {symbol}")
            else:
                logger.warning(f"No D1 data received for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching D1 data for {symbol}: {e}")
        return multi_tf_data
    
    def create_multi_timeframe_features(self, symbol, lookback_days=60):
        """
        Create features from multiple timeframes
        """
        logger.info(f"Creating multi-timeframe features for {symbol}")
        
        # Get data for all timeframes
        multi_tf_data = self.get_multi_timeframe_data(symbol, lookback_days)
        
        if not multi_tf_data:
            logger.error(f"No multi-timeframe data available for {symbol}")
            return None
            
        # Use 1h as the base timeframe (primary)
        base_data = multi_tf_data.get('1h')
        if base_data is None or base_data.empty:
            logger.error(f"No 1h data available for {symbol}")
            return None
            
        # Create multi-timeframe features
        features_df = self._create_mtf_features(base_data, multi_tf_data)
        
        logger.info(f"Created multi-timeframe features with shape: {features_df.shape}")
        return features_df
    
    def _create_mtf_features(self, base_data, multi_tf_data):
        """
        Create features from multiple timeframes
        """
        # Ensure base_data index is timezone-naive
        if base_data.index.tz is not None:
            base_data.index = base_data.index.tz_localize(None)
        features_df = base_data.copy()
        
        # Add structure trend for 1h base
        close_col = [col for col in base_data.columns if col.startswith('Close')][0]
        high_col = [col for col in base_data.columns if col.startswith('High')][0]
        low_col = [col for col in base_data.columns if col.startswith('Low')][0]
        features_df['structure_trend_1h'] = detect_structure_trend(base_data, high_col, low_col, window=10)
        logger.info(f"Added structure_trend_1h feature")
        
        # Add features from other timeframes
        for tf_name, tf_data in multi_tf_data.items():
            if tf_name == '1h':
                continue  # Skip base timeframe
                
            if tf_data is None or tf_data.empty:
                continue
                
            # Ensure tf_data index is timezone-naive
            if tf_data.index.tz is not None:
                tf_data.index = tf_data.index.tz_localize(None)
                
            # Resample other timeframes to match 1h timeframe
            resampled_data = self._resample_to_1h(tf_data, tf_name)
            
            if resampled_data is not None:
                # Ensure resampled_data index is timezone-naive
                if resampled_data.index.tz is not None:
                    resampled_data.index = resampled_data.index.tz_localize(None)
                # Add timeframe-specific features
                features_df = self._add_timeframe_features(features_df, resampled_data, tf_name)
        
        return features_df
    
    def _resample_to_1h(self, data, tf_name):
        """
        Resample data to 1h timeframe for alignment
        """
        try:
            if tf_name == '15m':
                # Forward fill 15m data to 1h
                resampled = data.resample('1h').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min', 
                    'Close': 'last',
                    'Volume': 'sum'
                }).ffill()
                
            elif tf_name == '4h':
                # Backward fill 4h data to 1h
                resampled = data.resample('1h').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last', 
                    'Volume': 'sum'
                }).bfill()
                
            else:
                resampled = data
                
            return resampled
            
        except Exception as e:
            logger.error(f"Error resampling {tf_name} data: {e}")
            return None
    
    def _add_timeframe_features(self, base_df, tf_data, tf_name):
        """
        Add features from a specific timeframe
        """
        try:
            # Ensure both dataframes have the same index and are timezone-naive
            if base_df.index.tz is not None:
                base_df.index = base_df.index.tz_localize(None)
            if tf_data.index.tz is not None:
                tf_data.index = tf_data.index.tz_localize(None)
            # --- Enhanced index alignment and logging ---
            base_index = base_df.index
            tf_index = tf_data.index
            if not base_index.equals(tf_index):
                logger.warning(f"[MTF ALIGNMENT] Index mismatch for {tf_name}: base_index({len(base_index)}) != tf_index({len(tf_index)}). Aligning with reindex.")
                logger.debug(f"[MTF ALIGNMENT] base_index sample: {list(base_index[:5])} ... {list(base_index[-5:])}")
                logger.debug(f"[MTF ALIGNMENT] tf_index sample: {list(tf_index[:5])} ... {list(tf_index[-5:])}")
            tf_data = tf_data.reindex(base_df.index, method='ffill')
            if tf_data.isnull().any().any():
                logger.warning(f"[MTF ALIGNMENT] After reindex, NaNs present in tf_data for {tf_name}. Filling with 0.")
                tf_data = tf_data.fillna(0)
            # --- Fix: Align tf_data to base_df index using reindex and ffill ---
            # Now, base_df and tf_data always have the same index
            # Calculate timeframe-specific indicators
            close_col = [col for col in tf_data.columns if col.startswith('Close')][0]
            high_col = [col for col in tf_data.columns if col.startswith('High')][0]
            low_col = [col for col in tf_data.columns if col.startswith('Low')][0]
            
            # Moving averages for this timeframe
            base_df[f'sma_20_{tf_name}'] = tf_data[close_col].rolling(window=20).mean()
            base_df[f'sma_50_{tf_name}'] = tf_data[close_col].rolling(window=50).mean()
            base_df[f'ema_12_{tf_name}'] = tf_data[close_col].ewm(span=12).mean()
            base_df[f'ema_26_{tf_name}'] = tf_data[close_col].ewm(span=26).mean()
            
            # RSI for this timeframe
            delta = tf_data[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            base_df[f'rsi_14_{tf_name}'] = 100 - (100 / (1 + rs))
            
            # MACD for this timeframe
            ema12 = tf_data[close_col].ewm(span=12).mean()
            ema26 = tf_data[close_col].ewm(span=26).mean()
            base_df[f'macd_{tf_name}'] = ema12 - ema26
            base_df[f'macd_signal_{tf_name}'] = base_df[f'macd_{tf_name}'].ewm(span=9).mean()
            
            # Bollinger Bands for this timeframe
            sma20 = tf_data[close_col].rolling(window=20).mean()
            std20 = tf_data[close_col].rolling(window=20).std()
            base_df[f'bb_high_{tf_name}'] = sma20 + (std20 * 2)
            base_df[f'bb_low_{tf_name}'] = sma20 - (std20 * 2)
            
            # ATR for this timeframe
            high_low = tf_data[high_col] - tf_data[low_col]
            high_close = np.abs(tf_data[high_col] - tf_data[close_col].shift())
            low_close = np.abs(tf_data[low_col] - tf_data[close_col].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            base_df[f'atr_14_{tf_name}'] = true_range.rolling(window=14).mean()
            
            # Volatility for this timeframe
            base_df[f'volatility_20_{tf_name}'] = tf_data[close_col].rolling(window=20).std()
            
            # Price position relative to timeframe
            base_df[f'price_vs_sma20_{tf_name}'] = (tf_data[close_col] - base_df[f'sma_20_{tf_name}']) / base_df[f'sma_20_{tf_name}']
            base_df[f'price_vs_sma50_{tf_name}'] = (tf_data[close_col] - base_df[f'sma_50_{tf_name}']) / base_df[f'sma_50_{tf_name}']
            
            # Trend strength indicators
            base_df[f'trend_strength_{tf_name}'] = abs(base_df[f'sma_20_{tf_name}'] - base_df[f'sma_50_{tf_name}']) / base_df[f'sma_50_{tf_name}']
            
            # Structure trend feature
            base_df[f'structure_trend_{tf_name}'] = detect_structure_trend(tf_data, high_col, low_col, window=10)
            logger.info(f"Added structure_trend_{tf_name} feature")
            
            logger.info(f"Added {tf_name} timeframe features")
            
        except Exception as e:
            logger.error(f"Error adding {tf_name} timeframe features: {e}")
            
        return base_df
    
    def create_timeframe_confluence_features(self, multi_tf_data):
        """
        Create confluence features across timeframes
        """
        logger.info("Creating timeframe confluence features")
        
        confluence_features = {}
        
        # Get base 1h data
        base_data = multi_tf_data.get('1h')
        if base_data is None:
            return confluence_features
            
        close_col = [col for col in base_data.columns if col.startswith('Close')][0]
        
        # Trend alignment across timeframes
        for tf_name in ['15m', '1h', '4h']:
            if tf_name in multi_tf_data:
                tf_data = multi_tf_data[tf_name]
                tf_close = [col for col in tf_data.columns if col.startswith('Close')][0]
                
                # Align timeframes
                common_index = base_data.index.intersection(tf_data.index)
                if len(common_index) > 0:
                    base_aligned = base_data.loc[common_index]
                    tf_aligned = tf_data.loc[common_index]
                    
                    # Trend direction
                    sma20 = tf_aligned[tf_close].rolling(window=20).mean()
                    sma50 = tf_aligned[tf_close].rolling(window=50).mean()
                    
                    confluence_features[f'trend_up_{tf_name}'] = (sma20 > sma50).astype(int)
                    confluence_features[f'trend_down_{tf_name}'] = (sma20 < sma50).astype(int)
                    
                    # RSI conditions
                    delta = tf_aligned[tf_close].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    confluence_features[f'rsi_oversold_{tf_name}'] = (rsi < 30).astype(int)
                    confluence_features[f'rsi_overbought_{tf_name}'] = (rsi > 70).astype(int)
                    
        # Cross-timeframe momentum
        if '15m' in multi_tf_data and '4h' in multi_tf_data:
            # Momentum alignment between 15m and 4h
            tf_15m = multi_tf_data['15m']
            tf_4h = multi_tf_data['4h']
            
            common_index = tf_15m.index.intersection(tf_4h.index)
            if len(common_index) > 0:
                tf_15m_aligned = tf_15m.loc[common_index]
                tf_4h_aligned = tf_4h.loc[common_index]
                
                close_15m = [col for col in tf_15m_aligned.columns if col.startswith('Close')][0]
                close_4h = [col for col in tf_4h_aligned.columns if col.startswith('Close')][0]
                
                # Momentum comparison
                momentum_15m = tf_15m_aligned[close_15m].pct_change(5)
                momentum_4h = tf_4h_aligned[close_4h].pct_change(5)
                
                confluence_features['momentum_aligned'] = ((momentum_15m > 0) & (momentum_4h > 0)).astype(int)
                confluence_features['momentum_divergence'] = ((momentum_15m > 0) & (momentum_4h < 0)).astype(int)
        
        logger.info(f"Created {len(confluence_features)} confluence features")
        return confluence_features
    
    def get_timeframe_summary(self, symbol, lookback_days=90):
        """
        Get a summary of all timeframes for a symbol
        """
        logger.info(f"Getting timeframe summary for {symbol}")
        
        summary = {}
        multi_tf_data = self.get_multi_timeframe_data(symbol, lookback_days)
        
        for tf_name, tf_data in multi_tf_data.items():
            if tf_data is not None and not tf_data.empty:
                close_col = [col for col in tf_data.columns if col.startswith('Close')][0]
                
                summary[tf_name] = {
                    'rows': len(tf_data),
                    'start_date': tf_data.index[0],
                    'end_date': tf_data.index[-1],
                    'current_price': tf_data[close_col].iloc[-1],
                    'price_change_24h': tf_data[close_col].pct_change(24).iloc[-1] if len(tf_data) > 24 else None,
                    'volatility': tf_data[close_col].rolling(window=20).std().iloc[-1]
                }
        
        return summary

# Global instance
multi_timeframe = MultiTimeframeData()

def get_multi_timeframe_features(symbol, lookback_days=60):
    """
    Get multi-timeframe features for a symbol
    """
    return multi_timeframe.create_multi_timeframe_features(symbol, lookback_days)

def get_timeframe_summary(symbol, lookback_days=90):
    """
    Get timeframe summary for a symbol
    """
    return multi_timeframe.get_timeframe_summary(symbol, lookback_days) 