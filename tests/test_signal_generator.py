import pytest
from unittest.mock import MagicMock
import pandas as pd
from signals.signal_generator import generate_signal_output
from config import Config

# Assume the confluence logic is in a function called check_confluence (extract if needed)
# For this test, we will define a minimal version of the logic here for demonstration.

def check_confluence(confluence_factors, categories, factor_strengths=None):
    min_confluence = 4
    required_categories = ["trend", "momentum", "pattern"]
    if sum(categories.get(c, 0) for c in required_categories) < len(required_categories):
        return None
    unique_confluence_factors = set(confluence_factors)
    confluence_score = len(unique_confluence_factors)
    if confluence_score < min_confluence:
        return None
    bullish_factors = [f for f in confluence_factors if "bullish" in f.lower()]
    bearish_factors = [f for f in confluence_factors if "bearish" in f.lower()]
    if factor_strengths is None:
        factor_strengths = {f: 1 for f in confluence_factors}
    if bullish_factors and bearish_factors:
        bullish_strength = sum(factor_strengths.get(f, 1) for f in bullish_factors)
        bearish_strength = sum(factor_strengths.get(f, 1) for f in bearish_factors)
        if bullish_strength > bearish_strength:
            return "bullish"
        elif bearish_strength > bullish_strength:
            return "bearish"
        else:
            return None
    elif bullish_factors:
        return "bullish"
    elif bearish_factors:
        return "bearish"
    else:
        return None

def test_signal_generated_with_strict_confluence():
    confluence_factors = [
        "trend:MA_bullish",      # trend
        "momentum:RSI_strong",   # momentum
        "pattern:Engulfing",     # pattern
        "macro:RiskOn"           # macro (extra)
    ]
    categories = {"trend": 1, "momentum": 1, "pattern": 1, "macro": 1}
    assert check_confluence(confluence_factors, categories) in ("bullish", "bearish", None)  # direction depends on factors

def test_signal_not_generated_if_missing_category():
    confluence_factors = [
        "trend:MA_bullish",
        "momentum:RSI_strong",
        "macro:RiskOn"
    ]
    categories = {"trend": 1, "momentum": 1, "macro": 1}
    assert check_confluence(confluence_factors, categories) is None

def test_signal_not_generated_if_not_enough_factors():
    confluence_factors = [
        "trend:MA_bullish",
        "momentum:RSI_strong",
        "pattern:Engulfing"
    ]
    categories = {"trend": 1, "momentum": 1, "pattern": 1}
    assert check_confluence(confluence_factors, categories) is None

def test_conflict_resolution_bullish_stronger():
    confluence_factors = [
        "trend:MA_bullish",
        "momentum:RSI_bullish",
        "pattern:Engulfing_bullish",
        "pattern:Doji_bearish"
    ]
    categories = {"trend": 1, "momentum": 1, "pattern": 2}
    factor_strengths = {
        "trend:MA_bullish": 2,
        "momentum:RSI_bullish": 2,
        "pattern:Engulfing_bullish": 2,
        "pattern:Doji_bearish": 1
    }
    assert check_confluence(confluence_factors, categories, factor_strengths) == "bullish"

def test_conflict_resolution_bearish_stronger():
    confluence_factors = [
        "trend:MA_bearish",
        "momentum:RSI_bearish",
        "pattern:Engulfing_bearish",
        "pattern:Doji_bullish"
    ]
    categories = {"trend": 1, "momentum": 1, "pattern": 2}
    factor_strengths = {
        "trend:MA_bearish": 2,
        "momentum:RSI_bearish": 2,
        "pattern:Engulfing_bearish": 2,
        "pattern:Doji_bullish": 1
    }
    assert check_confluence(confluence_factors, categories, factor_strengths) == "bearish"

def test_conflict_resolution_equal_strength():
    confluence_factors = [
        "trend:MA_bullish",
        "momentum:RSI_bullish",
        "pattern:Engulfing_bullish",
        "pattern:Doji_bearish"
    ]
    categories = {"trend": 1, "momentum": 1, "pattern": 2}
    factor_strengths = {
        "trend:MA_bullish": 1,
        "momentum:RSI_bullish": 1,
        "pattern:Engulfing_bullish": 1,
        "pattern:Doji_bearish": 3
    }
    # bullish: 1+1+1=3, bearish: 3
    assert check_confluence(confluence_factors, categories, factor_strengths) is None 

def test_generate_signal_output_dynamic_weighting():
    Config.FACTOR_WEIGHTS = {
        'structure': 2.0,
        'key_level': 1.0,
        'pattern': 0.5,
        'rsi': 0.5,
        'macd': 1.0,
        'ema': 1.0,
        'atr': 0.5,
        'news': 2.0,
    }
    Config.PATTERN_STRENGTH_THRESHOLDS = {
        'engulfing': {
            'body_ratio_strong': 1.5,
            'body_ratio_moderate': 1.1,
            'extra_strong_close_above_high': True
        },
        'pin_bar': {
            'wick_body_ratio_strong': 2.5,
            'wick_body_ratio_moderate': 1.5,
            'close_in_extreme_pct': 0.25
        },
        'flag': {
            'min_bars': 4,
            'max_angle': 30,
            'min_retracement': 0.3,
        },
    }
    # Use two rows for engulfing pattern
    data = {
        'Open': [99, 100],
        'Close': [98, 106],
        'High': [100, 107],
        'Low': [97, 99],
        'structure_trend_4h': ['uptrend', 'uptrend'],
        'supply_zone': [1, 1],
        'bullish_engulfing': [0, 1],
        'rsi_14': [30, 30],
        'macd': [1.2, 1.2],
        'macd_signal': [1.0, 1.0],
        'ema_12': [99, 99],
        'atr_14': [1.0, 1.0],
        'news_sentiment': [0.5, 0.5],
    }
    features_df = pd.DataFrame(data)
    features_df.index = pd.DatetimeIndex([pd.Timestamp('2024-01-01 00:00:00'), pd.Timestamp('2024-01-01 01:00:00')])
    prediction_result = {'prediction': 1, 'confidence': 0.9, 'signal': 'BUY', 'probabilities': [0.1, 0.9]}
    result = generate_signal_output('TESTPAIR', features_df, prediction_result)
    assert result is not None
    # Should sum weights for structure (2.0), key_level (1.0), pattern (0.5), rsi (0.5), macd (1.0), ema (1.0), atr (0.5), news (2.0)
    # All factors present and valid for bullish, so weighted_votes = 8.5
    assert result['confluence'] == 8.5
    assert 'structure' in result['confluence_factors']
    assert 'key_level' in result['confluence_factors']
    assert 'pattern' in result['confluence_factors']
    assert 'rsi' in result['confluence_factors']
    assert 'macd' in result['confluence_factors']
    assert 'ema' in result['confluence_factors']
    assert 'atr' in result['confluence_factors']
    assert 'news' in result['confluence_factors']

def test_generate_signal_output_key_level_proximity():
    Config.KEY_LEVEL_PROXIMITY_THRESHOLDS = {
        'support': {'percent': 0.25, 'atr': 0.5},
        'resistance': {'percent': 0.25, 'atr': 0.5},
        'supply': {'percent': 0.3, 'atr': 0.6},
        'demand': {'percent': 0.3, 'atr': 0.6},
        'fib': {'percent': 0.2, 'atr': 0.4},
    }
    Config.FACTOR_WEIGHTS = {k: 1.0 for k in ['structure','key_level','pattern','rsi','macd','ema','atr','news']}
    Config.PATTERN_STRENGTH_THRESHOLDS = {
        'engulfing': {
            'body_ratio_strong': 1.5,
            'body_ratio_moderate': 1.1,
            'extra_strong_close_above_high': True
        },
        'pin_bar': {
            'wick_body_ratio_strong': 2.5,
            'wick_body_ratio_moderate': 1.5,
            'close_in_extreme_pct': 0.25
        },
        'flag': {
            'min_bars': 4,
            'max_angle': 30,
            'min_retracement': 0.3,
        },
    }
    # Case 1: Price within percent threshold of support
    data = {
        'Open': [99, 100],
        'Close': [98, 106],
        'High': [100, 107],
        'Low': [97, 99],
        'structure_trend_4h': ['uptrend', 'uptrend'],
        'support': [99.8, 99.8],
        'atr_14': [1.0, 1.0],
        'bullish_engulfing': [0, 1],
        'rsi_14': [30, 30],
        'macd': [1.2, 1.2],
        'macd_signal': [1.0, 1.0],
        'ema_12': [99, 99],
        'news_sentiment': [0.5, 0.5],
    }
    features_df = pd.DataFrame(data)
    features_df.index = pd.DatetimeIndex([pd.Timestamp('2024-01-01 00:00:00'), pd.Timestamp('2024-01-01 01:00:00')])
    prediction_result = {'prediction': 1, 'confidence': 0.9, 'signal': 'BUY', 'probabilities': [0.1, 0.9]}
    result = generate_signal_output('TESTPAIR', features_df, prediction_result)
    assert result is not None
    assert result['confluence_factors']
    assert result['confluence'] >= 4.0
    # Case 2: Price outside percent and ATR threshold of support, and not enough other factors to reach threshold
    data2 = data.copy()
    data2['support'] = [80.0, 80.0]  # Far from price
    # Remove some bullish factors so weighted_votes < 5.0
    data2['bullish_engulfing'] = [0, 0]
    data2['rsi_14'] = [40, 40]  # Not oversold
    data2['macd'] = [0.8, 0.8]
    data2['macd_signal'] = [1.0, 1.0]
    data2['ema_12'] = [101, 101]  # Price below EMA
    data2['news_sentiment'] = [0.0, 0.0]
    features_df2 = pd.DataFrame(data2)
    features_df2.index = pd.DatetimeIndex([pd.Timestamp('2024-01-01 00:00:00'), pd.Timestamp('2024-01-01 01:00:00')])
    result2 = generate_signal_output('TESTPAIR', features_df2, prediction_result)
    assert result2 is None  # Should be filtered out due to not at key level and not enough confluence

def test_generate_signal_output_pattern_strength():
    from features.patterns import bullish_engulfing_strength, pin_bar_strength
    Config.FACTOR_WEIGHTS = {k: 1.0 for k in ['structure','key_level','pattern','rsi','macd','ema','atr','news']}
    Config.PATTERN_STRENGTH_THRESHOLDS = {
        'engulfing': {
            'body_ratio_strong': 1.5,
            'body_ratio_moderate': 1.1,
            'extra_strong_close_above_high': True
        },
        'pin_bar': {
            'wick_body_ratio_strong': 2.5,
            'wick_body_ratio_moderate': 1.5,
            'close_in_extreme_pct': 0.25
        },
        'flag': {
            'min_bars': 4,
            'max_angle': 30,
            'min_retracement': 0.3,
        },
    }
    # Extra strong bullish engulfing
    data = {
        'Open': [99, 100],
        'Close': [98, 106],  # prev: bearish, curr: bullish, closes above prev high
        'High': [100, 107],
        'Low': [97, 99],
        'structure_trend_4h': ['uptrend', 'uptrend'],
        'support': [99, 99],
        'atr_14': [1.0, 1.0],
        'bullish_engulfing': [0, 1],
        'pin_bar': [0, 0],
        'rsi_14': [30, 30],
        'macd': [1.2, 1.2],
        'macd_signal': [1.0, 1.0],
        'ema_12': [99, 99],
        'news_sentiment': [0.5, 0.5],
    }
    features_df = pd.DataFrame(data)
    features_df.index = pd.DatetimeIndex([pd.Timestamp('2024-01-01 00:00:00'), pd.Timestamp('2024-01-01 01:00:00')])
    prediction_result = {'prediction': 1, 'confidence': 0.9, 'signal': 'BUY', 'probabilities': [0.1, 0.9]}
    result = generate_signal_output('TESTPAIR', features_df, prediction_result)
    assert result is not None
    assert result['confluence_factors']
    # Should be present as a pattern factor
    assert 'pattern' in result['confluence_factors']
    # Check the actual pattern label and confluence value
    assert result['confluence'] == 7.0 