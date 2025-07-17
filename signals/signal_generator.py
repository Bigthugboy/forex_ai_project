import requests
from datetime import datetime, timedelta
from config import Config
import numpy as np
import os
from utils.logger import get_logger
import json
from data.fetch_news import fetch_economic_calendar
import dateutil.parser
logger = get_logger('signal_generator', log_file='logs/signal_generator.log')

def classify_trade_type(latest_close, latest_high, latest_low, prediction):
    logger.info('Classifying trade type...')
    """
    Classify trade type based on prediction and price action.
    Returns: 'Instant', 'Buy Stop', 'Sell Stop', 'Buy Limit', 'Sell Limit'
    """
    # Simple logic: if prediction is BUY and price is near high, use Buy Stop; if near low, Buy Limit
    # If prediction is SELL and price is near low, use Sell Stop; if near high, Sell Limit
    range_ = latest_high - latest_low
    if prediction == 1:  # BUY
        if abs(latest_close - latest_high) < 0.2 * range_:
            trade_type = 'Buy Stop'
            entry = latest_high + 0.1 * range_  # Entry above high
        elif abs(latest_close - latest_low) < 0.2 * range_:
            trade_type = 'Buy Limit'
            entry = latest_low - 0.1 * range_   # Entry below low
        else:
            trade_type = 'Instant Execution'
            entry = latest_close
    else:  # SELL
        if abs(latest_close - latest_low) < 0.2 * range_:
            trade_type = 'Sell Stop'
            entry = latest_low - 0.1 * range_   # Entry below low
        elif abs(latest_close - latest_high) < 0.2 * range_:
            trade_type = 'Sell Limit'
            entry = latest_high + 0.1 * range_ # Entry above high
        else:
            trade_type = 'Instant Execution'
            entry = latest_close
    logger.info(f'Trade type classified: {trade_type}, Entry: {entry}')
    return trade_type, entry

def calculate_pip_size(pair):
    # For JPY pairs, pip = 0.01; for others, pip = 0.0001
    if 'JPY' in pair:
        return 0.01
    else:
        return 0.0001

def calculate_sl_tp_pip(entry, prediction, pair):
    """
    Always use 30 pips for stop loss, 60 pips for take profit 1, and 2:1, 3:1, 4:1 reward:risk for TP2/TP3 (i.e., 60, 90, 120 pips from entry).
    Applies to all pairs (forex and crypto, pip size auto-calculated).
    """
    pip_size = calculate_pip_size(pair)
    sl_pips = 30
    tp1_pips = 60
    tp2_pips = 90
    tp3_pips = 120
    if prediction == 1:  # BUY
        sl = entry - (sl_pips * pip_size)
        tp1 = entry + (tp1_pips * pip_size)
        tp2 = entry + (tp2_pips * pip_size)
        tp3 = entry + (tp3_pips * pip_size)
    else:  # SELL
        sl = entry + (sl_pips * pip_size)
        tp1 = entry - (tp1_pips * pip_size)
        tp2 = entry - (tp2_pips * pip_size)
        tp3 = entry - (tp3_pips * pip_size)
    return round(sl, 5), round(tp1, 5), round(tp2, 5), round(tp3, 5)

def calculate_position_size(entry, sl, risk_per_trade, min_lot, max_lot, pair):
    pip_size = calculate_pip_size(pair)
    sl_distance = abs(entry - sl)
    if sl_distance == 0:
        return min_lot
    # For USDJPY, pip value per 0.01 lot is about $0.09 per pip
    # For simplicity, pip_value_per_lot = pip_size * 100000 (standard lot)
    pip_value_per_lot = pip_size * 100000
    # For 0.01 lot, pip value = pip_value_per_lot * 0.01
    # Position size in lots = risk_per_trade / (sl_pips * pip_value_per_lot)
    # But since sl_distance = sl_pips * pip_size, we can use:
    position_size = risk_per_trade / (sl_distance * pip_value_per_lot)
    position_size = max(min_lot, min(max_lot, position_size))
    return round(position_size, 4)

def is_high_impact_event_near(pair, window_minutes=30, latest_time=None):
    """
    Returns True if a high-impact economic event is within window_minutes of latest_time for the given pair.
    """
    logger.info(f'Checking for high-impact news event for {pair}...')
    events = fetch_economic_calendar(pair)
    if not events or latest_time is None:
        return False
    for event in events:
        # Try to parse event date/time
        event_time = None
        if 'date' in event:
            try:
                event_time = dateutil.parser.parse(event['date'])
            except Exception:
                continue
        if event_time is None:
            continue
        # Check for high impact
        impact = event.get('impact', '').lower()
        if 'high' in impact:
            delta = abs((event_time - latest_time).total_seconds()) / 60.0
            if delta <= window_minutes:
                logger.info(f"[BLOCKED] {pair} | High-impact event '{event.get('title','')}' at {event_time} within {window_minutes} min of {latest_time}.")
                return True
    return False

def regime_adaptive_features(latest, regime, all_features):
    # Define regime-specific indicator sets
    trend_indicators = [f for f in all_features if any(k in f for k in ['ma', 'ema', 'macd', 'atr', 'trend'])]
    range_indicators = [f for f in all_features if any(k in f for k in ['rsi', 'bb', 'boll', 'range', 'volatility'])]
    # Fallback: if none found, use all_features
    if regime == 'trend' and trend_indicators:
        logger.info(f"Regime is TREND. Using indicators: {trend_indicators}")
        return trend_indicators
    elif regime == 'range' and range_indicators:
        logger.info(f"Regime is RANGE. Using indicators: {range_indicators}")
        return range_indicators
    else:
        logger.info(f"Regime is {regime.upper()}. Using all available features: {all_features}")
        return all_features

def detect_ambiguity_conflict(latest, confluence_factors):
    # Simple logic: if both bullish and bearish factors present, or if confluence is split
    bullish = any(f in confluence_factors for f in ['trend_up', 'breakout_high', 'bullish_engulfing', 'rsi_oversold', 'macd_bull', 'news_bull'])
    bearish = any(f in confluence_factors for f in ['trend_down', 'breakout_low', 'bearish_engulfing', 'rsi_overbought', 'macd_bear', 'news_bear'])
    if bullish and bearish:
        logger.info('Ambiguity detected: both bullish and bearish confluence factors present.')
        return True
    return False

def generate_signal_output(pair, features_df, prediction_result):
    latest = features_df.iloc[-1]
    latest_time = getattr(latest, 'name', 'N/A')
    close_col = [col for col in features_df.columns if col.startswith('Close')][0]
    high_col = [col for col in features_df.columns if col.startswith('High')][0]
    low_col = [col for col in features_df.columns if col.startswith('Low')][0]
    latest_close = latest[close_col]
    latest_high = latest[high_col]
    latest_low = latest[low_col]
    prediction = prediction_result['prediction']
    confidence = prediction_result['confidence']
    
    # --- Indicator and pattern logging ---
    # Try to get RSI and MACD from different timeframes if not present in base
    rsi = None
    macd = None
    macd_signal = None
    
    # Check for RSI across all timeframes
    for col in features_df.columns:
        if 'rsi' in col.lower():
            rsi = latest[col]
            break
    
    # Check for MACD across all timeframes
    for col in features_df.columns:
        if 'macd' in col.lower() and 'signal' not in col.lower():
            macd = latest[col]
            break
    
    # Check for MACD signal across all timeframes
    for col in features_df.columns:
        if 'macd_signal' in col.lower():
            macd_signal = latest[col]
            break
    
    # Convert to string for logging
    rsi_str = str(rsi) if rsi is not None else 'N/A'
    macd_str = str(macd) if macd is not None else 'N/A'
    macd_signal_str = str(macd_signal) if macd_signal is not None else 'N/A'
    
    news_sentiment = latest.get('news_sentiment', 'N/A')
    if news_sentiment == 'N/A' or news_sentiment is None or (isinstance(news_sentiment, float) and np.isnan(news_sentiment)):
        logger.warning(f"News sentiment is N/A for {pair} at {latest_time}. No news found or error in fetching.")
    else:
        logger.info(f"News sentiment for {pair} at {latest_time}: {news_sentiment}")
    # --- Market regime logging ---
    market_regime = latest.get('market_regime', 'unknown')
    logger.info(f"Market regime for {pair} at {latest_time}: {market_regime}")
    logger.info(f"Analyzing {pair} at {latest_time}: RSI={rsi_str}, MACD={macd_str}, MACD_signal={macd_signal_str}, News sentiment={news_sentiment}")

    # --- Higher timeframe structure enforcement ---
    structure_4h = latest.get('structure_trend_4h', None)
    structure_1d = latest.get('structure_trend_1d', None)
    if structure_4h in ('uptrend', 'downtrend') or structure_1d in ('uptrend', 'downtrend'):
        higher_tf_trend = structure_4h if structure_4h in ('uptrend', 'downtrend') else structure_1d
        if higher_tf_trend == 'uptrend' and prediction == 0:
            logger.info(f"[FILTERED] {pair} @ {latest_time} | Signal is SHORT but higher timeframe ({'4h' if structure_4h else '1d'}) is UPTREND. Blocked by professional rule.")
            return None
        if higher_tf_trend == 'downtrend' and prediction == 1:
            logger.info(f"[FILTERED] {pair} @ {latest_time} | Signal is LONG but higher timeframe ({'4h' if structure_4h else '1d'}) is DOWNTREND. Blocked by professional rule.")
            return None

    # --- Blended Professional Confluence Logic ---
    factors = {}
    direction = 'bullish' if prediction == 1 else 'bearish'
    # 1. Structure (higher timeframe)
    structure_4h = latest.get('structure_trend_4h', None)
    structure_1d = latest.get('structure_trend_1d', None)
    structure_factor = None
    if structure_4h in ('uptrend', 'downtrend'):
        structure_factor = structure_4h
    elif structure_1d in ('uptrend', 'downtrend'):
        structure_factor = structure_1d
    if structure_factor == 'uptrend' and direction == 'bullish':
        factors['structure'] = 'bullish'
    elif structure_factor == 'downtrend' and direction == 'bearish':
        factors['structure'] = 'bearish'
    # 2. Key Level (granular proximity)
    key_level_cols = [
        'supply_zone', 'demand_zone', 'fib_0', 'fib_236', 'fib_382', 'fib_500', 'fib_618', 'fib_786', 'fib_100',
        'breakout_high', 'breakout_low', 'dist_to_high', 'dist_to_low', 'support', 'resistance'
    ]
    at_key_level = False
    proximity_log = {}
    atr_val = None
    # Try to get ATR for proximity checks
    for col in features_df.columns:
        if 'atr' in col.lower() and all(x not in col.lower() for x in ['_4h', '_1d', 'daily', 'week', 'month']):
            atr_val = latest[col]
            break
    for key_col in key_level_cols:
        if key_col in features_df.columns:
            key_val = latest.get(key_col, 0)
            # Direct hit (legacy logic)
            if key_col in ['supply_zone', 'demand_zone', 'breakout_high', 'breakout_low', 'support', 'resistance']:
                if key_val == 1:
                    at_key_level = True
                    factors['key_level'] = 'yes'
                    proximity_log[key_col] = 'direct hit'
                    break
            # Proximity logic for fibs, dist, support, resistance, supply, demand
            # Get reference price for proximity (support/resistance/fib value)
            if key_col.startswith('fib') or key_col in ['support', 'resistance', 'supply_zone', 'demand_zone']:
                # Assume value is the level, compare to latest_close
                level_val = key_val
                price = latest_close
                # Get thresholds
                if key_col.startswith('fib'):
                    thresholds = Config.KEY_LEVEL_PROXIMITY_THRESHOLDS.get('fib', {'percent': 0.2, 'atr': 0.4})
                elif key_col in ['support', 'resistance']:
                    thresholds = Config.KEY_LEVEL_PROXIMITY_THRESHOLDS.get(key_col, {'percent': 0.25, 'atr': 0.5})
                elif key_col in ['supply_zone', 'demand_zone']:
                    thresholds = Config.KEY_LEVEL_PROXIMITY_THRESHOLDS.get(key_col.replace('_zone',''), {'percent': 0.3, 'atr': 0.6})
                else:
                    thresholds = {'percent': 0.25, 'atr': 0.5}
                percent_dist = abs(price - level_val) / price * 100 if price else 0
                atr_dist = abs(price - level_val) / atr_val if atr_val else 0
                if percent_dist <= thresholds['percent'] or atr_dist <= thresholds['atr']:
                    at_key_level = True
                    factors['key_level'] = 'yes'
                    proximity_log[key_col] = f"within {percent_dist:.3f}% and {atr_dist:.3f} ATR"
                    break
                else:
                    proximity_log[key_col] = f"{percent_dist:.3f}% / {atr_dist:.3f} ATR away"
    if not at_key_level:
        factors['key_level'] = 'no'
    logger.info(f"[KEY LEVEL PROXIMITY] {pair} @ {latest_time} | Proximity details: {proximity_log}")
    # 3. Pattern (quantified strength)
    from features.patterns import bullish_engulfing_strength, pin_bar_strength
    pattern_strengths = {}
    # Bullish engulfing
    if 'bullish_engulfing' in features_df.columns:
        be_strength = bullish_engulfing_strength(features_df).iloc[-1]
        if be_strength == 3:
            factors['pattern'] = 'extra_strong_bullish'
            pattern_strengths['bullish_engulfing'] = 'extra_strong'
        elif be_strength == 2:
            factors['pattern'] = 'strong_bullish'
            pattern_strengths['bullish_engulfing'] = 'strong'
        elif be_strength == 1:
            factors['pattern'] = 'moderate_bullish'
            pattern_strengths['bullish_engulfing'] = 'moderate'
    # Pin bar
    if 'pin_bar' in features_df.columns:
        pb_strength = pin_bar_strength(features_df).iloc[-1]
        if pb_strength == 2:
            factors['pattern'] = 'strong_pin'
            pattern_strengths['pin_bar'] = 'strong'
        elif pb_strength == 1:
            factors['pattern'] = 'moderate_pin'
            pattern_strengths['pin_bar'] = 'moderate'
    # Legacy fallback: any pattern
    if 'pattern' not in factors:
        candlestick_patterns = [
            'doji', 'hammer', 'bullish_engulfing', 'bearish_engulfing', 'shooting_star',
            'bullish_flag', 'bearish_flag', 'bullish_pennant', 'bearish_pennant',
        ]
        pattern_values = {pattern: latest.get(pattern, 0) if pattern in features_df.columns else 0 for pattern in candlestick_patterns}
        detected_patterns = [pattern for pattern, val in pattern_values.items() if val]
        if detected_patterns:
            for pattern in detected_patterns:
                if direction in pattern:
                    factors['pattern'] = direction
                    break
            else:
                factors['pattern'] = 'other'
    logger.info(f"[PATTERN STRENGTH] {pair} @ {latest_time} | Pattern strengths: {pattern_strengths}")
    # Pattern weight adjustment for confluence
    pattern_weight = Config.FACTOR_WEIGHTS.get('pattern', 1.0)
    if 'pattern' in factors:
        if 'extra_strong' in factors['pattern']:
            pattern_weight = max(pattern_weight, 2.0)
        elif 'strong' in factors['pattern']:
            pattern_weight = max(pattern_weight, 1.5)
    # 4. RSI
    rsi_cols = [c for c in features_df.columns if 'rsi' in c.lower() and all(x not in c.lower() for x in ['_4h', '_1d', 'daily', 'week', 'month'])]
    for rsi_col in rsi_cols:
        rsi_val = latest[rsi_col]
        if direction == 'bullish' and rsi_val < 35:
            factors['rsi'] = 'bullish'
        if direction == 'bearish' and rsi_val > 65:
            factors['rsi'] = 'bearish'
    # 5. MACD
    macd_cols = [c for c in features_df.columns if 'macd' in c.lower() and 'signal' not in c.lower() and all(x not in c.lower() for x in ['_4h', '_1d', 'daily', 'week', 'month'])]
    macd_signal_cols = [c for c in features_df.columns if 'macd_signal' in c.lower() and all(x not in c.lower() for x in ['_4h', '_1d', 'daily', 'week', 'month'])]
    for macd_col, macd_signal_col in zip(macd_cols, macd_signal_cols):
        macd_val = latest[macd_col]
        macd_signal_val = latest[macd_signal_col]
        if direction == 'bullish' and macd_val > macd_signal_val:
            factors['macd'] = 'bullish'
        if direction == 'bearish' and macd_val < macd_signal_val:
            factors['macd'] = 'bearish'
    # 6. EMA
    ema_cols = [c for c in features_df.columns if 'ema' in c.lower() and all(x not in c.lower() for x in ['_4h', '_1d', 'daily', 'week', 'month'])]
    for ema_col in ema_cols:
        ema_val = latest[ema_col]
        if direction == 'bullish' and latest_close > ema_val:
            factors['ema'] = 'bullish'
        if direction == 'bearish' and latest_close < ema_val:
            factors['ema'] = 'bearish'
    # 7. ATR (volatility filter)
    atr_cols = [c for c in features_df.columns if 'atr' in c.lower() and all(x not in c.lower() for x in ['_4h', '_1d', 'daily', 'week', 'month'])]
    for atr_col in atr_cols:
        atr_val = latest[atr_col]
        # Use a simple filter: ATR not too high or low
        if 0.5 < atr_val < 3:
            factors['atr'] = 'normal'
    # 8. News
    news_sentiment = latest.get('news_sentiment', None)
    if news_sentiment is not None:
        if direction == 'bullish' and news_sentiment > 0.2:
            factors['news'] = 'bullish'
        if direction == 'bearish' and news_sentiment < -0.2:
            factors['news'] = 'bearish'
    # --- Count confluence (dynamic weighting, with pattern adjustment) ---
    factor_weights = getattr(Config, 'FACTOR_WEIGHTS', {})
    weighted_votes = 0.0
    contributing_factors = []
    for k, v in factors.items():
        if (v == direction or (k == 'key_level' and v == 'yes') or (k == 'atr' and v == 'normal')):
            weight = pattern_weight if k == 'pattern' else factor_weights.get(k, 1.0)
            weighted_votes += weight
            contributing_factors.append(k)
    logger.info(f"[CONFLUENCE] {pair} @ {latest_time} | Factors: {factors} | Weighted votes for {direction}: {weighted_votes}")
    # --- Signal rules (dynamic threshold) ---
    threshold = 4.0 if at_key_level else 5.0
    if weighted_votes >= threshold:
        logger.info(f"[SIGNAL] {pair} @ {latest_time} | {weighted_votes} weighted factors align. Signal allowed.")
    else:
        logger.info(f"[FILTERED] {pair} @ {latest_time} | Only {weighted_votes} weighted factors align. No signal generated. Factors: {factors}")
        return None
    # --- Economic event blocker ---
    if is_high_impact_event_near(pair, window_minutes=30, latest_time=latest_time):
        logger.info(f"[FILTERED] {pair} @ {latest_time} | Reason: High-impact news event. No signal generated.")
        return None
    # --- Continue with trade generation as before ---
    
    trade_type, entry = classify_trade_type(latest_close, latest_high, latest_low, prediction)
    
    # --- Use new risk management logic ---
    sl, tp1, tp2, tp3 = calculate_sl_tp_pip(entry, prediction, pair)
    position_size = calculate_position_size(entry, sl, Config.RISK_PER_TRADE, Config.MIN_LOT_SIZE, Config.MAX_LOT_SIZE, pair)
    # Assert SL/TP are within 0.1% to 10% of entry
    for val, name in zip([sl, tp1, tp2, tp3], ['SL', 'TP1', 'TP2', 'TP3']):
        pct = abs(val - entry) / max(1e-8, entry)
        if pct < 0.001 or pct > 0.10:
            logger.warning(f"{name} for {pair} at {latest_time} is {pct*100:.2f}% of entry, which is outside 0.1%-10% range.")
    logger.info(f"Dynamic position size: entry={entry}, sl={sl}, risk_per_trade={Config.RISK_PER_TRADE}, min_lot={Config.MIN_LOT_SIZE}, max_lot={Config.MAX_LOT_SIZE}, position_size={position_size}")
    # ---
    logger.info(f"Signal generated for {pair} at {latest_time}: {prediction_result['signal']} {trade_type} Confidence: {confidence:.2f}")
    return {
        'pair': pair,
        'trade_type': trade_type,
        'entry': round(entry, 5),
        'signal': prediction_result['signal'],
        'confidence': confidence,
        'confluence': weighted_votes, # Use the new weighted_votes
        'confluence_factors': contributing_factors, # Use the new contributing_factors
        'stop_loss': sl,
        'take_profit_1': tp1,
        'take_profit_2': tp2,
        'take_profit_3': tp3,
        'latest_close': latest_close,
        'latest_high': latest_high,
        'latest_low': latest_low,
        'probabilities': prediction_result['probabilities'],
        'position_size': round(position_size, 5)
    }
