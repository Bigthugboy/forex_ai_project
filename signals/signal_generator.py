import requests
from datetime import datetime, timedelta
from config import Config
import numpy as np
import os
from utils.logger import get_logger
import json
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

def calculate_sl_tp_pip(entry, prediction, pair, sl_pips):
    pip_size = calculate_pip_size(pair)
    if prediction == 1:  # BUY
        sl = entry - (sl_pips * pip_size)
        tp1 = entry + (sl_pips * 2 * pip_size)
        tp2 = entry + (sl_pips * 3 * pip_size)
        tp3 = entry + (sl_pips * 4 * pip_size)
    else:  # SELL
        sl = entry + (sl_pips * pip_size)
        tp1 = entry - (sl_pips * 2 * pip_size)
        tp2 = entry - (sl_pips * 3 * pip_size)
        tp3 = entry - (sl_pips * 4 * pip_size)
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

def is_high_impact_event_near(pair, window_minutes=30):
    logger.info(f'Checking for high-impact news event for {pair}... (Economic calendar check disabled, FMP API removed)')
    # TODO: Implement RSS-based economic calendar check
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
    patterns = [k for k in ['bullish_engulfing_bar', 'bearish_engulfing_bar', 'pin_bar'] if latest.get(k)]
    logger.info(f"Analyzing {pair} at {latest_time}: RSI={rsi_str}, MACD={macd_str}, MACD_signal={macd_signal_str}, News sentiment={news_sentiment}")
    logger.info(f"Detected patterns: {patterns if patterns else 'None'}")
    
    # --- Confluence Calculation ---
    confluence_factors = []
    
    # 1 Trend alignment (trendline or moving average)
    if 'trendline_up' in latest and prediction == 1 and latest['trendline_up']:
        confluence_factors.append('trend_up')
    if 'trendline_down' in latest and prediction == 0 and latest['trendline_down']:
        confluence_factors.append('trend_down')
    
    #2rice action (breakout, engulfing, pin bar, etc.)
    if prediction == 1 and ('breakout_high' in latest and latest['breakout_high']):
        confluence_factors.append('breakout_high')
    if prediction == 0 and ('breakout_low' in latest and latest['breakout_low']):
        confluence_factors.append('breakout_low')
    if prediction == 1 and ('bullish_engulfing_bar' in latest and latest['bullish_engulfing_bar']):
        confluence_factors.append('bullish_engulfing')
    if prediction == 0 and ('bearish_engulfing_bar' in latest and latest['bearish_engulfing_bar']):
        confluence_factors.append('bearish_engulfing')
    if 'pin_bar' in latest and latest['pin_bar']:
        confluence_factors.append('pin_bar')
    
    # 3. Technical indicators (RSI, MACD, etc.) - check across all timeframes
    if rsi is not None:
        if prediction == 1 and rsi < 30:
            confluence_factors.append('rsi_oversold')
        if prediction == 0 and rsi > 70:
            confluence_factors.append('rsi_overbought')
    
    if macd is not None and macd_signal is not None:
        if prediction == 1 and macd > macd_signal:
            confluence_factors.append('macd_bull')
        if prediction == 0 and macd < macd_signal:
            confluence_factors.append('macd_bear')
    
    #4. News sentiment (if available)
    if 'news_sentiment' in latest:
        if prediction == 1 and latest['news_sentiment'] > 0.2:
            confluence_factors.append('news_bull')
        if prediction == 0 and latest['news_sentiment'] < -0.2:
            confluence_factors.append('news_bear')
    
    # --- Strict filter: require at least 4 confirming factors and high confidence ---
    confluence_score = len(confluence_factors)
    logger.info(f"Confluence factors: {confluence_factors}, score={confluence_score}, confidence={confidence:.2f}")
    
    # --- Ambiguity/Conflict Detection ---
    ambiguous = detect_ambiguity_conflict(latest, confluence_factors)
    if ambiguous:
        logger.info(f"Signal for {pair} at {latest_time} flagged as ambiguous/conflicting. No signal generated.")
        return None
    
    # --- Granular Confidence Scoring ---
    logger.info(f"Signal confidence score: {confidence:.2f} (interpreted as probability of correctness)")
    if confluence_score < 4 or confidence < 0.75:
        logger.info(f"No signal for {pair} at {latest_time}: insufficient confluence ({confluence_score}) or confidence ({confidence:.2f})")
        # If confidence is not less than 0.65, keep studying the chart
        if confidence >= 0.65:
            logger.info(f"Model will keep studying the chart for {pair} at {latest_time}: confidence ({confidence:.2f}) not low enough to give up.")
        return None
    
    if is_high_impact_event_near(pair, window_minutes=30):
        logger.info(f"No signal for {pair} at {latest_time}: suppressed due to high-impact news event")
        return None
    
    trade_type, entry = classify_trade_type(latest_close, latest_high, latest_low, prediction)
    
    # --- Pip-based SL/TP and Dynamic Position Sizing ---
    sl_pips = min(max(Config.SL_PIPS, 20), 30)  # Clamp between 20 and 30
    sl, tp1, tp2, tp3 = calculate_sl_tp_pip(entry, prediction, pair, sl_pips)
    position_size = calculate_position_size(entry, sl, Config.RISK_PER_TRADE, Config.MIN_LOT_SIZE, Config.MAX_LOT_SIZE, pair)
    logger.info(f"Pip-based SL/TP: SL={sl}, TP1={tp1}, TP2={tp2}, TP3={tp3}, SL_PIPS={sl_pips}, Pip size={calculate_pip_size(pair)}")
    logger.info(f"Dynamic position size: entry={entry}, sl={sl}, risk_per_trade={Config.RISK_PER_TRADE}, min_lot={Config.MIN_LOT_SIZE}, max_lot={Config.MAX_LOT_SIZE}, position_size={position_size}")
    # ---
    logger.info(f"Signal generated for {pair} at {latest_time}: {prediction_result['signal']} {trade_type} Confidence: {confidence:.2f}")
    return {
        'pair': pair,
        'trade_type': trade_type,
        'entry': round(entry, 5),
        'signal': prediction_result['signal'],
        'confidence': confidence,
        'confluence': confluence_score,
        'confluence_factors': confluence_factors,
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
