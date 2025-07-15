import requests
from datetime import datetime, timedelta
from config import Config
import numpy as np

def classify_trade_type(latest_close, latest_high, latest_low, prediction):
    """
    Classify trade type based on prediction and price action.
    Returns: 'Instant', 'Buy Stop', 'Sell Stop', 'Buy Limit', 'Sell Limit'
    """
    # Simple logic: if prediction is BUY and price is near high, use Buy Stop; if near low, Buy Limit
    # If prediction is SELL and price is near low, use Sell Stop; if near high, Sell Limit
    range_ = latest_high - latest_low
    if prediction == 1:  # BUY
        if abs(latest_close - latest_high) < 0.2 * range_:
            return 'Buy Stop', latest_high + 0.1 * range_  # Entry above high
        elif abs(latest_close - latest_low) < 0.2 * range_:
            return 'Buy Limit', latest_low - 0.1 * range_   # Entry below low
        else:
            return 'Instant Execution', latest_close
    else:  # SELL
        if abs(latest_close - latest_low) < 0.2 * range_:
            return 'Sell Stop', latest_low - 0.1 * range_   # Entry below low
        elif abs(latest_close - latest_high) < 0.2 * range_:
            return 'Sell Limit', latest_high + 0.1 * range_ # Entry above high
        else:
            return 'Instant Execution', latest_close

def calculate_sl_tp(latest_close, prediction, entry=None):
    """
    Calculate Stop Loss (SL) and Take Profit (TP) using 3:1 reward-to-risk ratio from config.
    Returns: (SL, TP)
    """
    risk_pct = Config.DEFAULT_STOP_LOSS_PERCENTAGE / 100
    tp1_ratio = 2  # 2:1 for TP1
    tp2_ratio = 3  # 3:1 for TP2
    if entry is None:
        entry = latest_close
    if prediction == 1:  # BUY
        sl = entry * (1 - risk_pct)
        tp1 = entry * (1 + risk_pct * tp1_ratio)
        tp2 = entry * (1 + risk_pct * tp2_ratio)
    else:  # SELL
        sl = entry * (1 + risk_pct)
        tp1 = entry * (1 - risk_pct * tp1_ratio)
        tp2 = entry * (1 - risk_pct * tp2_ratio)
    return round(sl, 5), round(tp1, 5), round(tp2, 5)

def is_high_impact_event_near(pair, window_minutes=30):
    """
    Check if a high-impact economic event is within ±window_minutes for the pair's currencies.
    Returns True if an event is near, else False.
    """
    # Map pair to relevant currencies
    currency_map = {
        'USDJPY': ['USD', 'JPY'],
        'BTCUSD': ['USD'],
        'USDCHF': ['USD', 'CHF'],
        'JPYNZD': ['JPY', 'NZD'],
    }
    currencies = currency_map.get(pair, [])
    now = datetime.utcnow()
    from_date = now.strftime('%Y-%m-%d')
    to_date = (now + timedelta(days=1)).strftime('%Y-%m-%d')
    url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={from_date}&to={to_date}&apikey={Config.FMP_API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return False  # Fail open
        events = resp.json()
        for event in events:
            # Only consider high-impact events for relevant currencies
            if event.get('impact', '').lower() == 'high' and event.get('country'):
                for cur in currencies:
                    if cur in event.get('country') or cur in event.get('event', ''):
                        # Parse event time
                        event_time = event.get('date')
                        if event_time:
                            event_dt = datetime.strptime(event_time, '%Y-%m-%d %H:%M:%S')
                            if abs((event_dt - now).total_seconds()) <= window_minutes * 60:
                                return True
        return False
    except Exception as e:
        print(f"[Warning] Economic calendar API error: {e}")
        return False

def generate_signal_output(pair, features_df, prediction_result):
    """
    Generate full signal output including trade type, confidence, SL/TP, confluence, etc.
    Implements strict signal filtering: at least 4 confirming factors and confidence >= 0.8.
    """
    latest = features_df.iloc[-1]
    close_col = [col for col in features_df.columns if col.startswith('Close')][0]
    high_col = [col for col in features_df.columns if col.startswith('High')][0]
    low_col = [col for col in features_df.columns if col.startswith('Low')][0]
    latest_close = latest[close_col]
    latest_high = latest[high_col]
    latest_low = latest[low_col]
    prediction = prediction_result['prediction']
    confidence = prediction_result['confidence']
    # --- Confluence Calculation ---
    confluence_factors = []
    # 1. Trend alignment (trendline or moving average)
    if 'trendline_up' in latest and prediction == 1 and latest['trendline_up']:
        confluence_factors.append('trend_up')
    if 'trendline_down' in latest and prediction == 0 and latest['trendline_down']:
        confluence_factors.append('trend_down')
    # 2. Price action (breakout, engulfing, pin bar, etc.)
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
    # 3. Technical indicators (RSI, MACD, etc.)
    if 'rsi_14' in latest:
        if prediction == 1 and latest['rsi_14'] < 30:
            confluence_factors.append('rsi_oversold')
        if prediction == 0 and latest['rsi_14'] > 70:
            confluence_factors.append('rsi_overbought')
    if 'macd' in latest and 'macd_signal' in latest:
        if prediction == 1 and latest['macd'] > latest['macd_signal']:
            confluence_factors.append('macd_bull')
        if prediction == 0 and latest['macd'] < latest['macd_signal']:
            confluence_factors.append('macd_bear')
    # 4. News sentiment (if available)
    if 'news_sentiment' in latest:
        if prediction == 1 and latest['news_sentiment'] > 0.2:
            confluence_factors.append('news_bull')
        if prediction == 0 and latest['news_sentiment'] < -0.2:
            confluence_factors.append('news_bear')
    # --- Strict filter: require at least 4 confirming factors and high confidence ---
    confluence_score = len(confluence_factors)
    if confluence_score < 4 or confidence < 0.8:
        return None  # Do not generate signal
    # --- News event filter: suppress signals near high-impact events ---
    if is_high_impact_event_near(pair, window_minutes=30):
        return None  # Suppress signal due to news event
    trade_type, entry = classify_trade_type(latest_close, latest_high, latest_low, prediction)
    sl, tp1, tp2 = calculate_sl_tp(latest_close, prediction, entry)
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
        'latest_close': latest_close,
        'latest_high': latest_high,
        'latest_low': latest_low,
        'probabilities': prediction_result['probabilities']
    }
