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

def generate_signal_output(pair, features_df, prediction_result):
    """
    Generate full signal output including trade type, confidence, SL/TP, etc.
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
    trade_type, entry = classify_trade_type(latest_close, latest_high, latest_low, prediction)
    sl, tp1, tp2 = calculate_sl_tp(latest_close, prediction, entry)
    return {
        'pair': pair,
        'trade_type': trade_type,
        'entry': round(entry, 5),
        'signal': prediction_result['signal'],
        'confidence': confidence,
        'stop_loss': sl,
        'take_profit_1': tp1,
        'take_profit_2': tp2,
        'latest_close': latest_close,
        'latest_high': latest_high,
        'latest_low': latest_low,
        'probabilities': prediction_result['probabilities']
    }
