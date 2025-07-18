import pandas as pd
import numpy as np
from config import Config

class Pattern:
    def __init__(self, name, func, description=""):
        self.name = name
        self.func = func
        self.description = description

    def detect(self, df):
        return self.func(df)

# --- Pattern implementations ---
def doji(df):
    """Detect doji candlestick pattern."""
    open_col = [col for col in df.columns if col.startswith('Open')][0]
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    high_col = [col for col in df.columns if col.startswith('High')][0]
    low_col = [col for col in df.columns if col.startswith('Low')][0]
    return (abs(df[open_col] - df[close_col]) <= (df[high_col] - df[low_col]) * 0.1).astype(int)

def hammer(df):
    """Detect hammer candlestick pattern."""
    open_col = [col for col in df.columns if col.startswith('Open')][0]
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    high_col = [col for col in df.columns if col.startswith('High')][0]
    low_col = [col for col in df.columns if col.startswith('Low')][0]
    return ((df[close_col] > df[open_col]) &
            ((df[open_col] - df[low_col]) > 2 * abs(df[close_col] - df[open_col])) &
            ((df[high_col] - df[close_col]) < abs(df[close_col] - df[open_col]))).astype(int)

def bullish_engulfing(df):
    """Detect bullish engulfing candlestick pattern."""
    open_col = [col for col in df.columns if col.startswith('Open')][0]
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    return ((df[close_col].shift(1) < df[open_col].shift(1)) &
            (df[close_col] > df[open_col]) &
            (df[close_col] > df[open_col].shift(1)) &
            (df[open_col] < df[close_col].shift(1))).astype(int)

def bearish_engulfing(df):
    """Detect bearish engulfing candlestick pattern."""
    open_col = [col for col in df.columns if col.startswith('Open')][0]
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    return ((df[close_col].shift(1) > df[open_col].shift(1)) &
            (df[close_col] < df[open_col]) &
            (df[open_col] > df[close_col].shift(1)) &
            (df[close_col] < df[open_col].shift(1))).astype(int)

def shooting_star(df):
    """Detect shooting star candlestick pattern."""
    open_col = [col for col in df.columns if col.startswith('Open')][0]
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    high_col = [col for col in df.columns if col.startswith('High')][0]
    low_col = [col for col in df.columns if col.startswith('Low')][0]
    return ((df[open_col] > df[close_col]) &
            ((df[high_col] - df[open_col]) > 2 * abs(df[close_col] - df[open_col])) &
            ((df[open_col] - df[low_col]) < abs(df[close_col] - df[open_col]))).astype(int)

def pin_bar(df):
    """Detect pin bar price action pattern."""
    open_col = [col for col in df.columns if col.startswith('Open')][0]
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    high_col = [col for col in df.columns if col.startswith('High')][0]
    low_col = [col for col in df.columns if col.startswith('Low')][0]
    body = abs(df[close_col] - df[open_col])
    upper_wick = df[high_col] - df[[close_col, open_col]].max(axis=1)
    lower_wick = df[[close_col, open_col]].min(axis=1) - df[low_col]
    return ((body < (df[high_col] - df[low_col]) * 0.3) & ((upper_wick > 2 * body) | (lower_wick > 2 * body))).astype(int)

def bullish_engulfing_bar(df):
    """Detect bullish engulfing bar price action pattern."""
    open_col = [col for col in df.columns if col.startswith('Open')][0]
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    return ((df[close_col] > df[open_col]) & (df[close_col].shift(1) < df[open_col].shift(1)) & (df[close_col] > df[open_col].shift(1)) & (df[open_col] < df[close_col].shift(1))).astype(int)

def bearish_engulfing_bar(df):
    """Detect bearish engulfing bar price action pattern."""
    open_col = [col for col in df.columns if col.startswith('Open')][0]
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    return ((df[close_col] < df[open_col]) & (df[close_col].shift(1) > df[open_col].shift(1)) & (df[close_col] < df[open_col].shift(1)) & (df[open_col] > df[close_col].shift(1))).astype(int)

def inside_bar(df):
    """Detect inside bar price action pattern."""
    high_col = [col for col in df.columns if col.startswith('High')][0]
    low_col = [col for col in df.columns if col.startswith('Low')][0]
    return ((df[high_col] < df[high_col].shift(1)) & (df[low_col] > df[low_col].shift(1))).astype(int)

def head_shoulders(df):
    """Detect head & shoulders pattern."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    window = 20
    pattern = [0] * len(df)
    for i in range(window, len(df)):
        win = df[close_col].iloc[i-window:i].values
        if len(win) < window:
            continue
        mid = window // 2
        left = win[:mid]
        right = win[mid+1:]
        if win[mid] == max(win) and win[mid] > max(left) and win[mid] > max(right):
            pattern[i] = 1
    return pd.Series(pattern, index=df.index)

def inv_head_shoulders(df):
    """Detect inverse head & shoulders pattern."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    window = 20
    inv_pattern = [0] * len(df)
    for i in range(window, len(df)):
        win = df[close_col].iloc[i-window:i].values
        if len(win) < window:
            continue
        mid = window // 2
        left = win[:mid]
        right = win[mid+1:]
        if win[mid] == min(win) and win[mid] < min(left) and win[mid] < min(right):
            inv_pattern[i] = 1
    return pd.Series(inv_pattern, index=df.index)

def double_top(df):
    """Detect double top pattern."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    window = 20
    threshold = 0.001
    double_top = [0] * len(df)
    for i in range(window, len(df)):
        win = df[close_col].iloc[i-window:i].values
        if len(win) < window:
            continue
        peaks = (win == max(win)).nonzero()[0]
        if len(peaks) >= 2 and abs(peaks[0] - peaks[1]) > 2 and abs(win[peaks[0]] - win[peaks[1]]) < threshold * win[peaks[0]]:
            double_top[i] = 1
    return pd.Series(double_top, index=df.index)

def double_bottom(df):
    """Detect double bottom pattern."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    window = 20
    threshold = 0.001
    double_bottom = [0] * len(df)
    for i in range(window, len(df)):
        win = df[close_col].iloc[i-window:i].values
        if len(win) < window:
            continue
        troughs = (win == min(win)).nonzero()[0]
        if len(troughs) >= 2 and abs(troughs[0] - troughs[1]) > 2 and abs(win[troughs[0]] - win[troughs[1]]) < threshold * win[troughs[0]]:
            double_bottom[i] = 1
    return pd.Series(double_bottom, index=df.index)

def rising_wedge(df):
    """Detect rising wedge pattern."""
    high_col = [col for col in df.columns if col.startswith('High')][0]
    low_col = [col for col in df.columns if col.startswith('Low')][0]
    window = 20
    rising_wedge = [0] * len(df)
    for i in range(window, len(df)):
        highs = df[high_col].iloc[i-window:i].values
        lows = df[low_col].iloc[i-window:i].values
        if len(highs) < window or len(lows) < window:
            continue
        if np.polyfit(range(window), highs, 1)[0] > 0 and np.polyfit(range(window), lows, 1)[0] > 0:
            if np.polyfit(range(window), highs, 1)[0] < np.polyfit(range(window), lows, 1)[0]:
                rising_wedge[i] = 1
    return pd.Series(rising_wedge, index=df.index)

def falling_wedge(df):
    """Detect falling wedge pattern."""
    high_col = [col for col in df.columns if col.startswith('High')][0]
    low_col = [col for col in df.columns if col.startswith('Low')][0]
    window = 20
    falling_wedge = [0] * len(df)
    for i in range(window, len(df)):
        highs = df[high_col].iloc[i-window:i].values
        lows = df[low_col].iloc[i-window:i].values
        if len(highs) < window or len(lows) < window:
            continue
        if np.polyfit(range(window), highs, 1)[0] < 0 and np.polyfit(range(window), lows, 1)[0] < 0:
            if np.polyfit(range(window), highs, 1)[0] > np.polyfit(range(window), lows, 1)[0]:
                falling_wedge[i] = 1
    return pd.Series(falling_wedge, index=df.index)

def fakeout_up(df):
    """Detect fakeout up (false breakout above high)."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    high_col = [col for col in df.columns if col.startswith('High')][0]
    low_col = [col for col in df.columns if col.startswith('Low')][0]
    window = 20
    fakeout_up = [0] * len(df)
    for i in range(window, len(df)):
        win_close = df[close_col].iloc[i-window:i].values
        win_high = df[high_col].iloc[i-window:i].values
        win_low = df[low_col].iloc[i-window:i].values
        if len(win_close) < window:
            continue
        if win_close[-2] > max(win_high[:-2]) and win_close[-1] < max(win_high[:-2]):
            fakeout_up[i] = 1
    return pd.Series(fakeout_up, index=df.index)

def fakeout_down(df):
    """Detect fakeout down (false breakout below low)."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    high_col = [col for col in df.columns if col.startswith('High')][0]
    low_col = [col for col in df.columns if col.startswith('Low')][0]
    window = 20
    fakeout_down = [0] * len(df)
    for i in range(window, len(df)):
        win_close = df[close_col].iloc[i-window:i].values
        win_high = df[high_col].iloc[i-window:i].values
        win_low = df[low_col].iloc[i-window:i].values
        if len(win_close) < window:
            continue
        if win_close[-2] < min(win_low[:-2]) and win_close[-1] > min(win_low[:-2]):
            fakeout_down[i] = 1
    return pd.Series(fakeout_down, index=df.index)

def bullish_flag(df):
    """Detect bullish flag: sharp up move followed by downward-sloping consolidation."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    window = 20
    flag = [0] * len(df)
    for i in range(window, len(df)):
        win = df[close_col].iloc[i-window:i].values
        # Sharp up move
        if win[-10] < win[-20] and win[-1] > win[-10]:
            # Downward-sloping consolidation
            if np.polyfit(range(10), win[-10:], 1)[0] < 0:
                flag[i] = 1
    return pd.Series(flag, index=df.index)

def bearish_flag(df):
    """Detect bearish flag: sharp down move followed by upward-sloping consolidation."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    window = 20
    flag = [0] * len(df)
    for i in range(window, len(df)):
        win = df[close_col].iloc[i-window:i].values
        # Sharp down move
        if win[-10] > win[-20] and win[-1] < win[-10]:
            # Upward-sloping consolidation
            if np.polyfit(range(10), win[-10:], 1)[0] > 0:
                flag[i] = 1
    return pd.Series(flag, index=df.index)

def bullish_pennant(df):
    """Detect bullish pennant: sharp up move followed by converging triangle consolidation."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    window = 20
    pennant = [0] * len(df)
    for i in range(window, len(df)):
        win = df[close_col].iloc[i-window:i].values
        # Sharp up move
        if win[-10] < win[-20] and win[-1] > win[-10]:
            # Triangle consolidation: std dev drops, highs and lows converge
            if np.std(win[-10:]) < np.std(win[-20:-10]):
                pennant[i] = 1
    return pd.Series(pennant, index=df.index)

def bearish_pennant(df):
    """Detect bearish pennant: sharp down move followed by converging triangle consolidation."""
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    window = 20
    pennant = [0] * len(df)
    for i in range(window, len(df)):
        win = df[close_col].iloc[i-window:i].values
        # Sharp down move
        if win[-10] > win[-20] and win[-1] < win[-10]:
            # Triangle consolidation: std dev drops, highs and lows converge
            if np.std(win[-10:]) < np.std(win[-20:-10]):
                pennant[i] = 1
    return pd.Series(pennant, index=df.index)

def bullish_engulfing_strength(df):
    """Quantify bullish engulfing pattern strength: 3=extra_strong, 2=strong, 1=moderate, 0=none."""
    open_col = [col for col in df.columns if col.startswith('Open')][0]
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    high_col = [col for col in df.columns if col.startswith('High')][0]
    prev_body = abs(df[close_col].shift(1) - df[open_col].shift(1))
    curr_body = abs(df[close_col] - df[open_col])
    # Basic engulfing
    engulf = (
        (df[close_col].shift(1) < df[open_col].shift(1)) &
        (df[close_col] > df[open_col]) &
        (df[close_col] > df[open_col].shift(1)) &
        (df[open_col] < df[close_col].shift(1))
    )
    # Strong: body ratio
    strong = engulf & (curr_body >= Config.PATTERN_STRENGTH_THRESHOLDS['engulfing']['body_ratio_strong'] * prev_body)
    # Moderate: body ratio
    moderate = engulf & (curr_body >= Config.PATTERN_STRENGTH_THRESHOLDS['engulfing']['body_ratio_moderate'] * prev_body)
    # Extra strong: close above previous high
    if Config.PATTERN_STRENGTH_THRESHOLDS['engulfing'].get('extra_strong_close_above_high', False):
        extra_strong = strong & (df[close_col] > df[high_col].shift(1))
    else:
        extra_strong = pd.Series([False]*len(df), index=df.index)
    return extra_strong.astype(int) * 3 + (strong.astype(int) * (1 - extra_strong.astype(int))) * 2 + (moderate.astype(int) * (1 - strong.astype(int)) * (1 - extra_strong.astype(int)))

def pin_bar_strength(df):
    """Quantify pin bar pattern strength: 2=strong, 1=moderate, 0=none. (Upgraded: require close in top/bottom 25% of range)"""
    open_col = [col for col in df.columns if col.startswith('Open')][0]
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    high_col = [col for col in df.columns if col.startswith('High')][0]
    low_col = [col for col in df.columns if col.startswith('Low')][0]
    body = abs(df[close_col] - df[open_col])
    upper_wick = df[high_col] - df[[close_col, open_col]].max(axis=1)
    lower_wick = df[[close_col, open_col]].min(axis=1) - df[low_col]
    rng = df[high_col] - df[low_col]
    close_pct = (df[close_col] - df[low_col]) / rng.replace(0, np.nan)
    extreme_pct = Config.PATTERN_STRENGTH_THRESHOLDS['pin_bar']['close_in_extreme_pct']
    # Bullish pin: close in top X% of range, long lower wick
    bullish_strong = (
        (body < rng * 0.3) &
        (lower_wick >= Config.PATTERN_STRENGTH_THRESHOLDS['pin_bar']['wick_body_ratio_strong'] * body) &
        (close_pct >= 1 - extreme_pct)
    )
    bullish_moderate = (
        (body < rng * 0.3) &
        (lower_wick >= Config.PATTERN_STRENGTH_THRESHOLDS['pin_bar']['wick_body_ratio_moderate'] * body) &
        (close_pct >= 1 - extreme_pct)
    )
    # Bearish pin: close in bottom X% of range, long upper wick
    bearish_strong = (
        (body < rng * 0.3) &
        (upper_wick >= Config.PATTERN_STRENGTH_THRESHOLDS['pin_bar']['wick_body_ratio_strong'] * body) &
        (close_pct <= extreme_pct)
    )
    bearish_moderate = (
        (body < rng * 0.3) &
        (upper_wick >= Config.PATTERN_STRENGTH_THRESHOLDS['pin_bar']['wick_body_ratio_moderate'] * body) &
        (close_pct <= extreme_pct)
    )
    strong = bullish_strong | bearish_strong
    moderate = bullish_moderate | bearish_moderate
    return strong.astype(int) * 2 + (moderate.astype(int) * (1 - strong.astype(int)))

def morning_star(df):
    open_col = [col for col in df.columns if col.startswith('Open')][0]
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    low_col = [col for col in df.columns if col.startswith('Low')][0]
    # 3-bar pattern: bearish, small body, bullish
    ms = [0] * len(df)
    for i in range(2, len(df)):
        prev = i - 2
        mid = i - 1
        curr = i
        if (df[close_col].iloc[prev] < df[open_col].iloc[prev] and
            abs(df[close_col].iloc[mid] - df[open_col].iloc[mid]) < 0.5 * abs(df[close_col].iloc[prev] - df[open_col].iloc[prev]) and
            df[close_col].iloc[curr] > df[open_col].iloc[curr] and
            df[close_col].iloc[curr] > df[open_col].iloc[prev]):
            ms[i] = 1
    return pd.Series(ms, index=df.index)

def evening_star(df):
    open_col = [col for col in df.columns if col.startswith('Open')][0]
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    high_col = [col for col in df.columns if col.startswith('High')][0]
    # 3-bar pattern: bullish, small body, bearish
    es = [0] * len(df)
    for i in range(2, len(df)):
        prev = i - 2
        mid = i - 1
        curr = i
        if (df[close_col].iloc[prev] > df[open_col].iloc[prev] and
            abs(df[close_col].iloc[mid] - df[open_col].iloc[mid]) < 0.5 * abs(df[close_col].iloc[prev] - df[open_col].iloc[prev]) and
            df[close_col].iloc[curr] < df[open_col].iloc[curr] and
            df[close_col].iloc[curr] < df[open_col].iloc[prev]):
            es[i] = 1
    return pd.Series(es, index=df.index)

def three_white_soldiers(df):
    open_col = [col for col in df.columns if col.startswith('Open')][0]
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    tws = [0] * len(df)
    for i in range(2, len(df)):
        if (df[close_col].iloc[i-2] > df[open_col].iloc[i-2] and
            df[close_col].iloc[i-1] > df[open_col].iloc[i-1] and
            df[close_col].iloc[i] > df[open_col].iloc[i] and
            df[close_col].iloc[i-1] > df[close_col].iloc[i-2] and
            df[close_col].iloc[i] > df[close_col].iloc[i-1]):
            tws[i] = 1
    return pd.Series(tws, index=df.index)

def three_black_crows(df):
    open_col = [col for col in df.columns if col.startswith('Open')][0]
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    tbc = [0] * len(df)
    for i in range(2, len(df)):
        if (df[close_col].iloc[i-2] < df[open_col].iloc[i-2] and
            df[close_col].iloc[i-1] < df[open_col].iloc[i-1] and
            df[close_col].iloc[i] < df[open_col].iloc[i] and
            df[close_col].iloc[i-1] < df[close_col].iloc[i-2] and
            df[close_col].iloc[i] < df[close_col].iloc[i-1]):
            tbc[i] = 1
    return pd.Series(tbc, index=df.index)

def bullish_harami(df):
    open_col = [col for col in df.columns if col.startswith('Open')][0]
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    bh = [0] * len(df)
    for i in range(1, len(df)):
        if (df[close_col].iloc[i-1] < df[open_col].iloc[i-1] and
            df[close_col].iloc[i] > df[open_col].iloc[i] and
            df[open_col].iloc[i] > df[close_col].iloc[i-1] and
            df[close_col].iloc[i] < df[open_col].iloc[i-1]):
            bh[i] = 1
    return pd.Series(bh, index=df.index)

def bearish_harami(df):
    open_col = [col for col in df.columns if col.startswith('Open')][0]
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    bh = [0] * len(df)
    for i in range(1, len(df)):
        if (df[close_col].iloc[i-1] > df[open_col].iloc[i-1] and
            df[close_col].iloc[i] < df[open_col].iloc[i] and
            df[open_col].iloc[i] < df[close_col].iloc[i-1] and
            df[close_col].iloc[i] > df[open_col].iloc[i-1]):
            bh[i] = 1
    return pd.Series(bh, index=df.index)

def dark_cloud_cover(df):
    open_col = [col for col in df.columns if col.startswith('Open')][0]
    close_col = [col for col in df.columns if col.startswith('Close')][0]
    high_col = [col for col in df.columns if col.startswith('High')][0]
    dcc = [0] * len(df)
    for i in range(1, len(df)):
        if (df[close_col].iloc[i-1] > df[open_col].iloc[i-1] and
            df[close_col].iloc[i] < df[open_col].iloc[i] and
            df[open_col].iloc[i] > df[close_col].iloc[i-1] and
            df[close_col].iloc[i] < (df[open_col].iloc[i-1] + df[close_col].iloc[i-1]) / 2):
            dcc[i] = 1
    return pd.Series(dcc, index=df.index)

# --- Pattern registry ---
PATTERNS = [
    Pattern('doji', doji, 'Doji candlestick'),
    Pattern('hammer', hammer, 'Hammer candlestick'),
    Pattern('bullish_engulfing', bullish_engulfing, 'Bullish engulfing'),
    Pattern('bearish_engulfing', bearish_engulfing, 'Bearish engulfing'),
    Pattern('shooting_star', shooting_star, 'Shooting star'),
    Pattern('pin_bar', pin_bar, 'Pin bar'),
    Pattern('bullish_engulfing_bar', bullish_engulfing_bar, 'Bullish engulfing bar'),
    Pattern('bearish_engulfing_bar', bearish_engulfing_bar, 'Bearish engulfing bar'),
    Pattern('inside_bar', inside_bar, 'Inside bar'),
    Pattern('head_shoulders', head_shoulders, 'Head & Shoulders'),
    Pattern('inv_head_shoulders', inv_head_shoulders, 'Inverse Head & Shoulders'),
    Pattern('double_top', double_top, 'Double Top'),
    Pattern('double_bottom', double_bottom, 'Double Bottom'),
    Pattern('rising_wedge', rising_wedge, 'Rising Wedge'),
    Pattern('falling_wedge', falling_wedge, 'Falling Wedge'),
    Pattern('fakeout_up', fakeout_up, 'Fakeout Up'),
    Pattern('fakeout_down', fakeout_down, 'Fakeout Down'),
    Pattern('bullish_flag', bullish_flag, 'Bullish Flag'),
    Pattern('bearish_flag', bearish_flag, 'Bearish Flag'),
    Pattern('bullish_pennant', bullish_pennant, 'Bullish Pennant'),
    Pattern('bearish_pennant', bearish_pennant, 'Bearish Pennant'),
    # --- New patterns ---
    Pattern('morning_star', morning_star, 'Morning Star'),
    Pattern('evening_star', evening_star, 'Evening Star'),
    Pattern('three_white_soldiers', three_white_soldiers, 'Three White Soldiers'),
    Pattern('three_black_crows', three_black_crows, 'Three Black Crows'),
    Pattern('bullish_harami', bullish_harami, 'Bullish Harami'),
    Pattern('bearish_harami', bearish_harami, 'Bearish Harami'),
    Pattern('dark_cloud_cover', dark_cloud_cover, 'Dark Cloud Cover'),
] 