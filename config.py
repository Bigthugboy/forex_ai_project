import os
from dotenv import load_dotenv
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("config_debug")

load_dotenv()  # Load environment variables from .env file

class Config:
    # API Keys
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    CRYPTOPANIC_API_KEY = os.getenv('CRYPTOPANIC_API_KEY')

    # Debug: Log whether keys are loaded
    logger.info(f"ALPHA_VANTAGE_API_KEY loaded: {'Yes' if ALPHA_VANTAGE_API_KEY else 'No'}")
    logger.info(f"NEWS_API_KEY loaded: {'Yes' if NEWS_API_KEY else 'No'}")
    logger.info(f"CRYPTOPANIC_API_KEY loaded: {'Yes' if CRYPTOPANIC_API_KEY else 'No'}")
    
    # Trading pairs
    TRADING_PAIRS = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD',
        'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY'
    ]
    
    # Risk management
    RISK_REWARD_RATIO = 3.0  # 3:1 profit to loss ratio
    DEFAULT_STOP_LOSS_PERCENTAGE = 1.0  # 1% stop loss
    DEFAULT_TAKE_PROFIT_PERCENTAGE = 3.0  # 3% take profit
    ACCOUNT_SIZE = 100.0  # USD
    RISK_PER_TRADE = 2.0  # USD risk per trade
    SL_PIPS = 20  # Default stop loss in pips (can be set between 20 and 30)
    MIN_LOT_SIZE = 0.02  # Minimum lot size
    MAX_LOT_SIZE = 0.05  # Maximum lot size
    
    # Model parameters
    LOOKBACK_PERIOD = 90  # days of historical data
    PREDICTION_HORIZON = 24  # hours ahead to predict
    
    # Technical indicators
    INDICATORS = {
        'sma_periods': [20, 50, 200],
        'ema_periods': [12, 26],
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bollinger_period': 20,
        'bollinger_std': 2,
        'stoch_k': 14,
        'stoch_d': 3,
        'atr_period': 14
    }
    
    # News sentiment analysis
    NEWS_KEYWORDS = [
        'USD', 'JPY', 'JPY/USD', 'USD/JPY', 'Bitcoin', 'BTC', 'cryptocurrency',
        'Federal Reserve', 'Bank of Japan', 'interest rates', 'inflation',
        'economic data', 'GDP', 'employment', 'trade balance'
    ]
    
    # Confluence factor weights (default, can be overridden by asset, volatility, or session)
    FACTOR_WEIGHTS = {
        'structure': 1.0,
        'key_level': 1.0,
        'pattern': 1.0,
        'rsi': 1.0,
        'macd': 1.0,
        'ema': 1.0,
        'atr': 1.0,
        'news': 1.0,
    }
    
    # Model file paths
    MODEL_DIR = 'models/'
    DATA_DIR = 'data/'
    LOGS_DIR = 'logs/'
    
    # Create directories if they don't exist
    @staticmethod
    def create_directories():
        for directory in [Config.MODEL_DIR, Config.DATA_DIR, Config.LOGS_DIR]:
            os.makedirs(directory, exist_ok=True) 

    # Key level proximity thresholds (for granular key level logic)
    KEY_LEVEL_PROXIMITY_THRESHOLDS = {
        'support': {'percent': 0.25, 'atr': 0.5},      # within 0.25% or 0.5 ATR
        'resistance': {'percent': 0.25, 'atr': 0.5},
        'supply': {'percent': 0.3, 'atr': 0.6},
        'demand': {'percent': 0.3, 'atr': 0.6},
        'fib': {'percent': 0.2, 'atr': 0.4},
    }

    # Pattern strength thresholds (objective quantification)
    PATTERN_STRENGTH_THRESHOLDS = {
        'engulfing': {
            'body_ratio_strong': 1.5,   # Body must be 1.5x previous candle
            'body_ratio_moderate': 1.1, # Body must be 1.1x previous candle
            'extra_strong_close_above_high': True
        },
        'pin_bar': {
            'wick_body_ratio_strong': 2.5,   # Wick at least 2.5x body
            'wick_body_ratio_moderate': 1.5,
            'close_in_extreme_pct': 0.25  # Close in top/bottom 25% of range
        },
        'flag': {
            'min_bars': 4,
            'max_angle': 30,  # degrees, for strong
            'min_retracement': 0.3, # as fraction of pole
        },
        # Add more as needed
    }

# --- Mailgun Notification Config ---
MAILGUN_API_KEY = os.getenv('MAILGUN_API_KEY')
MAILGUN_DOMAIN = os.getenv('MAILGUN_DOMAIN')
MAILGUN_SENDER = os.getenv('MAILGUN_SENDER')
MAILGUN_RECIPIENT = os.getenv('MAILGUN_RECIPIENT') 

# Remove constant features before model training
REMOVE_CONSTANT_FEATURES = True

# Always keep these features even if constant
ALWAYS_KEEP_FEATURES = [
    'doji', 'hammer', 'bullish_engulfing', 'bearish_engulfing', 'shooting_star',
    'pin_bar', 'bullish_engulfing_bar', 'bearish_engulfing_bar', 'inside_bar',
    'head_shoulders', 'inv_head_shoulders', 'double_top', 'double_bottom',
    'rising_wedge', 'falling_wedge', 'fakeout_up', 'fakeout_down',
    'bullish_flag', 'bearish_flag', 'bullish_pennant', 'bearish_pennant',
    'morning_star', 'evening_star', 'three_white_soldiers', 'three_black_crows',
    'bullish_harami', 'bearish_harami', 'dark_cloud_cover'
] 