import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class Config:
    # API Keys
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    
    # Trading pairs
    TRADING_PAIRS = ['USDJPY', 'BTCUSD']
    
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
    LOOKBACK_PERIOD = 60  # days of historical data
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
    
    # Model file paths
    MODEL_DIR = 'models/'
    DATA_DIR = 'data/'
    LOGS_DIR = 'logs/'
    
    # Create directories if they don't exist
    @staticmethod
    def create_directories():
        for directory in [Config.MODEL_DIR, Config.DATA_DIR, Config.LOGS_DIR]:
            os.makedirs(directory, exist_ok=True) 

    FMP_API_KEY = os.getenv('FMP_API_KEY')

# --- Mailgun Notification Config ---
MAILGUN_API_KEY = os.getenv('MAILGUN_API_KEY')
MAILGUN_DOMAIN = os.getenv('MAILGUN_DOMAIN')
MAILGUN_SENDER = os.getenv('MAILGUN_SENDER')
MAILGUN_RECIPIENT = os.getenv('MAILGUN_RECIPIENT') 