import time
import schedule
from data.fetch_market import get_price_data, SYMBOL_MAP
from data.fetch_news import get_news_sentiment_with_cache
from data.preprocess import preprocess_features
from models.model_training_service import ModelTrainingService
from models.predict import predict_signal, get_signal_strength
from signals.signal_generator import generate_signal_output
from config import Config
from datetime import datetime, timedelta, date
from utils.notify import send_email
from utils.logger import get_logger
import pandas as pd
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from utils.attempt_log import AttemptLog
from data.data_fetcher import DataFetcher
from pipeline.pipeline_orchestrator import PipelineOrchestrator

logger = get_logger('main', log_file='logs/main.log')

PAIRS = ['USDJPY', 'BTCUSD', 'NZDJPY']
SIGNALS_CSV = 'logs/signals.csv'
ATTEMPT_LOG = 'logs/attempt_log.json'
MAX_ATTEMPTS = 30

# --- Session attempt times (UTC) ---
SESSION_ATTEMPT_TIMES = [
    '00:00', '01:30', '03:00',  # Asia
    '07:00', '08:30', '10:00',  # London
    '13:00', '14:30', '16:00',  # NY
    '22:00'                    # NY close
]

# --- Persistent attempt log ---
attempt_log = AttemptLog()
# --- Model training service ---
model_trainer = ModelTrainingService()
data_fetcher = DataFetcher()

def save_signal_to_csv(signal, features=None):
    logger.info(f'Saving signal to CSV: {signal}')
    fields = [
        'signal_id', 'time', 'pair', 'signal', 'trade_type', 'confidence', 'entry', 'stop_loss',
        'take_profit_1', 'take_profit_2', 'take_profit_3', 'latest_close', 'latest_high', 'latest_low', 'position_size', 'outcome'
    ]
    # Add features as a JSON string if provided
    row = {k: signal.get(k, None) for k in fields}
    row['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row['signal_id'] = f"{signal.get('pair','')}_{row['time'].replace(' ','T').replace(':','-')}"
    row['outcome'] = ''  # To be filled in by outcome tracker
    if features is not None:
        import json
        row['features'] = json.dumps(features)
        if 'features' not in fields:
            fields.append('features')
    df = pd.DataFrame([row])
    file_exists = os.path.exists(SIGNALS_CSV)
    if not file_exists:
        df.to_csv(SIGNALS_CSV, mode='w', index=False)
    else:
        df.to_csv(SIGNALS_CSV, mode='a', index=False, header=False)

def get_pair_keywords(pair):
    logger.info(f'Getting keywords for pair: {pair}')
    if pair == 'USDJPY':
        return ['USD/JPY', 'USDJPY', 'USD/JPY', 'USDJPY=X']
    elif pair == 'BTCUSD':
        return ['BTC/USD', 'BTCUSD', 'Bitcoin', 'BTC', 'BTC-USD']
    elif pair == 'USDCHF':
        return ['USD/CHF', 'USDCHF', 'CHF/USD', 'USDCHF=X']
    elif pair == 'NZDJPY':
        return ['NZD/JPY', 'NZDJPY', 'JPY/NZD', 'NZDJPY=X']
    else:
        return [pair]

def main():
    orchestrator = PipelineOrchestrator()
    orchestrator.run_forever(send_email_func=send_email)

if __name__ == "__main__":
    main()