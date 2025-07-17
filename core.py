import os
import pickle
import pandas as pd
from data.preprocess import preprocess_features
from data.fetch_market import get_price_data, SYMBOL_MAP
from data.fetch_news import get_news_sentiment_with_cache
from models.train_model import prepare_target_variable, train_signal_model
from models.predict import predict_signal
from signals.signal_generator import generate_signal_output
from models.continuous_learning import continuous_learner
from config import Config
from datetime import datetime, date
from utils.logger import get_logger
import json
import numpy as np
from utils.attempt_log import AttemptLog
from models.model_training_service import ModelTrainingService
from data.data_fetcher import DataFetcher
from pipeline.pipeline_orchestrator import PipelineOrchestrator
from utils.notify import send_email

logger = get_logger('core', log_file='logs/core.log')

ALL_PAIRS = ['USDJPY', 'BTCUSD', 'USDCHF', 'NZDJPY']
SIGNALS_CSV = 'logs/signals.csv'
ATTEMPT_LOG = 'logs/attempt_log.json'
MAX_ATTEMPTS = 30  # Updated as per your request

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
    row = {k: signal.get(k, None) for k in fields}
    row['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row['signal_id'] = f"{signal.get('pair','')}_{row['time'].replace(' ','T').replace(':','-')}"
    row['outcome'] = ''
    if features is not None:
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
        return ['NZD/JPY', 'NZDJPY', 'NZDJPY=X']
    else:
        return [pair]

def load_model_and_scaler():
    """Load the trained model and scaler from disk."""
    model_path = 'models/saved_models/signal_model.pkl'
    scaler_path = 'models/saved_models/scaler.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


def main():
    orchestrator = PipelineOrchestrator()
    orchestrator.run_session(send_email_func=send_email)
    orchestrator.run_continuous_learning()

if __name__ == "__main__":
    main() 