import os
import pickle
import pandas as pd
from data.preprocess import preprocess_features
from data.fetch_market import get_price_data, SYMBOL_MAP
from data.fetch_news import get_news_sentiment_with_cache
from models.train_model import prepare_target_variable, train_signal_model
from models.predict import predict_signal
from signals.signal_generator import generate_signal_output
from config import Config
from datetime import datetime, date
from utils.logger import get_logger
import json
import numpy as np

logger = get_logger('core', log_file='logs/core.log')

ALL_PAIRS = ['USDJPY', 'BTCUSD', 'USDCHF', 'JPYNZD']
SIGNALS_CSV = 'logs/signals.csv'
ATTEMPT_LOG = 'logs/attempt_log.json'
MAX_ATTEMPTS = 30  # Updated as per your request

# --- Persistent attempt log ---
def load_attempt_log():
    if os.path.exists(ATTEMPT_LOG):
        with open(ATTEMPT_LOG, 'r') as f:
            return json.load(f)
    return {}

def save_attempt_log(log):
    os.makedirs(os.path.dirname(ATTEMPT_LOG), exist_ok=True)
    with open(ATTEMPT_LOG, 'w') as f:
        json.dump(log, f)

def reset_attempts_if_new_day(log):
    today = str(date.today())
    if today not in log:
        log.clear()
        log[today] = {}
    return log

def increment_attempt(log, pair):
    today = str(date.today())
    if today not in log:
        log[today] = {}
    log[today][pair] = log[today].get(pair, 0) + 1
    save_attempt_log(log)

def get_attempts(log, pair):
    today = str(date.today())
    return log.get(today, {}).get(pair, 0)

def reset_attempts_for_pair(log, pair):
    today = str(date.today())
    if today in log and pair in log[today]:
        log[today][pair] = 0
        save_attempt_log(log)

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
    elif pair == 'JPYNZD':
        return ['JPY/NZD', 'JPYNZD', 'NZD/JPY', 'NZDJPY=X']
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


def run_market_analysis():
    """
    Run market analysis for all pairs and return the latest signals and summary for the UI.
    Returns:
        pairs (list): List of trading pairs
        signals (list): List of signal dicts (one per pair, or empty if no signal)
        summary (dict): Per-pair summary (e.g., error, no signal, etc.)
    """
    logger.info(f"Running market analysis...")
    signal_summary = {}
    signals = []
    attempt_log = load_attempt_log()
    attempt_log = reset_attempts_if_new_day(attempt_log)
    model, scaler = load_model_and_scaler()
    for pair in ALL_PAIRS:
        attempts = get_attempts(attempt_log, pair)
        if attempts >= MAX_ATTEMPTS:
            logger.info(f"Max attempts reached for {pair} today. Skipping.")
            signal_summary[pair] = f'Max attempts reached ({MAX_ATTEMPTS})'
            continue
        try:
            logger.info(f"Fetching price data for {pair}...")
            price_df = get_price_data(pair, interval='1h', lookback=Config.LOOKBACK_PERIOD)
            if price_df is None or price_df.empty:
                logger.warning(f"No price data for {pair}. Skipping analysis for this pair.")
                signal_summary[pair] = 'No price data'
                increment_attempt(attempt_log, pair)
                continue
            logger.info(f"Fetched {len(price_df)} rows of price data for {pair}")
            keywords = get_pair_keywords(pair)
            logger.info(f"Fetching news sentiment (with cache)...")
            from_date = price_df.index[-Config.LOOKBACK_PERIOD].strftime('%Y-%m-%d')
            to_date = price_df.index[-1].strftime('%Y-%m-%d')
            sentiment = get_news_sentiment_with_cache(keywords, from_date, to_date, pair)
            logger.info(f"News sentiment score: {sentiment:.4f}")
            logger.info(f"Generating technical indicators...")
            features_df = preprocess_features(price_df, sentiment)
            logger.info(f"Feature DataFrame shape: {features_df.shape}")
            logger.info(f"Feature summary (last 5 rows):\n{features_df.tail()}\n")
            df = prepare_target_variable(features_df)
            if 'target' in df.columns:
                class_counts = df['target'].value_counts()
                logger.info(f"Target class balance: {class_counts.to_dict()}")
            else:
                logger.warning(f"No target column found after feature engineering for {pair}.")
                signal_summary[pair] = 'No target column'
                increment_attempt(attempt_log, pair)
                continue
            logger.info(f"Making signal prediction...")
            prediction_result = predict_signal(features_df, model_path='models/saved_models/signal_model.pkl')
            signal = generate_signal_output(pair, features_df, prediction_result)
            if signal is not None:
                logger.info(f"SIGNAL GENERATED for {pair}: {signal['signal']} {signal['trade_type']} Confidence: {signal['confidence']:.2%}")
                signal_summary[pair] = f"Signal: {signal['signal']} {signal['trade_type']} Confidence: {signal['confidence']:.2%}"
                save_signal_to_csv(signal, features=features_df.iloc[-1].to_dict())
                reset_attempts_for_pair(attempt_log, pair)
                signals.append(signal)
            else:
                logger.info(f"No valid signal generated for {pair}: filtered out due to confluence, confidence, or news event.")
                signal_summary[pair] = 'No valid signal'
                increment_attempt(attempt_log, pair)
        except Exception as e:
            logger.error(f"Error analyzing {pair}: {e}", exc_info=True)
            signal_summary[pair] = f"Error: {e}"
            increment_attempt(attempt_log, pair)
    logger.info("Signal generation summary:")
    for pair, result in signal_summary.items():
        logger.info(f"{pair}: {result}")
    return ALL_PAIRS, signals, signal_summary 