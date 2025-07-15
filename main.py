import time
import schedule
from data.fetch_market import get_price_data, SYMBOL_MAP
from data.fetch_news import get_news_sentiment_with_cache
from data.preprocess import preprocess_features
from models.train_model import prepare_target_variable, train_signal_model
from models.predict import predict_signal, get_signal_strength
from signals.signal_generator import generate_signal_output
from config import Config
from datetime import datetime, timedelta
from utils.notify import send_email
from utils.logger import get_logger
import pandas as pd
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np


logger = get_logger()

PAIRS = ['USDJPY', 'BTCUSD', 'USDCHF', 'JPYNZD']
SIGNALS_CSV = 'logs/signals.csv'


def save_signal_to_csv(signal, features=None):
    fields = [
        'signal_id', 'time', 'pair', 'signal', 'trade_type', 'confidence', 'entry', 'stop_loss',
        'take_profit_1', 'take_profit_2', 'latest_close', 'latest_high', 'latest_low', 'outcome'
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


def analyze_and_signal():
    logger.info(f"Running market analysis...")
    signal_summary = {}
    max_attempts = 20
    for pair in PAIRS:
        try:
            logger.info(f"Fetching price data for {pair}...")
            price_df = get_price_data(pair, interval='1h', lookback=Config.LOOKBACK_PERIOD)
            if price_df is None or price_df.empty:
                logger.warning(f"No price data for {pair}. Skipping analysis for this pair.")
                signal_summary[pair] = 'No price data'
                continue
            logger.info(f"Fetched {len(price_df)} rows of price data for {pair}")
            # Use pair-specific keywords
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
            logger.debug(f"features_df columns before prepare_target_variable: {features_df.columns}")
            df = prepare_target_variable(features_df)
            logger.debug(f"df columns after prepare_target_variable: {df.columns}")
            logger.debug(f"df columns after dropna on target: {df.columns}")
            if 'target' in df.columns:
                class_counts = df['target'].value_counts()
                logger.info(f"Target class balance: {class_counts.to_dict()}")
            else:
                logger.warning(f"No target column found after feature engineering for {pair}.")
                signal_summary[pair] = 'No target column'
                continue
            logger.info(f"Training AI model...")
            model, scaler, calibration_model = train_signal_model(df)
            logger.info(f"Model trained successfully!")
            logger.info(f"Making signal prediction...")
            # Loop until signal is generated or max_attempts reached
            signal = None
            for attempt in range(1, max_attempts+1):
                prediction_result = predict_signal(features_df, model_path='models/saved_models/signal_model.pkl')
                signal = generate_signal_output(pair, features_df, prediction_result)
                if signal is not None:
                    logger.info(f"SIGNAL GENERATED for {pair} (attempt {attempt}): {signal['signal']} {signal['trade_type']} Confidence: {signal['confidence']:.2%}")
                    signal_summary[pair] = f"Signal: {signal['signal']} {signal['trade_type']} Confidence: {signal['confidence']:.2%} (attempt {attempt})"
                    save_signal_to_csv(signal, features=features_df.iloc[-1].to_dict())
                    break
                else:
                    logger.info(f"No valid signal generated for {pair} (attempt {attempt}). Re-analyzing in 60s...")
                    time.sleep(60)
            if signal is None:
                logger.info(f"No valid signal generated for {pair} after {max_attempts} attempts.")
                signal_summary[pair] = f'No valid signal after {max_attempts} attempts'
        except Exception as e:
            logger.error(f"Error analyzing {pair}: {e}", exc_info=True)
            signal_summary[pair] = f"Error: {e}"
    logger.info("Signal generation summary:")
    for pair, result in signal_summary.items():
        logger.info(f"{pair}: {result}")

def main():
    logger.info("AI-Powered Forex & Crypto Signal Generator (Continuous Mode)")
    schedule.every(15).minutes.do(analyze_and_signal)
    analyze_and_signal()  # Run once at startup
    while True:
        schedule.run_pending()
        time.sleep(5)

if __name__ == "__main__":
    main()
