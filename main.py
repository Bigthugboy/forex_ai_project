import time
import schedule
from data.fetch_market import get_price_data
from data.fetch_news import get_news_sentiment
from data.preprocess import preprocess_features
from models.train_model import train_signal_model
from models.predict import predict_signal, get_signal_strength
from signals.signal_generator import generate_signal_output
from config import Config
from datetime import datetime, timedelta
from utils.notify import send_email
from utils.logger import get_logger
import pandas as pd
import os

logger = get_logger()

PAIRS = ['USDJPY', 'BTCJPY', 'USDCHF', 'JPYNZD']
SIGNALS_CSV = 'logs/signals.csv'


def save_signal_to_csv(signal):
    fields = [
        'time', 'pair', 'signal', 'trade_type', 'confidence', 'entry', 'stop_loss',
        'take_profit_1', 'take_profit_2', 'latest_close', 'latest_high', 'latest_low'
    ]
    row = {k: signal.get(k, None) for k in fields}
    row['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df = pd.DataFrame([row])
    file_exists = os.path.exists(SIGNALS_CSV)
    if not file_exists:
        df.to_csv(SIGNALS_CSV, mode='w', index=False)
    else:
        df.to_csv(SIGNALS_CSV, mode='a', index=False, header=False)


def analyze_and_signal():
    logger.info(f"Running market analysis...")
    for pair in PAIRS:
        try:
            logger.info(f"Fetching price data for {pair}...")
            price_df = get_price_data(pair, interval='1h', lookback=Config.LOOKBACK_PERIOD)
            logger.info(f"Fetched {len(price_df)} rows of price data for {pair}")

            logger.info(f"Fetching news sentiment...")
            today = datetime.utcnow().date()
            from_date = (today - timedelta(days=2)).strftime('%Y-%m-%d')
            to_date = today.strftime('%Y-%m-%d')
            sentiment = get_news_sentiment(Config.NEWS_KEYWORDS, from_date, to_date)
            logger.info(f"News sentiment score: {sentiment:.4f}")

            logger.info(f"Generating technical indicators...")
            features_df = preprocess_features(price_df, sentiment)
            logger.info(f"Feature DataFrame shape: {features_df.shape}")

            logger.info(f"Training AI model...")
            model, scaler, feature_cols = train_signal_model(features_df)
            logger.info("Model trained successfully!")

            logger.info(f"Making signal prediction...")
            prediction_result = predict_signal(features_df)

            if prediction_result and prediction_result['confidence'] >= 0.8:
                signal = generate_signal_output(pair, features_df, prediction_result)
                logger.info(f"SIGNAL GENERATED for {pair}: {signal['signal']} {signal['trade_type']} Confidence: {signal['confidence']:.2%}")
                logger.info(f"Entry: {signal['entry']} SL: {signal['stop_loss']} TP1: {signal['take_profit_1']} TP2: {signal['take_profit_2']}")
                # --- Email Notification ---
                subject = f"[Forex AI Signal] {pair} {signal['signal']} ({signal['trade_type']}) {signal['confidence']:.0%}"
                message = (
                    f"PAIR: {signal['pair']}\n"
                    f"SIGNAL: {signal['signal']}\n"
                    f"STRENGTH: {get_signal_strength(signal['confidence'])}\n"
                    f"CONFIDENCE: {signal['confidence']:.2%}\n"
                    f"TRADE TYPE: {signal['trade_type']}\n"
                    f"ENTRY: {signal['entry']}\n"
                    f"STOP LOSS: {signal['stop_loss']}\n"
                    f"TAKE PROFIT 1 (2:1): {signal['take_profit_1']}\n"
                    f"TAKE PROFIT 2 (3:1): {signal['take_profit_2']}\n"
                    f"TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"LAST PRICE: {signal['latest_close']}\n"
                    f"HIGH: {signal['latest_high']}  LOW: {signal['latest_low']}\n"
                )
                send_email(subject, message)
                # --- Save to CSV for UI ---
                save_signal_to_csv(signal)
            else:
                logger.info(f"No high-confidence signal for {pair} (confidence: {prediction_result['confidence'] if prediction_result else 'N/A'})")
        except Exception as e:
            logger.error(f"Error analyzing {pair}: {e}", exc_info=True)

def main():
    logger.info("AI-Powered Forex & Crypto Signal Generator (Continuous Mode)")
    schedule.every(15).minutes.do(analyze_and_signal)
    analyze_and_signal()  # Run once at startup
    while True:
        schedule.run_pending()
        time.sleep(5)

if __name__ == "__main__":
    main()
