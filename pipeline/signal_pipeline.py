import pandas as pd
from data.data_fetcher import DataFetcher
from data.preprocess import preprocess_features
from models.model_training_service import ModelTrainingService
from models.predict import predict_signal
from signals.signal_generator import generate_signal_output
from utils.logger import get_logger
from config import Config
from utils.analytics_logger import AnalyticsLogger

logger = get_logger('signal_pipeline', log_file='logs/signal_pipeline.log')
analytics_logger = AnalyticsLogger()

class SignalPipeline:
    def __init__(self, mode='live', pair=None):
        """
        mode: 'live', 'backtest', or 'study-only'
        pair: Optional, restrict pipeline to a single pair (e.g., BTCUSD)
        """
        self.mode = mode
        self.pair = pair

    def run(self, pair, lookback=None, multi_timeframe=True, send_email_func=None):
        """
        Run the full signal pipeline for a given pair.
        Returns: dict with signal, features, and logs.
        """
        result = {'pair': pair, 'signal': None, 'features': None, 'error': None, 'logs': []}
        try:
            lookback = lookback or Config.LOOKBACK_PERIOD
            logger.info(f"Fetching price data for {pair}...")
            data_fetcher = DataFetcher()
            price_df = data_fetcher.get_price_data(pair, interval='1h', lookback=lookback)
            if price_df is None or price_df.empty:
                logger.warning(f"No price data for {pair}.")
                result['error'] = 'No price data'
                return result
            logger.info(f"Fetched {len(price_df)} rows of price data for {pair}")
            keywords = self.get_pair_keywords(pair)
            logger.info(f"Fetching news sentiment...")
            from_date = price_df.index[-lookback].strftime('%Y-%m-%d')
            to_date = price_df.index[-1].strftime('%Y-%m-%d')
            sentiment = data_fetcher.get_news_sentiment(keywords, from_date, to_date, pair)
            if sentiment is None:
                logger.warning(f"News sentiment returned None for {pair} {to_date}, setting to 0.")
                sentiment = 0.0
            logger.info(f"News sentiment score: {sentiment:.4f}")
            logger.info(f"Generating technical indicators with multi-timeframe analysis...")
            features_df = preprocess_features(price_df, sentiment, use_multi_timeframe=multi_timeframe)
            # Ensure BOS and CHoCH are present as features
            for col in ['bos', 'choch']:
                if col not in features_df.columns:
                    from features.indicators import bos, choch
                    if col == 'bos':
                        features_df['bos'] = bos(features_df)
                    if col == 'choch':
                        features_df['choch'] = choch(features_df)
            logger.info(f"Feature DataFrame shape: {features_df.shape}")
            logger.info(f"Feature summary (last 5 rows):\n{features_df.tail()}\n")
            result['features'] = features_df
            if self.mode == 'study-only':
                # In study mode, always return full analytics data for logs
                result['signal'] = {
                    'signal': 'STUDY',
                    'confidence': 1.0,
                    'confluence': 1.0,
                    'confluence_factors': ['bos', 'choch'],
                    'factors': {'bos': features_df['bos'].iloc[-1], 'choch': features_df['choch'].iloc[-1]},
                }
                result['error'] = None
                return result
            model_trainer = ModelTrainingService()
            df = model_trainer.prepare_target(features_df)
            if 'target' not in df.columns:
                logger.warning(f"No target column found after feature engineering for {pair}.")
                result['error'] = 'No target column'
                return result
            train_result = model_trainer.train(df, pair)
            model = train_result['model']
            scaler = train_result['scaler']
            feature_cols = train_result['feature_cols']
            logger.info(f"Model trained successfully!")
            prediction_result = predict_signal(features_df, pair)
            if prediction_result is None:
                logger.error(f"Prediction failed for {pair}. Skipping signal generation.")
                # Initialize context variables to safe defaults
                market_regime = 'N/A'
                proximity_log = {}
                pattern_strengths = {}
                rsi_str = 'N/A'
                macd_str = 'N/A'
                macd_signal_str = 'N/A'
                news_sentiment = 'N/A'
                factors = {}
                # Compose analytics log context with all fields filled
                analytics_logger.log_signal(
                    pair=pair,
                    trend=market_regime,
                    key_levels=proximity_log,
                    patterns=list(pattern_strengths.keys()),
                    indicators={
                        'rsi': rsi_str,
                        'macd': macd_str,
                        'macd_signal': macd_signal_str,
                        'news_sentiment': news_sentiment
                    },
                    confluence=factors,
                    model_action="No signal: prediction failed or confidence below threshold.",
                    decision="NO_SIGNAL",
                    confidence=0.0,
                    reason="Prediction failed or confidence below threshold."
                )
                return {
                    'pair': pair,
                    'trend': market_regime,
                    'key_levels': proximity_log,
                    'patterns': list(pattern_strengths.keys()),
                    'indicators': {
                        'rsi': rsi_str,
                        'macd': macd_str,
                        'macd_signal': macd_signal_str,
                        'news_sentiment': news_sentiment
                    },
                    'confluence': factors,
                    'model_action': "No signal: prediction failed or confidence below threshold.",
                    'decision': "NO_SIGNAL",
                    'confidence': 0.0,
                    'reason': "Prediction failed or confidence below threshold.",
                    'signal': None
                }
            signal = generate_signal_output(pair, features_df, prediction_result)
            result['signal'] = signal
            if signal and send_email_func:
                email_subject = f"AI Signal: {pair} {signal['signal']} {signal['trade_type']} ({signal['confidence']:.2%})"
                email_body = f"""
Pair: {pair}
Signal: {signal['signal']} ({signal['trade_type']})
Confidence: {signal['confidence']:.2%}
Entry: {signal['entry']}
Stop Loss: {signal['stop_loss']}
Take Profits: {signal['take_profit_1']}, {signal['take_profit_2']}, {signal['take_profit_3']}
Position Size: {signal['position_size']}
Time: {signal.get('time', 'N/A')}
"""
                send_email_func(email_subject, email_body)
            return result
        except Exception as e:
            logger.error(f"Error in signal pipeline for {pair}: {e}", exc_info=True)
            result['error'] = str(e)
            return result

    @staticmethod
    def get_pair_keywords(pair):
        # Dynamically generate keywords for any FX pair
        # e.g., 'EURUSD' -> ['EUR', 'USD', 'EUR/USD', 'USD/EUR']
        if len(pair) == 6:
            base = pair[:3]
            quote = pair[3:]
            return [
                base, quote,
                f"{base}/{quote}", f"{quote}/{base}",
                pair, f"{base}{quote}", f"{quote}{base}"
            ]
        else:
            return [pair] 