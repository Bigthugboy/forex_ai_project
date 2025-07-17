import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from utils.logger import get_logger
from config import Config
from models.train_model import train_signal_model
from models.predict import predict_signal
from data.fetch_market import get_price_data
from data.preprocess import preprocess_features
from data.fetch_news import get_news_sentiment_with_cache
from data.multi_timeframe import get_multi_timeframe_features

logger = get_logger('continuous_learning')

class ContinuousLearning:
    def __init__(self):
        self.outcomes_file = 'logs/signal_outcomes.csv'
        self.model_performance_file = 'logs/model_performance.csv'
        self.retrain_threshold = 50  # Retrain after 50 new outcomes
        self.min_accuracy_threshold = 0.65
        self.last_retrain_date = None
        self.load_last_retrain_date()
        
    def load_last_retrain_date(self):
        """Load the last retrain date from file"""
        try:
            if os.path.exists('logs/last_retrain.txt'):
                with open('logs/last_retrain.txt', 'r') as f:
                    date_str = f.read().strip()
                    self.last_retrain_date = datetime.fromisoformat(date_str)
                    logger.info(f"Last retrain date: {self.last_retrain_date}")
        except Exception as e:
            logger.warning(f"Could not load last retrain date: {e}")
            self.last_retrain_date = None
    
    def save_last_retrain_date(self):
        """Save the current retrain date"""
        try:
            os.makedirs('logs', exist_ok=True)
            with open('logs/last_retrain.txt', 'w') as f:
                f.write(datetime.now().isoformat())
        except Exception as e:
            logger.error(f"Could not save retrain date: {e}")
    
    def log_signal_outcome(self, signal_data, outcome_data):
        """Log the outcome of a generated signal"""
        try:
            # Create outcome record
            outcome_record = {
                'timestamp': datetime.now().isoformat(),
                'pair': signal_data.get('pair'),
                'signal_type': signal_data.get('signal'),
                'entry_price': signal_data.get('entry'),
                'stop_loss': signal_data.get('stop_loss'),
                'take_profit_1': signal_data.get('take_profit_1'),
                'take_profit_2': signal_data.get('take_profit_2'),
                'take_profit_3': signal_data.get('take_profit_3'),
                'confidence': signal_data.get('confidence'),
                'confluence': signal_data.get('confluence'),
                'confluence_factors': ', '.join(signal_data.get('confluence_factors', [])),
                'outcome': outcome_data.get('outcome'),
                'exit_price': outcome_data.get('exit_price'),
                'exit_time': outcome_data.get('exit_time'),
                'pnl': outcome_data.get('pnl'),
                'exit_reason': outcome_data.get('exit_reason'),
                'position_size': signal_data.get('position_size')
            }
            
            # Save to CSV
            df = pd.DataFrame([outcome_record])
            if os.path.exists(self.outcomes_file):
                df.to_csv(self.outcomes_file, mode='a', header=False, index=False)
            else:
                df.to_csv(self.outcomes_file, index=False)
            
            logger.info(f"Logged signal outcome: {signal_data.get('pair')} - {outcome_data.get('outcome')}")
            
        except Exception as e:
            logger.error(f"Error logging signal outcome: {e}")
    
    def get_recent_outcomes(self, days=30):
        """Get recent signal outcomes for analysis"""
        try:
            if not os.path.exists(self.outcomes_file):
                return pd.DataFrame()
            
            df = pd.read_csv(self.outcomes_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter for recent outcomes
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_df = df[df['timestamp'] >= cutoff_date]
            
            return recent_df
        except Exception as e:
            logger.error(f"Error reading recent outcomes: {e}")
            return pd.DataFrame()
    
    def calculate_model_performance(self, days=30):
        """Calculate current model performance metrics"""
        try:
            recent_outcomes = self.get_recent_outcomes(days)
            
            if recent_outcomes.empty:
                return {
                    'total_signals': 0,
                    'win_rate': 0,
                    'avg_pnl': 0,
                    'accuracy': 0,
                    'high_confidence_signals': 0
                }
            
            # Filter completed trades (not pending)
            completed = recent_outcomes[recent_outcomes['outcome'] != 'pending']
            
            if completed.empty:
                return {
                    'total_signals': len(recent_outcomes),
                    'win_rate': 0,
                    'avg_pnl': 0,
                    'accuracy': 0,
                    'high_confidence_signals': 0
                }
            
            wins = completed[completed['outcome'] == 'win']
            losses = completed[completed['outcome'] == 'loss']
            
            total_completed = len(completed)
            win_rate = len(wins) / total_completed if total_completed > 0 else 0
            avg_pnl = completed['pnl'].mean() if 'pnl' in completed.columns else 0
            
            # Calculate accuracy based on confidence vs outcome
            high_confidence = completed[completed['confidence'] >= 0.8]
            if not high_confidence.empty:
                high_conf_wins = high_confidence[high_confidence['outcome'] == 'win']
                accuracy = len(high_conf_wins) / len(high_confidence)
            else:
                accuracy = 0
            
            performance = {
                'total_signals': len(recent_outcomes),
                'completed_trades': total_completed,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'accuracy': accuracy,
                'high_confidence_signals': len(high_confidence)
            }
            
            # Log performance
            logger.info(f"Model Performance ({days} days): "
                       f"Win Rate: {win_rate:.2%}, "
                       f"Accuracy: {accuracy:.2%}, "
                       f"Avg PnL: {avg_pnl:.2f}")
            
            return performance
            
        except Exception as e:
            logger.error(f"Error calculating model performance: {e}")
            return {}
    
    def should_retrain(self):
        """Determine if the model should be retrained"""
        try:
            # Check if enough time has passed since last retrain
            if self.last_retrain_date:
                days_since_retrain = (datetime.now() - self.last_retrain_date).days
                if days_since_retrain < 7:  # Minimum 7 days between retrains
                    return False
            
            # Check number of new outcomes
            recent_outcomes = self.get_recent_outcomes(days=30)
            if len(recent_outcomes) >= self.retrain_threshold:
                logger.info(f"Retrain triggered: {len(recent_outcomes)} new outcomes")
                return True
            # Check performance degradation
            performance = self.calculate_model_performance(days=30)
            if performance.get('accuracy', 1) < self.min_accuracy_threshold:
                logger.info(f"Retrain triggered: Low accuracy {performance.get('accuracy', 0):.2%}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error checking retrain conditions: {e}")
            return False
    
    def retrain_model(self):
        """Retrain the model with updated data"""
        try:
            logger.info("Starting model retraining...")
            
            # Collect fresh data for all pairs
            all_features = []
            all_targets = []
            
            for pair in Config.TRADING_PAIRS:
                logger.info(f"Collecting data for {pair}...")
                
                # Fetch price data
                price_df = get_price_data(pair, interval='1h', lookback=Config.LOOKBACK_PERIOD)
                if price_df is None or price_df.empty:
                    logger.warning(f"No price data for {pair}")
                    continue
                
                # Fetch news sentiment
                keywords = self.get_pair_keywords(pair)
                from_date = price_df.index[-Config.LOOKBACK_PERIOD].strftime('%Y-%m-%d')
                to_date = price_df.index[-1].strftime('%Y-%m-%d')
                sentiment = get_news_sentiment_with_cache(keywords, from_date, to_date, pair)
                
                # Generate features
                features_df = preprocess_features(price_df, sentiment, use_multi_timeframe=True)
                
                # Prepare target variable
                from data.preprocess import prepare_target_variable
                df = prepare_target_variable(features_df)
                
                if 'target' in df.columns:
                    # Remove price columns, keep only features
                    feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'target']]
                    all_features.append(df[feature_cols])
                    all_targets.append(df['target'])
                    
                    logger.info(f"Added {len(df)} samples for {pair}")
            
            if not all_features:
                logger.error("No features collected for retraining")
                return False
            
            # Combine all data
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_targets = pd.concat(all_targets, ignore_index=True)
            
            logger.info(f"Total training data: {len(combined_features)} samples")
            
            # Train new model
            model, scaler, feature_cols = train_signal_model(
                pd.concat([combined_features, combined_targets], axis=1)
            )
            
            # Save updated model
            os.makedirs('models/saved_models', exist_ok=True)
            with open('models/saved_models/signal_model.pkl', 'wb') as f:
                pickle.dump({
                    'model': model,
                    'scaler': scaler,
                    'feature_cols': feature_cols,
                    'retrain_date': datetime.now().isoformat()
                }, f)
            
            # Update retrain date
            self.save_last_retrain_date()
            self.last_retrain_date = datetime.now()
            
            logger.info("Model retraining completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
            return False
    
    def get_pair_keywords(self, pair):
        """Get keywords for a specific pair"""
        if 'JPY' in pair:
            return ['USD/JPY', 'JPY/USD', 'Federal Reserve', 'Bank of Japan', 'interest rates']
        elif 'BTC' in pair:
            return ['Bitcoin', 'BTC', 'cryptocurrency', 'cryptodigital currency']
        else:
            return Config.NEWS_KEYWORDS
    
    def run_continuous_learning_cycle(self):
        """Complete continuous learning cycle"""
        try:
            logger.info("Running continuous learning cycle...")
            
            # Check if retraining is needed
            if self.should_retrain():
                logger.info("Retraining conditions met, starting retrain...")
                success = self.retrain_model()
                if success:
                    logger.info("Continuous learning cycle completed successfully")
                else:
                    logger.error("Continuous learning cycle failed")
            else:
                logger.info("No retraining needed at this time")
            
            # Log current performance
            performance = self.calculate_model_performance(days=30)
            logger.info(f"Current performance: {performance}")
            
        except Exception as e:
            logger.error(f"Error in continuous learning cycle: {e}")

# Global instance
continuous_learner = ContinuousLearning() 