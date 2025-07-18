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
from models.model_training_service import ModelTrainingService
from data.data_fetcher import DataFetcher
from models.model_registry import ModelRegistry
from utils.analytics_logger import AnalyticsLogger
import collections
import csv
import threading
import time

logger = get_logger('continuous_learning')

class ContinuousLearning:
    def __init__(self):
        self.outcomes_file = 'logs/signal_outcomes.csv'
        self.model_performance_file = 'logs/model_performance.csv'
        self.factor_success_stats_file = 'logs/factor_success_stats.json'
        self.retrain_threshold = 50  # Retrain after 50 new outcomes
        self.min_accuracy_threshold = 0.65
        self.last_retrain_date = None
        self.factor_success_counter = collections.Counter()
        self.load_last_retrain_date()
        self.load_factor_success_stats()
        
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
    
    def load_factor_success_stats(self):
        import os, json
        if os.path.exists(self.factor_success_stats_file):
            with open(self.factor_success_stats_file, 'r') as f:
                self.factor_success_counter = collections.Counter(json.load(f))

    def save_factor_success_stats(self):
        import json
        with open(self.factor_success_stats_file, 'w') as f:
            json.dump(self.factor_success_counter, f, indent=2)

    def generate_tp_summary_report(self, days=30):
        """Generate a summary report: per pair, % of signals hitting each TP and SL, and average pips. Also include factor success stats."""
        try:
            df = self.get_recent_outcomes(days)
            if df.empty:
                logger.info("No signal outcomes to summarize.")
                return
            summary = {}
            for pair in df['pair'].unique():
                pair_df = df[df['pair'] == pair]
                total = len(pair_df)
                tp1 = pair_df['tps_hit'].str.contains('tp1').sum()
                tp2 = pair_df['tps_hit'].str.contains('tp2').sum()
                tp3 = pair_df['tps_hit'].str.contains('tp3').sum()
                sl = pair_df['outcome'].eq('loss').sum()
                avg_pnl = pair_df['pnl'].mean() if 'pnl' in pair_df.columns else 0
                summary[pair] = {
                    'total_signals': total,
                    'tp1_pct': tp1 / total * 100 if total else 0,
                    'tp2_pct': tp2 / total * 100 if total else 0,
                    'tp3_pct': tp3 / total * 100 if total else 0,
                    'sl_pct': sl / total * 100 if total else 0,
                    'avg_pnl': avg_pnl
                }
            logger.info("=== TP/SL Summary Report (last %d days) ===" % days)
            for pair, stats in summary.items():
                logger.info(f"{pair}: Total={stats['total_signals']}, TP1={stats['tp1_pct']:.1f}%, TP2={stats['tp2_pct']:.1f}%, TP3={stats['tp3_pct']:.1f}%, SL={stats['sl_pct']:.1f}%, AvgPnL={stats['avg_pnl']:.1f}")
            # Factor success stats
            logger.info("=== Confluence Factor Success (TP hits) ===")
            for factor, count in self.factor_success_counter.most_common():
                logger.info(f"{factor}: {count} TP hits")
            print("=== TP/SL Summary Report (last %d days) ===" % days)
            for pair, stats in summary.items():
                print(f"{pair}: Total={stats['total_signals']}, TP1={stats['tp1_pct']:.1f}%, TP2={stats['tp2_pct']:.1f}%, TP3={stats['tp3_pct']:.1f}%, SL={stats['sl_pct']:.1f}%, AvgPnL={stats['avg_pnl']:.1f}")
            print("=== Confluence Factor Success (TP hits) ===")
            for factor, count in self.factor_success_counter.most_common():
                print(f"{factor}: {count} TP hits")
        except Exception as e:
            logger.error(f"Error generating TP summary report: {e}")
    
    def log_signal_outcome(self, signal_data, outcome_data):
        """Log the outcome of a generated signal, including all TPs hit and factor success."""
        try:
            # Track factor success for explainability
            tps_hit = outcome_data.get('tps_hit', [])
            if tps_hit:
                confluence_factors = signal_data.get('confluence_factors', [])
                for factor in confluence_factors:
                    self.factor_success_counter[factor] += 1
                self.save_factor_success_stats()
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
                'tps_hit': ', '.join(outcome_data.get('tps_hit', [])),
                'position_size': signal_data.get('position_size')
            }
            
            # Save to CSV
            df = pd.DataFrame([outcome_record])
            if os.path.exists(self.outcomes_file):
                df.to_csv(self.outcomes_file, mode='a', header=False, index=False)
            else:
                df.to_csv(self.outcomes_file, index=False)
            
            logger.info(f"Logged signal outcome: {signal_data.get('pair')} - {outcome_data.get('outcome')}")
            
            analytics_logger = AnalyticsLogger()
            analytics_logger.log_signal(
                pair=signal_data.get('pair'),
                trend=signal_data.get('trend', 'N/A'),
                key_levels=signal_data.get('key_levels', {}),
                patterns=signal_data.get('patterns', []),
                indicators=signal_data.get('indicators', {}),
                confluence=signal_data.get('confluence', {}),
                model_action=f"Outcome: {outcome_data.get('exit_reason', 'N/A')}",
                decision=signal_data.get('signal', 'N/A'),
                confidence=signal_data.get('confidence', 0.0),
                reason=f"Outcome: {outcome_data.get('outcome', 'N/A')}, PnL: {outcome_data.get('pnl', 0.0)}, TPs hit: {', '.join(outcome_data.get('tps_hit', []))}"
            )
            # Call summary report after logging
            self.generate_tp_summary_report(days=30)
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
    
    def should_retrain_pair(self, pair, current_version_hash):
        """Check ModelRegistry for the latest version for this pair. Retrain if version hash differs."""
        registry = ModelRegistry()
        entry = registry.get_latest_model(f'signal_model_{pair}')
        if entry and entry.get('version') == current_version_hash:
            logger.info(f"[Registry] Model for {pair} is up-to-date (version {current_version_hash}). No retrain needed.")
            return False
        logger.info(f"[Registry] Model for {pair} is outdated or missing. Retrain needed.")
        return True

    def retrain_model(self):
        """Retrain the model with updated data, only if registry indicates a new version is needed."""
        try:
            logger.info("Starting model retraining...")
            all_features = []
            all_targets = []
            for pair in Config.TRADING_PAIRS:
                logger.info(f"Collecting data for {pair}...")
                price_df = data_fetcher.get_price_data(pair, interval='1h', lookback=Config.LOOKBACK_PERIOD)
                if price_df is None or price_df.empty:
                    logger.warning(f"No price data for {pair}")
                    continue
                keywords = self.get_pair_keywords(pair)
                from_date = price_df.index[-Config.LOOKBACK_PERIOD].strftime('%Y-%m-%d')
                to_date = price_df.index[-1].strftime('%Y-%m-%d')
                sentiment = data_fetcher.get_news_sentiment(keywords, from_date, to_date, pair)
                features_df = preprocess_features(price_df, sentiment, use_multi_timeframe=True)
                df = model_trainer.prepare_target(features_df)
                if 'target' in df.columns:
                    feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'target']]
                    all_features.append(df[feature_cols])
                    all_targets.append(df['target'])
                    logger.info(f"Added {len(df)} samples for {pair}")
            if not all_features:
                logger.error("No features collected for retraining")
                return False
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_targets = pd.concat(all_targets, ignore_index=True)
            logger.info(f"Total training data: {len(combined_features)} samples")
            train_df = pd.concat([combined_features, combined_targets], axis=1)
            # Compute current version hash (simulate as in train_model)
            import hashlib
            code_path = os.path.abspath(__file__)
            with open(code_path, 'rb') as f:
                code_bytes = f.read()
            code_hash = hashlib.md5(code_bytes).hexdigest()
            current_version_hash = code_hash  # Optionally add config hash
            # Check registry for each pair
            retrain_needed = False
            for pair in Config.TRADING_PAIRS:
                if self.should_retrain_pair(pair, current_version_hash):
                    retrain_needed = True
                    break
            if not retrain_needed:
                logger.info("All models up-to-date. Skipping retrain.")
                return True
            train_result = model_trainer.train(train_df, 'ALL')
            model = train_result['model']
            scaler = train_result['scaler']
            feature_cols = train_result['feature_cols']
            os.makedirs('models/saved_models', exist_ok=True)
            with open('models/saved_models/signal_model.pkl', 'wb') as f:
                pickle.dump({
                    'model': model,
                    'scaler': scaler,
                    'feature_cols': feature_cols,
                    'retrain_date': datetime.now().isoformat()
                }, f)
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

    def export_analytics_to_csv(self, days=30, out_path='logs/analytics_summary.csv'):
        """Export TP/SL summary and factor success stats to CSV."""
        df = self.get_recent_outcomes(days)
        if df.empty:
            logger.info("No signal outcomes to export.")
            return
        summary_rows = []
        for pair in df['pair'].unique():
            pair_df = df[df['pair'] == pair]
            for signal_type in ['BUY', 'SELL']:
                st_df = pair_df[pair_df['signal_type'] == signal_type]
                total = len(st_df)
                if total == 0:
                    continue
                tp1 = st_df['tps_hit'].str.contains('tp1').sum()
                tp2 = st_df['tps_hit'].str.contains('tp2').sum()
                tp3 = st_df['tps_hit'].str.contains('tp3').sum()
                sl = st_df['outcome'].eq('loss').sum()
                avg_pnl = st_df['pnl'].mean() if 'pnl' in st_df.columns else 0
                avg_days = st_df['days_to_outcome'].mean() if 'days_to_outcome' in st_df.columns else 0
                med_days = st_df['days_to_outcome'].median() if 'days_to_outcome' in st_df.columns else 0
                summary_rows.append({
                    'pair': pair,
                    'signal_type': signal_type,
                    'total_signals': total,
                    'tp1_pct': tp1 / total * 100 if total else 0,
                    'tp2_pct': tp2 / total * 100 if total else 0,
                    'tp3_pct': tp3 / total * 100 if total else 0,
                    'sl_pct': sl / total * 100 if total else 0,
                    'avg_pnl': avg_pnl,
                    'avg_days_to_outcome': avg_days,
                    'med_days_to_outcome': med_days
                })
        # Factor success stats
        factor_rows = [{'factor': k, 'tp_hits': v} for k, v in self.factor_success_counter.most_common()]
        # Write to CSV
        with open(out_path, 'w', newline='') as csvfile:
            fieldnames = list(summary_rows[0].keys()) if summary_rows else []
            factor_fieldnames = ['factor', 'tp_hits']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
            # Add a blank line and then factor stats
            csvfile.write('\n')
            factor_writer = csv.DictWriter(csvfile, fieldnames=factor_fieldnames)
            factor_writer.writeheader()
            for row in factor_rows:
                factor_writer.writerow(row)
        logger.info(f"Exported analytics summary to {out_path}")

    def send_tp_summary_email(self, days=30, send_email_func=None):
        """Send the TP/SL summary report via email (placeholder for actual email logic)."""
        try:
            import matplotlib.pyplot as plt
            df = self.get_recent_outcomes(days)
            if df.empty:
                logger.info("No signal outcomes to email.")
                return
            summary_lines = [f"=== TP/SL Summary Report (last {days} days) ==="]
            for pair in df['pair'].unique():
                pair_df = df[df['pair'] == pair]
                for signal_type in ['BUY', 'SELL']:
                    st_df = pair_df[pair_df['signal_type'] == signal_type]
                    total = len(st_df)
                    if total == 0:
                        continue
                    tp1 = st_df['tps_hit'].str.contains('tp1').sum()
                    tp2 = st_df['tps_hit'].str.contains('tp2').sum()
                    tp3 = st_df['tps_hit'].str.contains('tp3').sum()
                    sl = st_df['outcome'].eq('loss').sum()
                    avg_pnl = st_df['pnl'].mean() if 'pnl' in st_df.columns else 0
                    avg_days = st_df['days_to_outcome'].mean() if 'days_to_outcome' in st_df.columns else 0
                    med_days = st_df['days_to_outcome'].median() if 'days_to_outcome' in st_df.columns else 0
                    avg_hours = avg_days * 24
                    med_hours = med_days * 24
                    summary_lines.append(
                        f"{pair} [{signal_type}]: Total={total}, TP1={tp1/total*100 if total else 0:.1f}%, TP2={tp2/total*100 if total else 0:.1f}%, TP3={tp3/total*100 if total else 0:.1f}%, SL={sl/total*100 if total else 0:.1f}%, AvgPnL={avg_pnl:.1f}, AvgTime={avg_days:.2f}d/{avg_hours:.1f}h, MedTime={med_days:.2f}d/{med_hours:.1f}h"
                    )
                # Visualize signal duration distribution
                self.save_signal_duration_histogram(pair_df, pair)
            # Add factor success stats
            summary_lines.append("=== Confluence Factor Success (TP hits) ===")
            for factor, count in self.factor_success_counter.most_common():
                summary_lines.append(f"{factor}: {count} TP hits")
            summary_text = '\n'.join(summary_lines)
            logger.info("Sending TP/SL summary email...")
            if send_email_func:
                send_email_func(subject="TP/SL Summary Report", body=summary_text)
            else:
                logger.info("[Email Placeholder] Would send email with body:\n" + summary_text)
            # After sending email, export analytics to CSV
            self.export_analytics_to_csv(days=days)
        except Exception as e:
            logger.error(f"Error sending TP summary email: {e}")

    def save_signal_duration_histogram(self, df, pair):
        """Save a histogram plot of signal duration (days_to_outcome) for a pair."""
        try:
            import matplotlib.pyplot as plt
            if 'days_to_outcome' not in df.columns or df['days_to_outcome'].empty:
                return
            plt.figure(figsize=(6, 4))
            plt.hist(df['days_to_outcome'].dropna(), bins=20, color='skyblue', edgecolor='black')
            plt.title(f'Signal Duration Distribution: {pair}')
            plt.xlabel('Days to Outcome')
            plt.ylabel('Count')
            plt.tight_layout()
            out_path = f'logs/signal_duration_{pair}.png'
            plt.savefig(out_path)
            plt.close()
            logger.info(f"Saved signal duration histogram: {out_path}")
        except Exception as e:
            logger.error(f"Error saving signal duration histogram for {pair}: {e}")

    def health_check_no_signals(self, alert_hours=24, send_email_func=None):
        """Check if any signals have been generated in the last alert_hours. Log warning only if none."""
        while True:
            try:
                df = self.get_recent_outcomes(days=2)
                now = pd.Timestamp.utcnow()
                recent = df[df['timestamp'] >= (now - pd.Timedelta(hours=alert_hours))]
                if recent.empty:
                    msg = f"ALERT: No signals generated in the last {alert_hours} hours!"
                    logger.warning(msg)
                else:
                    logger.info(f"Health check: {len(recent)} signals generated in the last {alert_hours} hours.")
            except Exception as e:
                logger.error(f"Error in health check: {e}")
            time.sleep(3600)  # Check every hour

    def run_daily_summary_email(self, send_email_func=None):
        """Schedule summary email every day at 23:00 UTC. Also run health check in background."""
        health_thread = threading.Thread(target=self.health_check_no_signals, args=(24, send_email_func), daemon=True)
        health_thread.start()
        import datetime
        while True:
            now = datetime.datetime.utcnow()
            if now.hour == 23 and now.minute == 0:
                self.send_tp_summary_email(days=30, send_email_func=send_email_func)
                time.sleep(60)  # Wait a minute to avoid duplicate sends
            else:
                time.sleep(30)  # Check every 30 seconds

# Global instance
continuous_learner = ContinuousLearning() 
model_trainer = ModelTrainingService() 
data_fetcher = DataFetcher() 
analytics_logger = AnalyticsLogger() 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--export-analytics', action='store_true', help='Export analytics summary to CSV')
    args = parser.parse_args()
    if args.export_analytics:
        cl = ContinuousLearning()
        cl.export_analytics_to_csv() 