import pandas as pd
import numpy as np
import pickle
import os
import json
import time
import threading
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
        self.learning_insights_file = 'logs/learning_insights.json'
        self.pattern_memory_file = 'logs/pattern_memory.json'
        self.retrain_threshold = 50  # Retrain after 50 new outcomes
        self.min_accuracy_threshold = 0.65
        self.last_retrain_date = None
        self.factor_success_counter = collections.Counter()
        
        # Market analysis and pattern learning state
        self.learning_insights = {}
        self.pattern_memory = {}
        self.last_market_analysis = {}
        self.last_pattern_learning = None
        
        # Learning intervals (in seconds)
        self.market_analysis_interval = 300  # 5 minutes
        self.pattern_learning_interval = 1800  # 30 minutes
        
        self.load_last_retrain_date()
        self.load_factor_success_stats()
        self.load_learning_state()
        
    def load_learning_state(self):
        """Load learning insights and pattern memory"""
        try:
            if os.path.exists(self.learning_insights_file):
                with open(self.learning_insights_file, 'r') as f:
                    self.learning_insights = json.load(f)
                logger.info("Loaded learning insights from file")
            
            if os.path.exists(self.pattern_memory_file):
                with open(self.pattern_memory_file, 'r') as f:
                    self.pattern_memory = json.load(f)
                logger.info("Loaded pattern memory from file")
        except Exception as e:
            logger.error(f"Error loading learning state: {e}")
    
    def save_learning_state(self):
        """Save learning insights and pattern memory"""
        try:
            with open(self.learning_insights_file, 'w') as f:
                json.dump(self.learning_insights, f, indent=2)
            
            with open(self.pattern_memory_file, 'w') as f:
                json.dump(self.pattern_memory, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving learning state: {e}")
    
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

    def analyze_market_patterns(self, pair):
        """Analyze current market patterns and store insights"""
        try:
            # Check if we need to analyze this pair
            now = datetime.now()
            if pair in self.last_market_analysis:
                time_since_last = (now - self.last_market_analysis[pair]).total_seconds()
                if time_since_last < self.market_analysis_interval:
                    return  # Too soon to analyze again
            
            logger.info(f"Analyzing market patterns for {pair}")
            
            # Fetch recent data
            price_df = get_price_data(pair, interval='1h', lookback=48)
            if price_df is None or price_df.empty:
                return
            
            # Get news sentiment
            from_date = price_df.index[-48].strftime('%Y-%m-%d')
            to_date = price_df.index[-1].strftime('%Y-%m-%d')
            sentiment = get_news_sentiment_with_cache(
                self.get_pair_keywords(pair), from_date, to_date, pair
            )
            
            # Generate features
            features_df = preprocess_features(price_df, sentiment or 0.0, use_multi_timeframe=True)
            
            # Analyze patterns
            patterns = self.extract_patterns(features_df)
            
            # Store insights
            insight = {
                'timestamp': datetime.now().isoformat(),
                'pair': pair,
                'patterns': patterns,
                'sentiment': sentiment,
                'price_range': {
                    'high': float(price_df['High'].max()),
                    'low': float(price_df['Low'].min()),
                    'current': float(price_df['Close'].iloc[-1])
                },
                'volatility': float(price_df['Close'].pct_change().std()),
                'volume_trend': self.analyze_volume_trend(price_df)
            }
            
            # Store in memory and file
            if pair not in self.learning_insights:
                self.learning_insights[pair] = []
            self.learning_insights[pair].append(insight)
            
            # Keep only last 100 insights per pair
            if len(self.learning_insights[pair]) > 100:
                self.learning_insights[pair] = self.learning_insights[pair][-100:]
            
            self.last_market_analysis[pair] = now
            self.save_learning_state()
            
            logger.info(f"Market analysis completed for {pair}: {len(patterns)} patterns detected")
            
        except Exception as e:
            logger.error(f"Error analyzing market patterns for {pair}: {e}")
    
    def extract_patterns(self, features_df):
        """Extract and analyze patterns from features"""
        patterns = {}
        
        try:
            # Technical patterns
            pattern_columns = [col for col in features_df.columns if any(
                pattern in col.lower() for pattern in [
                    'doji', 'hammer', 'engulfing', 'pin_bar', 'inside_bar',
                    'head_shoulders', 'double_top', 'double_bottom',
                    'wedge', 'flag', 'pennant', 'fakeout'
                ]
            )]
            
            latest = features_df.iloc[-1]
            active_patterns = []
            
            for col in pattern_columns:
                if latest[col] == 1:
                    active_patterns.append(col)
            
            patterns['technical'] = active_patterns
            
            # Trend analysis
            if 'trendline_up' in features_df.columns and latest['trendline_up'] == 1:
                patterns['trend'] = 'uptrend'
            elif 'trendline_down' in features_df.columns and latest['trendline_down'] == 1:
                patterns['trend'] = 'downtrend'
            else:
                patterns['trend'] = 'sideways'
            
            # Volatility analysis
            if 'vol_cluster_high' in features_df.columns and latest['vol_cluster_high'] == 1:
                patterns['volatility'] = 'high'
            else:
                patterns['volatility'] = 'normal'
            
            # Market regime
            regime_cols = [col for col in features_df.columns if 'market_regime_' in col]
            for col in regime_cols:
                if latest[col] == 1:
                    patterns['regime'] = col.replace('market_regime_', '')
                    break
            
            # RSI conditions
            if 'rsi_14' in features_df.columns:
                rsi = latest['rsi_14']
                if rsi < 30:
                    patterns['rsi'] = 'oversold'
                elif rsi > 70:
                    patterns['rsi'] = 'overbought'
                else:
                    patterns['rsi'] = 'neutral'
                
        except Exception as e:
            logger.error(f"Error extracting patterns: {e}")
        
        return patterns
    
    def analyze_volume_trend(self, price_df):
        """Analyze volume trend"""
        try:
            if 'Volume' not in price_df.columns:
                return 'unknown'
            
            recent_volume = price_df['Volume'].tail(10).mean()
            historical_volume = price_df['Volume'].tail(50).mean()
            
            if recent_volume > historical_volume * 1.2:
                return 'increasing'
            elif recent_volume < historical_volume * 0.8:
                return 'decreasing'
            else:
                return 'stable'
        except:
            return 'unknown'
    
    def learn_from_patterns(self):
        """Learn from accumulated pattern data"""
        try:
            # Check if we need to learn
            now = datetime.now()
            if self.last_pattern_learning:
                time_since_last = (now - self.last_pattern_learning).total_seconds()
                if time_since_last < self.pattern_learning_interval:
                    return  # Too soon to learn again
            
            logger.info("Starting pattern learning cycle")
            
            for pair, insights in self.learning_insights.items():
                if len(insights) < 10:  # Need minimum data
                    continue
                
                # Analyze pattern success rates
                pattern_success = self.analyze_pattern_success(pair, insights)
                
                # Update pattern memory
                if pair not in self.pattern_memory:
                    self.pattern_memory[pair] = {}
                
                self.pattern_memory[pair].update(pattern_success)
                
                # Log learning insights
                logger.info(f"Pattern learning for {pair}: {len(pattern_success)} patterns analyzed")
            
            self.last_pattern_learning = now
            self.save_learning_state()
            
        except Exception as e:
            logger.error(f"Error in pattern learning: {e}")
    
    def analyze_pattern_success(self, pair, insights):
        """Analyze which patterns lead to successful outcomes"""
        pattern_success = {}
        
        try:
            # Group insights by pattern type
            for insight in insights[-50:]:  # Last 50 insights
                patterns = insight.get('patterns', {})
                
                for pattern_type, pattern_data in patterns.items():
                    if pattern_type not in pattern_success:
                        pattern_success[pattern_type] = {
                            'count': 0,
                            'price_changes': [],
                            'success_rate': 0
                        }
                    
                    pattern_success[pattern_type]['count'] += 1
                    
                    # Calculate price change after pattern
                    price_range = insight.get('price_range', {})
                    if 'current' in price_range and 'high' in price_range:
                        price_change = (price_range['current'] - price_range['low']) / price_range['low']
                        pattern_success[pattern_type]['price_changes'].append(price_change)
            
            # Calculate success rates
            for pattern_type, data in pattern_success.items():
                if data['count'] > 0:
                    avg_change = sum(data['price_changes']) / len(data['price_changes'])
                    success_rate = len([c for c in data['price_changes'] if c > 0.01]) / len(data['price_changes'])
                    pattern_success[pattern_type]['success_rate'] = success_rate
                    pattern_success[pattern_type]['avg_price_change'] = avg_change
                
        except Exception as e:
            logger.error(f"Error analyzing pattern success for {pair}: {e}")
        
        return pattern_success
    
    def run_market_analysis_cycle(self):
        """Run market analysis for all pairs"""
        logger.info("Running market analysis cycle for all pairs")
        for pair in Config.TRADING_PAIRS:
            self.analyze_market_patterns(pair)
            time.sleep(2)  # Small delay between pairs
    
    def run_pattern_learning_cycle(self):
        """Run pattern learning cycle"""
        logger.info("Running pattern learning cycle")
        self.learn_from_patterns()

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
        """Calculate model performance metrics"""
        try:
            completed_trades = self.get_recent_outcomes(days=days)
            total_signals = len(completed_trades)
            wins = len(completed_trades[completed_trades['outcome'] == 'win'])
            win_rate = wins / total_signals if total_signals > 0 else 0
            avg_pnl = completed_trades['pnl'].mean() if 'pnl' in completed_trades.columns else 0
            # Calculate accuracy based on confidence vs actual outcomes
            high_confidence = completed_trades[completed_trades['confidence'] >= 0.8]
            high_confidence_signals = len(high_confidence)
            if high_confidence_signals > 0:
                high_confidence_wins = len(high_confidence[high_confidence['outcome'] == 'win'])
                accuracy = high_confidence_wins / high_confidence_signals
            else:
                accuracy = 0
            return {
                'total_signals': total_signals,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'accuracy': accuracy,
                'high_confidence_signals': high_confidence_signals
            }
        except Exception as e:
            logger.error(f"Error calculating model performance: {e}")
            return {
                'total_signals': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'accuracy': 0,
                'high_confidence_signals': 0
            }

    def should_retrain(self):
        """Check if model should be retrained based on performance and new data"""
        try:
            # Check if we have enough new outcomes
            recent_outcomes = self.get_recent_outcomes(days=7)
            if len(recent_outcomes) < self.retrain_threshold:
                return False
            
            # Check performance
            performance = self.calculate_model_performance(days=30)
            
            # Retrain if accuracy is below threshold
            if performance['accuracy'] < self.min_accuracy_threshold:
                logger.info(f"Retraining triggered: accuracy {performance['accuracy']:.2f} < {self.min_accuracy_threshold}")
                return True
            
            # Retrain if win rate is very low
            if performance['win_rate'] < 0.4:
                logger.info(f"Retraining triggered: win rate {performance['win_rate']:.2f} < 0.4")
                return True
            
            # Retrain if we haven't retrained in a while
            if self.last_retrain_date:
                days_since_retrain = (datetime.now() - self.last_retrain_date).days
                if days_since_retrain > 7:
                    logger.info(f"Retraining triggered: {days_since_retrain} days since last retrain")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking retrain conditions: {e}")
            return False

    def should_retrain_pair(self, pair, current_version_hash):
        """Check if a specific pair should be retrained"""
        try:
            # Check if model version has changed
            from models.model_registry import ModelRegistry
            registry = ModelRegistry()
            model_type = f'signal_model_{pair}'
            latest_model = registry.get_latest_model(model_type)
            if latest_model and latest_model.get('version') != current_version_hash:
                logger.info(f"Retraining {pair}: version mismatch ({current_version_hash} != {latest_model.get('version')})")
                return True
            # Check if we have enough new outcomes for this pair
            recent_outcomes = self.get_recent_outcomes(days=7)
            pair_outcomes = recent_outcomes[recent_outcomes['pair'] == pair]
            if len(pair_outcomes) < 10:  # Need at least 10 outcomes
                return False
            # Check pair-specific performance
            pair_performance = self.calculate_model_performance(days=30)
            # Retrain if pair performance is poor
            if pair_performance['win_rate'] < 0.4:
                logger.info(f"Retraining {pair}: win rate {pair_performance['win_rate']:.2f} < 0.4")
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking retrain conditions for {pair}: {e}")
            return False

    def retrain_model(self):
        """Retrain the model with new data"""
        try:
            logger.info("Starting model retraining...")
            # Get recent data for all pairs
            all_features = []
            all_targets = []
            for pair in Config.TRADING_PAIRS:
                try:
                    # Get price data
                    price_df = get_price_data(pair, interval='1h', lookback=200)
                    if price_df is None or price_df.empty:
                        continue
                    # Get news sentiment
                    from_date = price_df.index[-200].strftime('%Y-%m-%d')
                    to_date = price_df.index[-1].strftime('%Y-%m-%d')
                    sentiment = get_news_sentiment_with_cache(
                        self.get_pair_keywords(pair), from_date, to_date, pair
                    )
                    # Generate features
                    features_df = preprocess_features(price_df, sentiment or 0.0, use_multi_timeframe=True)
                    if features_df is not None and not features_df.empty:
                        # Prepare features and targets
                        feature_cols = [col for col in features_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'future_return', 'target']]
                        X = features_df[feature_cols].dropna()
                        y = features_df['target'].dropna()
                        # Align X and y
                        common_index = X.index.intersection(y.index)
                        X = X.loc[common_index]
                        y = y.loc[common_index]
                        if len(X) > 0:
                            all_features.append(X)
                            all_targets.append(y)
                            logger.info(f"Added {len(X)} samples for {pair}")
                except Exception as e:
                    logger.error(f"Error processing {pair} for retraining: {e}")
                    continue
            if not all_features:
                logger.error("No features available for retraining")
                return False
            # Combine all features
            combined_features = pd.concat(all_features, axis=0)
            combined_targets = pd.concat(all_targets, axis=0)
            # Align features and targets
            common_index = combined_features.index.intersection(combined_targets.index)
            combined_features = combined_features.loc[common_index]
            combined_targets = combined_targets.loc[common_index]
            logger.info(f"Retraining with {len(combined_features)} total samples")
            # Train model
            success = train_signal_model(combined_features, 'ALL_PAIRS', model_dir='models/saved_models/')
            if success:
                # Save retrain info
                with open('logs/last_retrain_info.json', 'w') as f:
                    json.dump({
                        'retrain_date': datetime.now().isoformat(),
                        'samples_used': len(combined_features),
                        'pairs_used': list(Config.TRADING_PAIRS),
                        'performance_before': self.calculate_model_performance(days=30)
                    }, f)
                self.save_last_retrain_date()
                self.last_retrain_date = datetime.now()
                logger.info("Model retraining completed successfully")
                return True
            else:
                logger.error("Model retraining failed")
                return False
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
        """Complete continuous learning cycle - enhanced with market analysis and pattern learning"""
        try:
            logger.info("Running enhanced continuous learning cycle...")
            
            # 1. Run market analysis for all pairs
            self.run_market_analysis_cycle()
            
            # 2. Run pattern learning
            self.run_pattern_learning_cycle()
            
            # 3. Check if retraining is needed
            if self.should_retrain():
                logger.info("Retraining conditions met, starting retrain...")
                success = self.retrain_model()
                if success:
                    logger.info("Continuous learning cycle completed successfully")
                else:
                    logger.error("Continuous learning cycle failed")
            else:
                logger.info("No retraining needed at this time")
            
            # 4. Log current performance
            performance = self.calculate_model_performance(days=30)
            logger.info(f"Current performance: {performance}")
            
            # 5. Generate summary report
            self.generate_tp_summary_report(days=30)
            
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
    parser.add_argument('--market-analysis', action='store_true', help='Run market analysis cycle')
    parser.add_argument('--pattern-learning', action='store_true', help='Run pattern learning cycle')
    args = parser.parse_args()
    
    cl = ContinuousLearning()
    
    if args.export_analytics:
        cl.export_analytics_to_csv()
    elif args.market_analysis:
        cl.run_market_analysis_cycle()
    elif args.pattern_learning:
        cl.run_pattern_learning_cycle()
    else:
        cl.run_continuous_learning_cycle() 