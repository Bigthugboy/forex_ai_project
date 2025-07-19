#!/usr/bin/env python3
"""
24/7 Continuous Learning Daemon
Runs in the background to continuously monitor markets and learn patterns
even when the main trading process is offline.
"""

import os
import sys
import time
import signal
import threading
import schedule
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import logging
import daemon
from daemon import pidfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils.logger import get_logger
from data.data_fetcher import DataFetcher
from data.preprocess import preprocess_features
from models.model_training_service import ModelTrainingService
from models.continuous_learning import continuous_learner
from models.outcome_tracker import OutcomeTracker
from utils.analytics_logger import AnalyticsLogger

class ContinuousLearningDaemon:
    def __init__(self):
        self.logger = get_logger('continuous_learning_daemon', log_file='logs/continuous_learning_daemon.log')
        self.data_fetcher = DataFetcher()
        self.model_trainer = ModelTrainingService()
        self.outcome_tracker = OutcomeTracker()
        self.analytics_logger = AnalyticsLogger()
        
        # Daemon configuration
        self.pid_file = '/tmp/forex_ai_daemon.pid'
        self.log_file = 'logs/continuous_learning_daemon.log'
        self.working_dir = os.getcwd()
        
        # Learning intervals
        self.market_analysis_interval = 300  # 5 minutes
        self.pattern_learning_interval = 1800  # 30 minutes
        self.model_update_interval = 3600  # 1 hour
        self.performance_check_interval = 7200  # 2 hours
        
        # State tracking
        self.running = False
        self.last_analysis = {}
        self.learning_insights = {}
        self.pattern_memory = {}
        
        # Initialize state files
        self.state_file = 'logs/daemon_state.json'
        self.insights_file = 'logs/learning_insights.json'
        self.pattern_file = 'logs/pattern_memory.json'
        
        self.load_state()
        
    def load_state(self):
        """Load daemon state from file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.last_analysis = state.get('last_analysis', {})
                    self.learning_insights = state.get('learning_insights', {})
                    self.pattern_memory = state.get('pattern_memory', {})
                self.logger.info("Loaded daemon state from file")
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
    
    def save_state(self):
        """Save daemon state to file"""
        try:
            state = {
                'last_analysis': self.last_analysis,
                'learning_insights': self.learning_insights,
                'pattern_memory': self.pattern_memory,
                'last_save': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
    
    def analyze_market_patterns(self, pair):
        """Analyze current market patterns and store insights"""
        try:
            self.logger.info(f"Analyzing market patterns for {pair}")
            
            # Fetch recent data
            price_df = self.data_fetcher.get_price_data(pair, interval='1h', lookback=48)
            if price_df is None or price_df.empty:
                return
            
            # Get news sentiment
            from_date = price_df.index[-48].strftime('%Y-%m-%d')
            to_date = price_df.index[-1].strftime('%Y-%m-%d')
            sentiment = self.data_fetcher.get_news_sentiment(
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
            
            self.last_analysis[pair] = datetime.now().isoformat()
            self.save_state()
            
            self.logger.info(f"Market analysis completed for {pair}: {len(patterns)} patterns detected")
            
        except Exception as e:
            self.logger.error(f"Error analyzing market patterns for {pair}: {e}")
    
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
            self.logger.error(f"Error extracting patterns: {e}")
        
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
            self.logger.info("Starting pattern learning cycle")
            
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
                self.logger.info(f"Pattern learning for {pair}: {len(pattern_success)} patterns analyzed")
            
            # Save pattern memory
            with open(self.pattern_file, 'w') as f:
                json.dump(self.pattern_memory, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error in pattern learning: {e}")
    
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
            self.logger.error(f"Error analyzing pattern success for {pair}: {e}")
        
        return pattern_success
    
    def update_models_if_needed(self):
        """Check if models need updating based on new insights"""
        try:
            self.logger.info("Checking if models need updates")
            
            # Check if we have enough new insights
            total_insights = sum(len(insights) for insights in self.learning_insights.values())
            
            if total_insights > 50:  # Threshold for model update
                self.logger.info("Triggering model update due to new insights")
                
                # Trigger continuous learning cycle
                continuous_learner.run_continuous_learning_cycle()
                
                # Clear old insights after model update
                for pair in self.learning_insights:
                    self.learning_insights[pair] = self.learning_insights[pair][-10:]  # Keep last 10
                
                self.save_state()
                
        except Exception as e:
            self.logger.error(f"Error updating models: {e}")
    
    def check_performance_metrics(self):
        """Check and log performance metrics"""
        try:
            self.logger.info("Checking performance metrics")
            
            # Get recent outcomes
            outcomes_df = continuous_learner.get_recent_outcomes(days=7)
            
            if not outcomes_df.empty:
                # Calculate metrics
                total_signals = len(outcomes_df)
                wins = len(outcomes_df[outcomes_df['outcome'] == 'win'])
                win_rate = wins / total_signals if total_signals > 0 else 0
                
                avg_pnl = outcomes_df['pnl'].mean() if 'pnl' in outcomes_df.columns else 0
                
                # Log performance
                self.logger.info(f"Performance (7 days): {total_signals} signals, "
                               f"{win_rate:.2%} win rate, {avg_pnl:.2f} avg PnL")
                
                # Store performance in insights
                performance_insight = {
                    'timestamp': datetime.now().isoformat(),
                    'total_signals': total_signals,
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl,
                    'period': '7_days'
                }
                
                if 'performance' not in self.learning_insights:
                    self.learning_insights['performance'] = []
                self.learning_insights['performance'].append(performance_insight)
                
                # Keep only last 20 performance records
                if len(self.learning_insights['performance']) > 20:
                    self.learning_insights['performance'] = self.learning_insights['performance'][-20:]
                
                self.save_state()
            
        except Exception as e:
            self.logger.error(f"Error checking performance: {e}")
    
    def get_pair_keywords(self, pair):
        """Get keywords for a specific pair"""
        if 'JPY' in pair:
            return ['USD/JPY', 'JPY/USD', 'Federal Reserve', 'Bank of Japan', 'interest rates']
        elif 'BTC' in pair:
            return ['Bitcoin', 'BTC', 'cryptocurrency', 'cryptodigital currency']
        else:
            return Config.NEWS_KEYWORDS
    
    def run_market_analysis(self):
        """Run market analysis for all pairs"""
        for pair in Config.TRADING_PAIRS:
            self.analyze_market_patterns(pair)
            time.sleep(10)  # Small delay between pairs
    
    def schedule_tasks(self):
        """Schedule all background tasks"""
        # Market analysis every 5 minutes
        schedule.every(self.market_analysis_interval).seconds.do(self.run_market_analysis)
        
        # Pattern learning every 30 minutes
        schedule.every(self.pattern_learning_interval).seconds.do(self.learn_from_patterns)
        
        # Model updates every hour
        schedule.every(self.model_update_interval).seconds.do(self.update_models_if_needed)
        
        # Performance checks every 2 hours
        schedule.every(self.performance_check_interval).seconds.do(self.check_performance_metrics)
        
        # State saving every 15 minutes
        schedule.every(900).seconds.do(self.save_state)
        
        self.logger.info("Scheduled all background tasks")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully")
        self.running = False
        self.save_state()
        sys.exit(0)
    
    def run(self):
        """Main daemon run loop"""
        self.logger.info("Starting Continuous Learning Daemon")
        
        # Set up signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Schedule tasks
        self.schedule_tasks()
        
        # Run initial analysis
        self.run_market_analysis()
        
        self.running = True
        
        # Main loop
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(10)
    
    def start_daemon(self):
        """Start the daemon process"""
        context = daemon.DaemonContext(
            working_directory=self.working_dir,
            umask=0o002,
            pidfile=pidfile.TimeoutPIDLockFile(self.pid_file)
        )
        
        with context:
            self.run()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Forex AI Continuous Learning Daemon')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    parser.add_argument('--stop', action='store_true', help='Stop daemon')
    parser.add_argument('--status', action='store_true', help='Check daemon status')
    parser.add_argument('--foreground', action='store_true', help='Run in foreground')
    
    args = parser.parse_args()
    
    daemon_instance = ContinuousLearningDaemon()
    
    if args.stop:
        # Stop daemon
        if os.path.exists(daemon_instance.pid_file):
            with open(daemon_instance.pid_file, 'r') as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)
            print(f"Sent SIGTERM to daemon (PID: {pid})")
        else:
            print("Daemon not running")
    
    elif args.status:
        # Check status
        if os.path.exists(daemon_instance.pid_file):
            with open(daemon_instance.pid_file, 'r') as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, 0)  # Check if process exists
                print(f"Daemon is running (PID: {pid})")
            except OSError:
                print("Daemon PID file exists but process is not running")
        else:
            print("Daemon is not running")
    
    elif args.daemon:
        # Start as daemon
        daemon_instance.start_daemon()
    
    elif args.foreground:
        # Run in foreground
        daemon_instance.run()
    
    else:
        # Default: run in foreground
        daemon_instance.run()

if __name__ == "__main__":
    main() 