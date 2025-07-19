import schedule
import time
import threading
from datetime import datetime
from config import Config
from utils.logger import get_logger
from pipeline.signal_pipeline import SignalPipeline
from models.advanced_continuous_learning import AdvancedContinuousLearning
import json
import numpy as np

class AdvancedPipelineOrchestrator:
    """
    Advanced Pipeline Orchestrator with sophisticated continuous learning integration
    
    Features:
    - Multi-level scheduling for different learning tasks
    - Adaptive learning intervals based on market conditions
    - Performance monitoring and optimization
    - Real-time analytics integration
    """
    
    def __init__(self, 
                 trading_pairs=None, 
                 session_times=None, 
                 signal_pipeline=None, 
                 continuous_learner=None, 
                 logger=None):
        self.trading_pairs = trading_pairs or Config.TRADING_PAIRS
        self.session_times = session_times or [
            '00:00', '01:30', '03:00',  # Asia
            '07:00', '08:30', '10:00',  # London
            '13:00', '14:30', '16:00',  # New York
            '18:00', '19:30', '21:00'   # Evening
        ]
        self.signal_pipeline = signal_pipeline or SignalPipeline()
        self.continuous_learner = continuous_learner or AdvancedContinuousLearning()
        self.logger = logger or get_logger('advanced_pipeline_orchestrator')
        
        # Advanced scheduling state
        self.is_running = False
        self.scheduler_thread = None
        self.learning_thread = None
        self.performance_monitor_thread = None
        
        # Learning intervals (in seconds)
        self.market_analysis_interval = 300  # 5 minutes
        self.pattern_learning_interval = 1800  # 30 minutes
        self.regime_analysis_interval = 3600  # 1 hour
        self.adaptive_learning_interval = 7200  # 2 hours
        
        # Performance tracking
        self.performance_metrics = {
            'market_analysis_count': 0,
            'pattern_learning_count': 0,
            'regime_analysis_count': 0,
            'adaptive_learning_count': 0,
            'last_market_analysis': None,
            'last_pattern_learning': None,
            'last_regime_analysis': None,
            'last_adaptive_learning': None
        }
        
        # Market condition tracking
        self.market_conditions = {
            'volatility_level': 'normal',
            'trend_strength': 'neutral',
            'volume_profile': 'normal',
            'last_condition_update': None
        }
        
    def start_advanced_continuous_learning(self):
        """Start the advanced continuous learning system"""
        try:
            self.logger.info("Starting advanced continuous learning system")
            
            # Start learning thread
            self.learning_thread = threading.Thread(
                target=self._run_advanced_learning_loop,
                daemon=True
            )
            self.learning_thread.start()
            
            # Start performance monitoring thread
            self.performance_monitor_thread = threading.Thread(
                target=self._run_performance_monitoring,
                daemon=True
            )
            self.performance_monitor_thread.start()
            
            self.is_running = True
            self.logger.info("Advanced continuous learning system started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting advanced continuous learning: {e}")
    
    def stop_advanced_continuous_learning(self):
        """Stop the advanced continuous learning system"""
        try:
            self.logger.info("Stopping advanced continuous learning system")
            self.is_running = False
            
            if self.learning_thread and self.learning_thread.is_alive():
                self.learning_thread.join(timeout=5)
            
            if self.performance_monitor_thread and self.performance_monitor_thread.is_alive():
                self.performance_monitor_thread.join(timeout=5)
            
            self.logger.info("Advanced continuous learning system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping advanced continuous learning: {e}")
    
    def _run_advanced_learning_loop(self):
        """Main advanced learning loop with adaptive scheduling"""
        try:
            self.logger.info("Advanced learning loop started")
            
            while self.is_running:
                current_time = datetime.now()
                
                # Market analysis (every 5 minutes)
                if (self.performance_metrics['last_market_analysis'] is None or
                    (current_time - self.performance_metrics['last_market_analysis']).total_seconds() >= self.market_analysis_interval):
                    
                    self._run_market_analysis_cycle()
                    self.performance_metrics['last_market_analysis'] = current_time
                    self.performance_metrics['market_analysis_count'] += 1
                
                # Pattern learning (every 30 minutes)
                if (self.performance_metrics['last_pattern_learning'] is None or
                    (current_time - self.performance_metrics['last_pattern_learning']).total_seconds() >= self.pattern_learning_interval):
                    
                    self._run_pattern_learning_cycle()
                    self.performance_metrics['last_pattern_learning'] = current_time
                    self.performance_metrics['pattern_learning_count'] += 1
                
                # Regime analysis (every 1 hour)
                if (self.performance_metrics['last_regime_analysis'] is None or
                    (current_time - self.performance_metrics['last_regime_analysis']).total_seconds() >= self.regime_analysis_interval):
                    
                    self._run_regime_analysis_cycle()
                    self.performance_metrics['last_regime_analysis'] = current_time
                    self.performance_metrics['regime_analysis_count'] += 1
                
                # Adaptive learning (every 2 hours)
                if (self.performance_metrics['last_adaptive_learning'] is None or
                    (current_time - self.performance_metrics['last_adaptive_learning']).total_seconds() >= self.adaptive_learning_interval):
                    
                    self._run_adaptive_learning_cycle()
                    self.performance_metrics['last_adaptive_learning'] = current_time
                    self.performance_metrics['adaptive_learning_count'] += 1
                
                # Sleep for a short interval
                time.sleep(60)  # Check every minute
                
        except Exception as e:
            self.logger.error(f"Error in advanced learning loop: {e}")
    
    def _run_market_analysis_cycle(self):
        """Run market analysis cycle for all pairs"""
        try:
            self.logger.info("Starting market analysis cycle")
            
            for pair in self.trading_pairs:
                try:
                    self.continuous_learner.advanced_market_analysis(pair)
                except Exception as e:
                    self.logger.error(f"Error analyzing market for {pair}: {e}")
            
            self.logger.info("Market analysis cycle completed")
            
        except Exception as e:
            self.logger.error(f"Error in market analysis cycle: {e}")
    
    def _run_pattern_learning_cycle(self):
        """Run pattern learning cycle"""
        try:
            self.logger.info("Starting pattern learning cycle")
            self.continuous_learner.adaptive_pattern_learning()
            self.logger.info("Pattern learning cycle completed")
            
        except Exception as e:
            self.logger.error(f"Error in pattern learning cycle: {e}")
    
    def _run_regime_analysis_cycle(self):
        """Run market regime analysis cycle"""
        try:
            self.logger.info("Starting regime analysis cycle")
            
            # Update market conditions
            self._update_market_conditions()
            
            # Adjust learning intervals based on market conditions
            self._adjust_learning_intervals()
            
            self.logger.info("Regime analysis cycle completed")
            
        except Exception as e:
            self.logger.error(f"Error in regime analysis cycle: {e}")
    
    def _run_adaptive_learning_cycle(self):
        """Run adaptive learning cycle"""
        try:
            self.logger.info("Starting adaptive learning cycle")
            
            # Generate comprehensive analytics report
            self.continuous_learner.generate_advanced_analytics_report()
            
            # Check if models need retraining
            if self.continuous_learner.should_retrain():
                self.logger.info("Triggering model retraining")
                self.continuous_learner.retrain_model()
            
            # Update performance metrics
            self._update_performance_metrics()
            
            self.logger.info("Adaptive learning cycle completed")
            
        except Exception as e:
            self.logger.error(f"Error in adaptive learning cycle: {e}")
    
    def _update_market_conditions(self):
        """Update market conditions based on recent analysis"""
        try:
            # Analyze recent learning insights to determine market conditions
            insights = self.continuous_learner.learning_insights
            
            if not insights:
                return
            
            # Calculate average volatility across all pairs
            volatility_levels = []
            trend_strengths = []
            volume_profiles = []
            
            for pair, pair_insights in insights.items():
                if len(pair_insights) > 0:
                    latest_insight = pair_insights[-1]
                    
                    # Extract volatility information
                    if 'volatility' in latest_insight:
                        for tf, vol_data in latest_insight['volatility'].items():
                            if 'current_level' in vol_data:
                                volatility_levels.append(vol_data['current_level'])
                    
                    # Extract trend information
                    if 'regime' in latest_insight:
                        for tf, regime_data in latest_insight['regime'].items():
                            if 'trend_strength' in regime_data:
                                trend_strengths.append(regime_data['trend_strength'])
                    
                    # Extract volume information
                    if 'volume_profile' in latest_insight:
                        for tf, vol_profile in latest_insight['volume_profile'].items():
                            if 'trend' in vol_profile:
                                volume_profiles.append(vol_profile['trend'])
            
            # Update market conditions
            if volatility_levels:
                avg_volatility = np.mean(volatility_levels)
                if avg_volatility > 0.02:
                    self.market_conditions['volatility_level'] = 'high'
                elif avg_volatility < 0.005:
                    self.market_conditions['volatility_level'] = 'low'
                else:
                    self.market_conditions['volatility_level'] = 'normal'
            
            if trend_strengths:
                avg_trend_strength = np.mean(trend_strengths)
                if avg_trend_strength > 0.7:
                    self.market_conditions['trend_strength'] = 'strong'
                elif avg_trend_strength < 0.3:
                    self.market_conditions['trend_strength'] = 'weak'
                else:
                    self.market_conditions['trend_strength'] = 'neutral'
            
            if volume_profiles:
                increasing_volume = sum(1 for vp in volume_profiles if vp == 'increasing')
                if increasing_volume > len(volume_profiles) * 0.7:
                    self.market_conditions['volume_profile'] = 'high'
                elif increasing_volume < len(volume_profiles) * 0.3:
                    self.market_conditions['volume_profile'] = 'low'
                else:
                    self.market_conditions['volume_profile'] = 'normal'
            
            self.market_conditions['last_condition_update'] = datetime.now().isoformat()
            
            self.logger.info(f"Market conditions updated: {self.market_conditions}")
            
        except Exception as e:
            self.logger.error(f"Error updating market conditions: {e}")
    
    def _adjust_learning_intervals(self):
        """Adjust learning intervals based on market conditions"""
        try:
            # Base intervals
            base_market_analysis = 300  # 5 minutes
            base_pattern_learning = 1800  # 30 minutes
            base_regime_analysis = 3600  # 1 hour
            base_adaptive_learning = 7200  # 2 hours
            
            # Adjust based on volatility
            if self.market_conditions['volatility_level'] == 'high':
                # More frequent analysis during high volatility
                self.market_analysis_interval = int(base_market_analysis * 0.5)  # 2.5 minutes
                self.pattern_learning_interval = int(base_pattern_learning * 0.7)  # 21 minutes
            elif self.market_conditions['volatility_level'] == 'low':
                # Less frequent analysis during low volatility
                self.market_analysis_interval = int(base_market_analysis * 2)  # 10 minutes
                self.pattern_learning_interval = int(base_pattern_learning * 1.5)  # 45 minutes
            else:
                # Normal intervals
                self.market_analysis_interval = base_market_analysis
                self.pattern_learning_interval = base_pattern_learning
            
            # Adjust based on trend strength
            if self.market_conditions['trend_strength'] == 'strong':
                # More frequent regime analysis during strong trends
                self.regime_analysis_interval = int(base_regime_analysis * 0.7)  # 42 minutes
            elif self.market_conditions['trend_strength'] == 'weak':
                # Less frequent regime analysis during weak trends
                self.regime_analysis_interval = int(base_regime_analysis * 1.5)  # 90 minutes
            else:
                self.regime_analysis_interval = base_regime_analysis
            
            self.logger.info(f"Learning intervals adjusted: market_analysis={self.market_analysis_interval}s, "
                           f"pattern_learning={self.pattern_learning_interval}s, "
                           f"regime_analysis={self.regime_analysis_interval}s")
            
        except Exception as e:
            self.logger.error(f"Error adjusting learning intervals: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Save performance metrics to file
            metrics_file = 'logs/advanced_learning_performance.json'
            with open(metrics_file, 'w') as f:
                json.dump({
                    'performance_metrics': self.performance_metrics,
                    'market_conditions': self.market_conditions,
                    'learning_intervals': {
                        'market_analysis': self.market_analysis_interval,
                        'pattern_learning': self.pattern_learning_interval,
                        'regime_analysis': self.regime_analysis_interval,
                        'adaptive_learning': self.adaptive_learning_interval
                    }
                }, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _run_performance_monitoring(self):
        """Run performance monitoring loop"""
        try:
            self.logger.info("Performance monitoring started")
            
            while self.is_running:
                try:
                    # Check system health
                    self._check_system_health()
                    
                    # Log performance statistics
                    self._log_performance_statistics()
                    
                    # Sleep for monitoring interval
                    time.sleep(300)  # Check every 5 minutes
                    
                except Exception as e:
                    self.logger.error(f"Error in performance monitoring: {e}")
                    time.sleep(60)  # Shorter sleep on error
                    
        except Exception as e:
            self.logger.error(f"Error in performance monitoring loop: {e}")
    
    def _check_system_health(self):
        """Check system health and performance"""
        try:
            # Check if learning is progressing
            current_time = datetime.now()
            
            # Check market analysis health
            if (self.performance_metrics['last_market_analysis'] and
                (current_time - self.performance_metrics['last_market_analysis']).total_seconds() > self.market_analysis_interval * 2):
                self.logger.warning("Market analysis appears to be delayed")
            
            # Check pattern learning health
            if (self.performance_metrics['last_pattern_learning'] and
                (current_time - self.performance_metrics['last_pattern_learning']).total_seconds() > self.pattern_learning_interval * 2):
                self.logger.warning("Pattern learning appears to be delayed")
            
            # Check memory usage
            self._check_memory_usage()
            
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
    
    def _check_memory_usage(self):
        """Check memory usage of learning data"""
        try:
            insights_count = sum(len(insights) for insights in self.continuous_learner.learning_insights.values())
            
            if insights_count > 1000:
                self.logger.warning(f"High memory usage detected: {insights_count} insights stored")
                
                # Clean up old insights if needed
                if insights_count > 2000:
                    self._cleanup_old_insights()
                    
        except Exception as e:
            self.logger.error(f"Error checking memory usage: {e}")
    
    def _cleanup_old_insights(self):
        """Clean up old insights to manage memory"""
        try:
            self.logger.info("Cleaning up old insights")
            
            for pair in self.continuous_learner.learning_insights:
                insights = self.continuous_learner.learning_insights[pair]
                if len(insights) > 100:  # Keep only last 100 insights per pair
                    self.continuous_learner.learning_insights[pair] = insights[-100:]
            
            self.continuous_learner.save_learning_state()
            self.logger.info("Old insights cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old insights: {e}")
    
    def _log_performance_statistics(self):
        """Log performance statistics"""
        try:
            stats = {
                'timestamp': datetime.now().isoformat(),
                'market_analysis_count': self.performance_metrics['market_analysis_count'],
                'pattern_learning_count': self.performance_metrics['pattern_learning_count'],
                'regime_analysis_count': self.performance_metrics['regime_analysis_count'],
                'adaptive_learning_count': self.performance_metrics['adaptive_learning_count'],
                'market_conditions': self.market_conditions,
                'learning_intervals': {
                    'market_analysis': self.market_analysis_interval,
                    'pattern_learning': self.pattern_learning_interval,
                    'regime_analysis': self.regime_analysis_interval,
                    'adaptive_learning': self.adaptive_learning_interval
                }
            }
            
            self.logger.info(f"Performance statistics: {json.dumps(stats, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Error logging performance statistics: {e}")
    
    def get_learning_status(self):
        """Get current learning system status"""
        try:
            return {
                'is_running': self.is_running,
                'performance_metrics': self.performance_metrics,
                'market_conditions': self.market_conditions,
                'learning_intervals': {
                    'market_analysis': self.market_analysis_interval,
                    'pattern_learning': self.pattern_learning_interval,
                    'regime_analysis': self.regime_analysis_interval,
                    'adaptive_learning': self.adaptive_learning_interval
                },
                'thread_status': {
                    'learning_thread_alive': self.learning_thread.is_alive() if self.learning_thread else False,
                    'performance_monitor_alive': self.performance_monitor_thread.is_alive() if self.performance_monitor_thread else False
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting learning status: {e}")
            return {}
    
    def run_signal_generation(self):
        """Run signal generation with integrated learning"""
        try:
            self.logger.info("Running signal generation with advanced learning integration")
            
            # Run market analysis before signal generation
            self._run_market_analysis_cycle()
            
            # Generate signals using the pipeline
            signals = self.signal_pipeline.generate_signals()
            
            # Log outcomes for learning
            for signal in signals:
                self.continuous_learner.log_signal_outcome(signal, {})
            
            self.logger.info(f"Signal generation completed: {len(signals)} signals generated")
            return signals
            
        except Exception as e:
            self.logger.error(f"Error in signal generation: {e}")
            return []

    def run_session(self, send_email_func=None):
        """Run a full signal generation session (all pairs) and return results. Compatible with legacy orchestrator tests."""
        return self.run_signal_generation()

# Legacy compatibility
PipelineOrchestrator = AdvancedPipelineOrchestrator 