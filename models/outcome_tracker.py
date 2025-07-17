import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.logger import get_logger
from data.fetch_market import get_price_data
from models.continuous_learning import continuous_learner
import os
import json

logger = get_logger('outcome_tracker')

class OutcomeTracker:
    def __init__(self):
        self.pending_signals_file = 'logs/pending_signals.json'
        self.outcome_check_interval = 60  # Check outcomes every 60 minutes
        
    def log_pending_signal(self, signal_data):
        """Log a new signal for outcome tracking"""
        try:
            pending_signals = self.load_pending_signals()
            
            # Add signal to pending list
            signal_id = f"{signal_data['pair']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            pending_signals[signal_id] = {
                'signal_data': signal_data,
                'timestamp': datetime.now().isoformat(),
                'status': 'pending',
                'entry_time': datetime.now().isoformat()
            }
            
            self.save_pending_signals(pending_signals)
            logger.info(f"Logged pending signal: {signal_id}")
            
        except Exception as e:
            logger.error(f"Error logging pending signal: {e}")
    
    def load_pending_signals(self):
        """Load pending signals from file"""
        try:
            if os.path.exists(self.pending_signals_file):
                with open(self.pending_signals_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading pending signals: {e}")
            return {}
    
    def save_pending_signals(self, pending_signals):
        """Save pending signals to file"""
        try:
            os.makedirs('logs', exist_ok=True)
            with open(self.pending_signals_file, 'w') as f:
                json.dump(pending_signals, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving pending signals: {e}")
    
    def check_signal_outcomes(self):
        """Check outcomes for all pending signals"""
        try:
            pending_signals = self.load_pending_signals()
            if not pending_signals:
                return
            
            logger.info(f"Checking outcomes for {len(pending_signals)} pending signals")
            
            for signal_id, signal_info in list(pending_signals.items()):
                if signal_info['status'] != 'pending':
                    continue
                
                outcome = self.determine_signal_outcome(signal_info)
                if outcome:
                    # Log the outcome
                    self.log_signal_outcome(signal_info['signal_data'], outcome)
                    
                    # Remove from pending list
                    del pending_signals[signal_id]
                    logger.info(f"Signal {signal_id} completed: {outcome['outcome']}")
            
            # Save updated pending signals
            self.save_pending_signals(pending_signals)
            
        except Exception as e:
            logger.error(f"Error checking signal outcomes: {e}")
    
    def determine_signal_outcome(self, signal_info):
        """Determine if a signal has hit SL/TP levels"""
        try:
            signal_data = signal_info['signal_data']
            pair = signal_data['pair']
            signal_type = signal_data['signal']  # BUY or SELL
            entry_price = signal_data['entry']
            stop_loss = signal_data['stop_loss']
            take_profit_1 = signal_data['take_profit_1']
            take_profit_2 = signal_data['take_profit_2']
            take_profit_3 = signal_data['take_profit_3']
            
            # Fetch recent price data (last 5 hours since signal)
            price_df = get_price_data(pair, interval=1h
            if price_df is None or price_df.empty:
                return None
            
            # Get high and low prices since signal
            signal_time = datetime.fromisoformat(signal_info['timestamp'])
            recent_data = price_df[price_df.index > signal_time]
            
            if recent_data.empty:
                return None
            
            high_price = recent_data['High'].max()
            low_price = recent_data['Low'].min()
            current_price = recent_data['Close'].iloc[-1]
            # Determine outcome based on signal type
            if signal_type == 'BUY':
                # Check if TP levels were hit
                if high_price >= take_profit_3:
                    return self.create_outcome('win', take_profit_3, 'tp3', 300)
                elif high_price >= take_profit_2:
                    return self.create_outcome('win', take_profit_2, 'tp2', 200)
                elif high_price >= take_profit_1:
                    return self.create_outcome('win', take_profit_1, 'tp1', 100)
                elif low_price <= stop_loss:
                    return self.create_outcome('loss', stop_loss, 'sl', -100)
            elif signal_type == 'SELL':
                # Check if TP levels were hit (reverse logic for SELL)
                if low_price <= take_profit_3:
                    return self.create_outcome('win', take_profit_3, 'tp3', 300)
                elif low_price <= take_profit_2:
                    return self.create_outcome('win', take_profit_2, 'tp2', 200)
                elif low_price <= take_profit_1:
                    return self.create_outcome('win', take_profit_1, 'tp1', 100)
                elif high_price >= stop_loss:
                    return self.create_outcome('loss', stop_loss, 'sl', -100)
            # Check if 5 hours have passed (prediction window)
            hours_since_signal = (datetime.now() - signal_time).total_seconds() / 3600          if hours_since_signal >= 5
                # Calculate PnL at current price
                if signal_type == 'BUY':
                    pnl = (current_price - entry_price) *10 # Convert to pips
                else:
                    pnl = (entry_price - current_price) * 100
                
                if pnl > 0:
                    return self.create_outcome('win', current_price, datetime.now().isoformat(), pnl)
                else:
                    return self.create_outcome('loss', current_price, datetime.now().isoformat(), pnl)
            
            return None  # Still pending
            
        except Exception as e:
            logger.error(f"Error determining signal outcome: {e}")
            return None
    
    def create_outcome(self, outcome, exit_price, exit_time, pnl):
        """Create outcome data structure"""
        return {
            'outcome': outcome,
            'exit_price': exit_price,
            'exit_time': exit_time,
            'pnl': pnl,
            'exit_reason': 'SL/TP hit' if outcome == 'loss' else 'Time exit'
        }
    
    def log_signal_outcome(self, signal_data, outcome_data):
        """Log the outcome using continuous learning module"""
        try:
            continuous_learner.log_signal_outcome(signal_data, outcome_data)
            logger.info(f"Logged outcome: {signal_data['pair']} - {outcome_data['outcome']} ({outcome_data['pnl']:.1f} pips)")
        except Exception as e:
            logger.error(f"Error logging signal outcome: {e}")
    
    def run_outcome_check_cycle(self):
        """Complete outcome checking cycle"""
        try:
            logger.info("Running outcome check cycle...")
            self.check_signal_outcomes()
            
            # Log summary
            pending_signals = self.load_pending_signals()
            logger.info(f"Pending signals remaining: {len(pending_signals)}")
            
        except Exception as e:
            logger.error(f"Error in outcome check cycle: {e}")

# Global instance
outcome_tracker = OutcomeTracker() 