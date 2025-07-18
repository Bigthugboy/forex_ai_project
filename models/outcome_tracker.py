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
        """Determine if a signal has hit SL/TP levels, tracking all TPs hit."""
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
            price_df = get_price_data(pair, interval='1h')
            if price_df is None or price_df.empty:
                return None

            # Get high and low prices since signal
            signal_time = datetime.fromisoformat(signal_info['timestamp'])
            recent_data = price_df[price_df.index > signal_time]

            if recent_data.empty:
                return None

            high_price = recent_data['High'].cummax()
            low_price = recent_data['Low'].cummin()
            current_price = recent_data['Close'].iloc[-1]
            tps_hit = []
            sl_hit = False
            exit_time = None
            exit_price = None
            pnl = 0
            # Track all TPs hit before SL
            if signal_type == 'BUY':
                for idx, row in recent_data.iterrows():
                    if not sl_hit:
                        if row['High'] >= take_profit_1 and 'tp1' not in tps_hit:
                            tps_hit.append('tp1')
                        if row['High'] >= take_profit_2 and 'tp2' not in tps_hit:
                            tps_hit.append('tp2')
                        if row['High'] >= take_profit_3 and 'tp3' not in tps_hit:
                            tps_hit.append('tp3')
                        if row['Low'] <= stop_loss:
                            sl_hit = True
                            exit_time = idx.isoformat() if hasattr(idx, 'isoformat') else str(idx)
                            exit_price = row['Low']
                            pnl = (exit_price - entry_price) * 10000
                            break
                if not sl_hit and tps_hit:
                    # If no SL, but at least one TP hit, exit at last TP
                    last_tp = tps_hit[-1]
                    if last_tp == 'tp3':
                        exit_price = take_profit_3
                    elif last_tp == 'tp2':
                        exit_price = take_profit_2
                    elif last_tp == 'tp1':
                        exit_price = take_profit_1
                    exit_time = recent_data.index[-1].isoformat() if hasattr(recent_data.index[-1], 'isoformat') else str(recent_data.index[-1])
                    pnl = (exit_price - entry_price) * 10000
                elif not sl_hit and not tps_hit:
                    # Neither TP nor SL hit, exit at current price
                    exit_price = current_price
                    exit_time = recent_data.index[-1].isoformat() if hasattr(recent_data.index[-1], 'isoformat') else str(recent_data.index[-1])
                    pnl = (exit_price - entry_price) * 10000
            elif signal_type == 'SELL':
                for idx, row in recent_data.iterrows():
                    if not sl_hit:
                        if row['Low'] <= take_profit_1 and 'tp1' not in tps_hit:
                            tps_hit.append('tp1')
                        if row['Low'] <= take_profit_2 and 'tp2' not in tps_hit:
                            tps_hit.append('tp2')
                        if row['Low'] <= take_profit_3 and 'tp3' not in tps_hit:
                            tps_hit.append('tp3')
                        if row['High'] >= stop_loss:
                            sl_hit = True
                            exit_time = idx.isoformat() if hasattr(idx, 'isoformat') else str(idx)
                            exit_price = row['High']
                            pnl = (entry_price - exit_price) * 10000
                            break
                if not sl_hit and tps_hit:
                    last_tp = tps_hit[-1]
                    if last_tp == 'tp3':
                        exit_price = take_profit_3
                    elif last_tp == 'tp2':
                        exit_price = take_profit_2
                    elif last_tp == 'tp1':
                        exit_price = take_profit_1
                    exit_time = recent_data.index[-1].isoformat() if hasattr(recent_data.index[-1], 'isoformat') else str(recent_data.index[-1])
                    pnl = (entry_price - exit_price) * 10000
                elif not sl_hit and not tps_hit:
                    exit_price = current_price
                    exit_time = recent_data.index[-1].isoformat() if hasattr(recent_data.index[-1], 'isoformat') else str(recent_data.index[-1])
                    pnl = (entry_price - exit_price) * 10000
            # Outcome logic
            if sl_hit:
                return self.create_outcome('loss', exit_price, exit_time, pnl, tps_hit)
            elif tps_hit:
                return self.create_outcome('win', exit_price, exit_time, pnl, tps_hit)
            else:
                return self.create_outcome('pending', exit_price, exit_time, pnl, tps_hit)
        except Exception as e:
            logger.error(f"Error determining signal outcome: {e}")
            return None

    def create_outcome(self, outcome, exit_price, exit_time, pnl, tps_hit, timeframe='1h', days_to_outcome=0):
        """Create outcome data structure, including all TPs hit."""
        return {
            'outcome': outcome,
            'exit_price': exit_price,
            'exit_time': exit_time,
            'pnl': pnl,
            'exit_reason': 'SL/TP hit' if outcome == 'loss' else 'TP hit' if outcome == 'win' else 'Time exit',
            'tps_hit': tps_hit,
            'timeframe': timeframe,
            'days_to_outcome': days_to_outcome
        }

    def log_signal_outcome(self, signal_data, outcome_data):
        """Log the outcome using continuous learning module, including all TPs hit."""
        try:
            continuous_learner.log_signal_outcome(signal_data, outcome_data)
            logger.info(f"Logged outcome: {signal_data['pair']} - {outcome_data['outcome']} (TPs hit: {outcome_data.get('tps_hit', [])}, {outcome_data['pnl']:.1f} pips)")
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