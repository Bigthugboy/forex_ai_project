import schedule
import time
from datetime import datetime
from config import Config
from utils.logger import get_logger
from pipeline.signal_pipeline import SignalPipeline
from models.continuous_learning import ContinuousLearning
import json

class PipelineOrchestrator:
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
            '13:00', '14:30', '16:00',  # NY
            '22:00'                    # NY close
        ]
        self.signal_pipeline = signal_pipeline or SignalPipeline(mode='live')
        self.continuous_learner = continuous_learner or ContinuousLearning()
        self.logger = logger or get_logger('orchestrator', log_file='logs/orchestrator.log')

    def run_session(self, send_email_func=None):
        self.logger.info(f"Running orchestrated session for pairs: {self.trading_pairs}")
        results = {}
        for pair in self.trading_pairs:
            result = self.signal_pipeline.run(pair, send_email_func=send_email_func)
            results[pair] = result
            # --- Analytics logging for study ---
            analytics_log = {
                'timestamp': datetime.utcnow().isoformat(),
                'pair': pair,
                'signal': result.get('signal', {}).get('signal') if result.get('signal') else None,
                'confidence': result.get('signal', {}).get('confidence') if result.get('signal') else None,
                'confluence': result.get('signal', {}).get('confluence') if result.get('signal') else None,
                'confluence_factors': result.get('signal', {}).get('confluence_factors') if result.get('signal') else None,
                'all_factors': result.get('signal', {}).get('factors') if result.get('signal') and 'factors' in result['signal'] else None,
                'error': result.get('error')
            }
            with open('logs/analytics_signals.jsonl', 'a') as f:
                f.write(json.dumps(analytics_log) + '\n')
        self.logger.info(f"Session results: {results}")
        return results

    def schedule_sessions(self, send_email_func=None):
        for t in self.session_times:
            schedule.every().day.at(t).do(self.run_session, send_email_func=send_email_func)
        self.logger.info(f"Scheduled sessions at: {self.session_times}")

    def run_continuous_learning(self):
        self.logger.info("Running orchestrated continuous learning cycle...")
        self.continuous_learner.run_continuous_learning_cycle()

    def run_forever(self, send_email_func=None):
        self.logger.info("Starting orchestrator main loop...")
        self.schedule_sessions(send_email_func=send_email_func)
        self.run_session(send_email_func=send_email_func)  # Run once at startup
        while True:
            schedule.run_pending()
            time.sleep(5) 