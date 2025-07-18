import os
import json
from datetime import datetime
from pymongo import MongoClient
import logging

class AnalyticsLogger:
    """
    Logs signal analytics to MongoDB and a user-friendly log file.
    """
    def __init__(self, mongo_uri=None, db_name="forex_ai", collection_name="analytics_signals", log_file="logs/analytics_signals_pretty.log"):
        self.mongo_uri = mongo_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.db_name = db_name
        self.collection_name = collection_name
        self.log_file = log_file
        self.client = MongoClient(self.mongo_uri)
        self.collection = self.client[self.db_name][self.collection_name]
        self.logger = logging.getLogger("AnalyticsLogger")
        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_signal(self, pair, trend, key_levels, patterns, indicators, confluence, model_action, decision, confidence, reason, timestamp=None, extra=None):
        """
        Log a signal attempt with all context to MongoDB and pretty log file.
        """
        ts = timestamp or datetime.utcnow().isoformat()
        # Ensure no None or empty data
        def safe(val, default):
            if val is None:
                return default
            if isinstance(val, str) and not val.strip():
                return default
            if isinstance(val, (list, dict)) and not val:
                return default
            return val
        doc = {
            "pair": safe(pair, "N/A"),
            "timestamp": ts,
            "trend": safe(trend, "N/A"),
            "key_levels": safe(key_levels, {}),
            "patterns": safe(patterns, []),
            "indicators": safe(indicators, {}),
            "confluence": safe(confluence, {}),
            "model_action": safe(model_action, "N/A"),
            "decision": safe(decision, "N/A"),
            "confidence": confidence if confidence is not None else 0.0,
            "reason": safe(reason, "N/A"),
        }
        if extra:
            doc.update(extra)
        # Log to MongoDB
        try:
            self.collection.insert_one(doc)
        except Exception as e:
            self.logger.error(f"[MongoDB ERROR] {e}")
        # Log pretty message
        pretty = self.format_pretty_log(doc)
        self.logger.info(pretty)

    @staticmethod
    def format_pretty_log(doc):
        # Compose a user-friendly log message with safe defaults
        pair = doc.get('pair', 'N/A')
        timestamp = doc.get('timestamp', 'N/A')
        trend = doc.get('trend', 'N/A')
        key_levels = doc.get('key_levels', {})
        patterns = doc.get('patterns', [])
        indicators = doc.get('indicators', {})
        confluence = doc.get('confluence', {})
        model_action = doc.get('model_action', 'N/A')
        decision = doc.get('decision', 'N/A')
        confidence = doc.get('confidence', 0.0)
        reason = doc.get('reason', 'N/A')
        msg = (
            f"[{pair}] [{timestamp}] Trend: {trend}. "
            f"Key Levels: {key_levels if key_levels else '{}'}. "
            f"Patterns: {', '.join(patterns) if patterns else 'None'}. "
            f"Indicators: {json.dumps(indicators) if indicators else '{}'}. "
            f"Confluence: {json.dumps(confluence) if confluence else '{}'}. "
            f"Model: {model_action}. Decision: {decision} (Confidence: {confidence:.2f}). Reason: {reason}"
        )
        return msg

    def close(self):
        self.client.close() 