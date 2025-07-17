import os
import json
from datetime import date

class AttemptLog:
    def __init__(self, log_path='logs/attempt_log.json', max_attempts=30):
        self.log_path = log_path
        self.max_attempts = max_attempts
        self._log = self._load()
        self._reset_if_new_day()

    def _load(self):
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    return json.load(f)
            except Exception:
                # Corrupted file, start fresh
                return {}
        return {}

    def _save(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w') as f:
            json.dump(self._log, f)

    def _reset_if_new_day(self):
        today = str(date.today())
        if list(self._log.keys()) != [today]:
            self._log.clear()
            self._log[today] = {}
            self._save()

    def increment(self, pair):
        today = str(date.today())
        if today not in self._log:
            self._log[today] = {}
        self._log[today][pair] = self._log[today].get(pair, 0) + 1
        self._save()

    def get(self, pair):
        today = str(date.today())
        return self._log.get(today, {}).get(pair, 0)

    def reset_for_pair(self, pair):
        today = str(date.today())
        if today in self._log and pair in self._log[today]:
            self._log[today][pair] = 0
            self._save()

    def reset_all(self):
        today = str(date.today())
        self._log[today] = {}
        self._save()

    def max_reached(self, pair):
        return self.get(pair) >= self.max_attempts

    def get_all(self):
        today = str(date.today())
        return self._log.get(today, {}).copy() 