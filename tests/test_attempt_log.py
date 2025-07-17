import unittest
import os
import json
from datetime import date, timedelta
from utils.attempt_log import AttemptLog

class TestAttemptLog(unittest.TestCase):
    def setUp(self):
        self.test_log_path = 'logs/test_attempt_log.json'
        # Ensure clean state
        if os.path.exists(self.test_log_path):
            os.remove(self.test_log_path)
        self.attempt_log = AttemptLog(log_path=self.test_log_path, max_attempts=3)

    def tearDown(self):
        if os.path.exists(self.test_log_path):
            os.remove(self.test_log_path)

    def test_increment_and_get(self):
        self.assertEqual(self.attempt_log.get('PAIR1'), 0)
        self.attempt_log.increment('PAIR1')
        self.assertEqual(self.attempt_log.get('PAIR1'), 1)
        self.attempt_log.increment('PAIR1')
        self.assertEqual(self.attempt_log.get('PAIR1'), 2)

    def test_reset_for_pair(self):
        self.attempt_log.increment('PAIR2')
        self.attempt_log.increment('PAIR2')
        self.attempt_log.reset_for_pair('PAIR2')
        self.assertEqual(self.attempt_log.get('PAIR2'), 0)

    def test_reset_all(self):
        self.attempt_log.increment('PAIR1')
        self.attempt_log.increment('PAIR2')
        self.attempt_log.reset_all()
        self.assertEqual(self.attempt_log.get('PAIR1'), 0)
        self.assertEqual(self.attempt_log.get('PAIR2'), 0)

    def test_max_reached(self):
        for _ in range(3):
            self.attempt_log.increment('PAIR3')
        self.assertTrue(self.attempt_log.max_reached('PAIR3'))
        self.assertFalse(self.attempt_log.max_reached('PAIR1'))

    def test_get_all(self):
        self.attempt_log.increment('PAIR1')
        self.attempt_log.increment('PAIR2')
        all_attempts = self.attempt_log.get_all()
        self.assertEqual(all_attempts['PAIR1'], 1)
        self.assertEqual(all_attempts['PAIR2'], 1)

    def test_missing_file(self):
        # Should not raise error if file is missing
        if os.path.exists(self.test_log_path):
            os.remove(self.test_log_path)
        log = AttemptLog(log_path=self.test_log_path)
        self.assertEqual(log.get('ANY'), 0)

    def test_corrupted_file(self):
        # Write invalid JSON
        with open(self.test_log_path, 'w') as f:
            f.write('{corrupted: true')
        log = AttemptLog(log_path=self.test_log_path)
        self.assertEqual(log.get('ANY'), 0)

    def test_new_day_resets(self):
        self.attempt_log.increment('PAIR1')
        # Simulate a new day by manually editing the log file
        with open(self.test_log_path, 'r') as f:
            data = json.load(f)
        yesterday = str(date.today() - timedelta(days=1))
        data[yesterday] = {'PAIR1': 5}
        with open(self.test_log_path, 'w') as f:
            json.dump(data, f)
        # Reload should reset to today only
        log = AttemptLog(log_path=self.test_log_path)
        self.assertEqual(log.get('PAIR1'), 0)

if __name__ == '__main__':
    unittest.main() 