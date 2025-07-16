import schedule
import time
import subprocess
from datetime import datetime
from utils.logger import get_logger

logger = get_logger('auto_retrain', log_file='logs/auto_retrain.log')

RETRAIN_SCRIPT = 'retrain_from_signal_log.py'


def retrain():
    logger.info(f"Retraining started at {datetime.utcnow().isoformat()} UTC")
    try:
        result = subprocess.run(['python', RETRAIN_SCRIPT], capture_output=True, text=True)
        logger.info(f"Retrain stdout: {result.stdout}")
        logger.info(f"Retrain stderr: {result.stderr}")
        logger.info(f"Retraining finished at {datetime.utcnow().isoformat()} UTC")
    except Exception as e:
        logger.error(f"Retraining failed: {e}")

def schedule_retraining():
    schedule.every().day.at("00:00").do(retrain)
    schedule.every().day.at("06:00").do(retrain)
    schedule.every().day.at("12:00").do(retrain)
    schedule.every().day.at("18:00").do(retrain)

if __name__ == "__main__":
    logger.info("Starting auto-retrain scheduler (4x daily)")
    schedule_retraining()
    retrain()  # Run once at startup
    while True:
        schedule.run_pending()
        time.sleep(30) 