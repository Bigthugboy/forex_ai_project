import requests
from config import MAILGUN_API_KEY, MAILGUN_DOMAIN, MAILGUN_SENDER, MAILGUN_RECIPIENT
from utils.logger import get_logger

logger = get_logger('notify')

def send_email(subject, message, recipient=MAILGUN_RECIPIENT):
    """
    Send an email via Mailgun API.
    """
    url = f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages"
    auth = ("api", MAILGUN_API_KEY)
    data = {
        "from": MAILGUN_SENDER,
        "to": recipient,
        "subject": subject,
        "text": message
    }
    try:
        logger.info(f"Sending email: {subject} to {recipient}")
        response = requests.post(url, auth=auth, data=data)
        if response.status_code == 200:
            logger.info(f"Email sent: {subject}")
        else:
            logger.error(f"Failed to send email: {response.status_code} {response.text}")
    except Exception as e:
        logger.error(f"Exception sending email: {e}", exc_info=True)

if __name__ == "__main__":
    send_email("Test Signal Alert", "This is a test email from Forex AI Bot.") 