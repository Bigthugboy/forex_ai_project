import os
from config import Config
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

newsapi = NewsApiClient(api_key=Config.NEWS_API_KEY)
analyzer = SentimentIntensityAnalyzer()

def get_news_sentiment(keywords, from_date, to_date):
    """
    Fetch news headlines and compute average sentiment for given keywords.
    Args:
        keywords (list): List of keywords to search for.
        from_date (str): Start date (YYYY-MM-DD).
        to_date (str): End date (YYYY-MM-DD).
    Returns:
        float: Average sentiment compound score.
    """
    all_headlines = []
    for kw in keywords:
        articles = newsapi.get_everything(q=kw, from_param=from_date, to=to_date, language='en', sort_by='relevancy', page_size=20)
        for article in articles['articles']:
            all_headlines.append(article['title'])
    if not all_headlines:
        return 0.0
    scores = [analyzer.polarity_scores(headline)['compound'] for headline in all_headlines]
    return sum(scores) / len(scores)

import unittest
from unittest.mock import patch

class TestFetchNews(unittest.TestCase):
    def test_get_news_sentiment(self):
        from datetime import timedelta
        today = datetime.utcnow().date()
        from_date = (today - timedelta(days=2)).strftime('%Y-%m-%d')
        to_date = today.strftime('%Y-%m-%d')
        keywords = Config.NEWS_KEYWORDS[:2]
        # This will actually call the API; in production, mock this
        sentiment = get_news_sentiment(keywords, from_date, to_date)
        self.assertIsInstance(sentiment, float)
        self.assertGreaterEqual(sentiment, -1.0)
        self.assertLessEqual(sentiment, 1.0)

if __name__ == "__main__":
    unittest.main()
