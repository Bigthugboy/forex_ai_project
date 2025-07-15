import os
from config import Config
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import requests
import json
from pathlib import Path

newsapi = NewsApiClient(api_key=Config.NEWS_API_KEY)
analyzer = SentimentIntensityAnalyzer()

def get_news_sentiment(keywords, from_date, to_date):
    """
    Fetch news headlines and compute average sentiment for given keywords.
    Tries NewsAPI first, then Alpha Vantage if rate-limited. Returns 0.0 if all fail.
    Args:
        keywords (list): List of keywords to search for.
        from_date (str): Start date (YYYY-MM-DD).
        to_date (str): End date (YYYY-MM-DD).
    Returns:
        float: Average sentiment compound score.
    """
    all_headlines = []
    # --- Try NewsAPI first ---
    for kw in keywords:
        try:
            articles = newsapi.get_everything(q=kw, from_param=from_date, to=to_date, language='en', sort_by='relevancy', page_size=20)
            for article in articles['articles']:
                all_headlines.append(article['title'])
            print(f"[DEBUG] NewsAPI: {len(articles['articles'])} headlines for '{kw}'. Sample: {articles['articles'][:3]}")
        except Exception as e:
            if hasattr(e, 'args') and e.args and 'rateLimited' in str(e.args[0]):
                print(f"[Warning] NewsAPI rate limit hit for keyword '{kw}'. Trying Alpha Vantage...")
                # --- Try Alpha Vantage as backup ---
                try:
                    av_key = Config.ALPHA_VANTAGE_API_KEY
                    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={kw}&apikey={av_key}"
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        data = resp.json()
                        if 'feed' in data:
                            av_headlines = [article['title'] for article in data['feed'] if 'title' in article]
                            all_headlines.extend(av_headlines)
                            print(f"[DEBUG] Alpha Vantage: {len(av_headlines)} headlines for '{kw}'. Sample: {av_headlines[:3]}")
                        else:
                            print(f"[DEBUG] Alpha Vantage: No 'feed' in response for '{kw}'.")
                    else:
                        print(f"[DEBUG] Alpha Vantage: Non-200 response for '{kw}': {resp.status_code}")
                except Exception as e2:
                    print(f"[Error] Alpha Vantage news sentiment error for '{kw}': {e2}")
            else:
                print(f"[Error] NewsAPI error for keyword '{kw}': {e}")
    if not all_headlines:
        print(f"[WARNING] No headlines found for keywords {keywords} from NewsAPI or Alpha Vantage.")
        return 0.0
    scores = [analyzer.polarity_scores(headline)['compound'] for headline in all_headlines]
    return sum(scores) / len(scores)

def get_news_sentiment_with_cache(keywords, from_date, to_date, pair):
    """
    Fetch news sentiment for a pair, using a 4-hour cache to avoid API rate limits.
    Args:
        keywords (list): List of keywords to search for.
        from_date (str): Start date (YYYY-MM-DD).
        to_date (str): End date (YYYY-MM-DD).
        pair (str): Trading pair (e.g., 'USDJPY').
    Returns:
        float: Average sentiment compound score.
    """
    cache_path = Path('logs/news_cache.json')
    cache = {}
    now = datetime.utcnow()
    # Load cache if exists
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to load news cache: {e}")
            cache = {}
    # Check cache for this pair
    if pair in cache:
        cached = cache[pair]
        ts = datetime.strptime(cached['timestamp'], '%Y-%m-%dT%H:%M:%S')
        if (now - ts).total_seconds() < 4 * 3600:
            print(f"[DEBUG] Using cached news sentiment for {pair}: {cached['sentiment']}")
            return cached['sentiment']
    # If not cached or expired, fetch new
    sentiment = get_news_sentiment(keywords, from_date, to_date)
    # Update cache
    cache[pair] = {
        'sentiment': sentiment,
        'timestamp': now.strftime('%Y-%m-%dT%H:%M:%S')
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"[Warning] Failed to save news cache: {e}")
    return sentiment

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
