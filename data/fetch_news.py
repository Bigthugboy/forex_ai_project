import os
from config import Config
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import requests
import json
from pathlib import Path
from utils.logger import get_logger

logger = get_logger('news_fetch', log_file='logs/news_fetch.log')

newsapi = NewsApiClient(api_key=Config.NEWS_API_KEY)
analyzer = SentimentIntensityAnalyzer()

def get_news_sentiment(keywords, from_date, to_date, pair=None):
    logger.info(f"Starting news sentiment fetch for {keywords}, from={from_date}, to={to_date}, pair={pair}")
    all_headlines = []
    # Try NewsAPI
    try:
        logger.info("Trying NewsAPI...")
        for kw in keywords:
            try:
                logger.info(f"Trying NewsAPI for keyword '{kw}'...")
                articles = newsapi.get_everything(q=kw, from_param=from_date, to=to_date, language='en', sort_by='relevancy', page_size=20)
                for article in articles['articles']:
                    all_headlines.append(article['title'])
                logger.info(f"NewsAPI: {len(articles['articles'])} headlines for '{kw}'. Sample: {articles['articles'][:3]}")
            except Exception as e:
                if hasattr(e, 'args') and e.args and 'rateLimited' in str(e.args[0]):
                    logger.warning(f"NewsAPI rate limit hit for keyword '{kw}'. Trying Alpha Vantage...")
                    # --- Try Alpha Vantage as backup ---
                    try:
                        av_key = Config.ALPHA_VANTAGE_API_KEY
                        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={kw}&apikey={av_key}"
                        logger.info(f"Trying Alpha Vantage for keyword '{kw}'...")
                        resp = requests.get(url, timeout=10)
                        if resp.status_code == 200:
                            data = resp.json()
                            if 'feed' in data:
                                av_headlines = [article['title'] for article in data['feed'] if 'title' in article]
                                all_headlines.extend(av_headlines)
                                logger.info(f"Alpha Vantage: {len(av_headlines)} headlines for '{kw}'. Sample: {av_headlines[:3]}")
                            else:
                                logger.info(f"Alpha Vantage: No 'feed' in response for '{kw}'. Trying FMP...")
                                # --- Try FMP as next fallback ---
                                try:
                                    fmp_key = Config.FMP_API_KEY
                                    fmp_url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={kw}&from={from_date}&to={to_date}&apikey={fmp_key}"
                                    logger.info(f"Trying FMP for keyword '{kw}'...")
                                    fmp_resp = requests.get(fmp_url, timeout=10)
                                    if fmp_resp.status_code == 200:
                                        fmp_data = fmp_resp.json()
                                        fmp_headlines = [article['title'] for article in fmp_data if 'title' in article]
                                        all_headlines.extend(fmp_headlines)
                                        logger.info(f"FMP: {len(fmp_headlines)} headlines for '{kw}'. Sample: {fmp_headlines[:3]}")
                                    else:
                                        logger.warning(f"FMP: Non-200 response for '{kw}': {fmp_resp.status_code}")
                                except Exception as e3:
                                    logger.error(f"FMP news sentiment error for '{kw}': {e3}")
                        else:
                            logger.info(f"Alpha Vantage: Non-200 response for '{kw}': {resp.status_code}. Trying FMP...")
                            # --- Try FMP as next fallback ---
                            try:
                                fmp_key = Config.FMP_API_KEY
                                fmp_url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={kw}&from={from_date}&to={to_date}&apikey={fmp_key}"
                                logger.info(f"Trying FMP for keyword '{kw}'...")
                                fmp_resp = requests.get(fmp_url, timeout=10)
                                if fmp_resp.status_code == 200:
                                    fmp_data = fmp_resp.json()
                                    fmp_headlines = [article['title'] for article in fmp_data if 'title' in article]
                                    all_headlines.extend(fmp_headlines)
                                    logger.info(f"FMP: {len(fmp_headlines)} headlines for '{kw}'. Sample: {fmp_headlines[:3]}")
                                else:
                                    logger.warning(f"FMP: Non-200 response for '{kw}': {fmp_resp.status_code}")
                            except Exception as e3:
                                logger.error(f"FMP news sentiment error for '{kw}': {e3}")
                    except Exception as e2:
                        logger.error(f"Alpha Vantage news sentiment error for '{kw}': {e2}")
                else:
                    logger.error(f"NewsAPI error for keyword '{kw}': {e}")
        if all_headlines:
            logger.info(f"NewsAPI: {len(all_headlines)} headlines found. Sample: {all_headlines[:1]}")
            sentiment_score = sum([analyzer.polarity_scores(headline)['compound'] for headline in all_headlines]) / len(all_headlines)
            logger.info(f"NewsAPI sentiment score: {sentiment_score:.4f}")
            return sentiment_score
    except Exception as e:
        logger.warning(f"NewsAPI failed: {e}")
    # Try Alpha Vantage
    try:
        logger.info("Trying Alpha Vantage...")
        av_key = Config.ALPHA_VANTAGE_API_KEY
        for kw in keywords:
            try:
                url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={kw}&apikey={av_key}"
                logger.info(f"Trying Alpha Vantage for keyword '{kw}'...")
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if 'feed' in data:
                        av_headlines = [article['title'] for article in data['feed'] if 'title' in article]
                        all_headlines.extend(av_headlines)
                        logger.info(f"Alpha Vantage: {len(av_headlines)} headlines for '{kw}'. Sample: {av_headlines[:3]}")
                    else:
                        logger.info(f"Alpha Vantage: No 'feed' in response for '{kw}'. Trying FMP...")
                        # --- Try FMP as next fallback ---
                        try:
                            fmp_key = Config.FMP_API_KEY
                            fmp_url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={kw}&from={from_date}&to={to_date}&apikey={fmp_key}"
                            logger.info(f"Trying FMP for keyword '{kw}'...")
                            fmp_resp = requests.get(fmp_url, timeout=10)
                            if fmp_resp.status_code == 200:
                                fmp_data = fmp_resp.json()
                                fmp_headlines = [article['title'] for article in fmp_data if 'title' in article]
                                all_headlines.extend(fmp_headlines)
                                logger.info(f"FMP: {len(fmp_headlines)} headlines for '{kw}'. Sample: {fmp_headlines[:3]}")
                            else:
                                logger.warning(f"FMP: Non-200 response for '{kw}': {fmp_resp.status_code}")
                        except Exception as e3:
                            logger.error(f"FMP news sentiment error for '{kw}': {e3}")
                else:
                    logger.info(f"Alpha Vantage: Non-200 response for '{kw}': {resp.status_code}. Trying FMP...")
                    # --- Try FMP as next fallback ---
                    try:
                        fmp_key = Config.FMP_API_KEY
                        fmp_url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={kw}&from={from_date}&to={to_date}&apikey={fmp_key}"
                        logger.info(f"Trying FMP for keyword '{kw}'...")
                        fmp_resp = requests.get(fmp_url, timeout=10)
                        if fmp_resp.status_code == 200:
                            fmp_data = fmp_resp.json()
                            fmp_headlines = [article['title'] for article in fmp_data if 'title' in article]
                            all_headlines.extend(fmp_headlines)
                            logger.info(f"FMP: {len(fmp_headlines)} headlines for '{kw}'. Sample: {fmp_headlines[:3]}")
                        else:
                            logger.warning(f"FMP: Non-200 response for '{kw}': {fmp_resp.status_code}")
                    except Exception as e3:
                        logger.error(f"FMP news sentiment error for '{kw}': {e3}")
            except Exception as e2:
                logger.error(f"Alpha Vantage news sentiment error for '{kw}': {e2}")
        if all_headlines:
            logger.info(f"Alpha Vantage: {len(all_headlines)} headlines found. Sample: {all_headlines[:1]}")
            sentiment_score = sum([analyzer.polarity_scores(headline)['compound'] for headline in all_headlines]) / len(all_headlines)
            logger.info(f"Alpha Vantage sentiment score: {sentiment_score:.4f}")
            return sentiment_score
    except Exception as e:
        logger.warning(f"Alpha Vantage failed: {e}")
    # Try FMP
    try:
        logger.info("Trying FMP...")
        fmp_key = Config.FMP_API_KEY
        for kw in keywords:
            try:
                fmp_url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={kw}&from={from_date}&to={to_date}&apikey={fmp_key}"
                logger.info(f"Trying FMP for keyword '{kw}'...")
                fmp_resp = requests.get(fmp_url, timeout=10)
                if fmp_resp.status_code == 200:
                    fmp_data = fmp_resp.json()
                    fmp_headlines = [article['title'] for article in fmp_data if 'title' in article]
                    all_headlines.extend(fmp_headlines)
                    logger.info(f"FMP: {len(fmp_headlines)} headlines for '{kw}'. Sample: {fmp_headlines[:3]}")
                else:
                    logger.warning(f"FMP: Non-200 response for '{kw}': {fmp_resp.status_code}")
            except Exception as e3:
                logger.error(f"FMP news sentiment error for '{kw}': {e3}")
        if all_headlines:
            logger.info(f"FMP: {len(all_headlines)} headlines found. Sample: {all_headlines[:1]}")
            sentiment_score = sum([analyzer.polarity_scores(headline)['compound'] for headline in all_headlines]) / len(all_headlines)
            logger.info(f"FMP sentiment score: {sentiment_score:.4f}")
            return sentiment_score
    except Exception as e:
        logger.warning(f"FMP failed: {e}")
    # Try Finnhub (if available)
    try:
        logger.info("Trying Finnhub...")
        # Finnhub news sentiment is not directly available via a single API call for multiple keywords.
        # It would require a separate Finnhub API call for each keyword.
        # For now, we'll skip this fallback as it's not a direct API integration.
        # If Finnhub news sentiment is needed, it would require a more complex setup.
        logger.warning("Finnhub news sentiment fallback skipped due to complexity.")
    except Exception as e:
        logger.warning(f"Finnhub failed: {e}")
    logger.warning("No news headlines found from any API. Returning neutral sentiment.")
    return 0.0

def get_news_sentiment_with_cache(keywords, from_date, to_date, pair):
    logger.info(f"IN get_news_sentiment_with_cache: Checking cache for pair={pair}, from={from_date}, to={to_date}")
    cache_path = Path('logs/news_cache.json')
    cache = {}
    now = datetime.utcnow()
    # Load cache if exists
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)
            logger.info(f"Loaded news cache from {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to load news cache: {e}")
            cache = {}
    # Check cache for this pair
    if pair in cache:
        cached = cache[pair]
        ts = datetime.strptime(cached['timestamp'], '%Y-%m-%dT%H:%M:%S')
        if (now - ts).total_seconds() < 4 * 3600:
            logger.info(f"Using cached news sentiment for {pair}: {cached['sentiment']}")
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
        logger.info(f"Updated news cache for {pair}")
    except Exception as e:
        logger.warning(f"Failed to save news cache: {e}")
    logger.info(f"IN get_news_sentiment_with_cache: Returning sentiment={sentiment} for pair={pair}")
    return sentiment

# Macroeconomic event detection

def get_macro_events(pair, from_date, to_date):
    logger.info(f"Fetching macroeconomic events for {pair}, from={from_date}, to={to_date}")
    # Use FMP or Finnhub economic calendar
    try:
        url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={from_date}&to={to_date}&apikey=YOUR_FMP_API_KEY"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            events = resp.json()
            # Filter for relevant currencies
            currency_map = {
                'USDJPY': ['USD', 'JPY'],
                'BTCUSD': ['USD'],
                'USDCHF': ['USD', 'CHF'],
                'JPYNZD': ['JPY', 'NZD'],
            }
            currencies = currency_map.get(pair, [])
            relevant_events = [e for e in events if any(cur in e.get('country', '') or cur in e.get('event', '') for cur in currencies)]
            logger.info(f"Found {len(relevant_events)} macro events for {pair}.")
            return relevant_events
        else:
            logger.warning(f"FMP economic calendar API returned status {resp.status_code}")
    except Exception as e:
        logger.warning(f"Failed to fetch macro events: {e}")
    return []

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
