import os
from config import Config
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import requests
import json
from pathlib import Path
from utils.logger import get_logger
import feedparser

logger = get_logger('news_fetch', log_file='logs/news_fetch.log')

newsapi = NewsApiClient(api_key=Config.NEWS_API_KEY)
analyzer = SentimentIntensityAnalyzer()

NEWS_CACHE_DIR = 'logs/news_cache/'
os.makedirs(NEWS_CACHE_DIR, exist_ok=True)

# RSS Feed URLs for forex news
RSS_FEEDS = [
    "https://www.fxstreet.com/rss/news",
    "https://www.forexcrunch.com/feed/"
]

def fetch_rss_news(keywords, hours_back=24):
    """Fetch news from multiple RSS feeds"""
    logger.info(f"Fetching RSS news for keywords: {keywords}")
    all_headlines = []
    
    for feed_url in RSS_FEEDS:
        try:
            logger.info(f"Trying RSS feed: {feed_url}")
            feed = feedparser.parse(feed_url)
            
            if hasattr(feed, 'status') and feed.status != 200:
                logger.warning(f"RSS feed {feed_url} returned status {feed.status}")
                continue
                
            feed_headlines = []
            for entry in feed.entries:
                try:
                    # Check if entry is recent enough
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        # Safely parse the published_parsed tuple
                        parsed = entry.published_parsed
                        if isinstance(parsed, tuple) and len(parsed) >= 6:
                            pub_date = datetime(parsed[0], parsed[1], parsed[2], 
                                              parsed[3], parsed[4], parsed[5])
                            if pub_date > datetime.now() - timedelta(hours=hours_back):
                                # Check if keywords match - use both title and summary
                                title = getattr(entry, 'title', '').lower()
                                summary = getattr(entry, 'summary', '').lower()
                                content = title + ' ' + summary
                                
                                # Check for any keyword match in the content
                                if any(kw.lower() in content for kw in keywords):
                                    feed_headlines.append(entry.title)
                                    logger.debug(f"RSS match found: '{entry.title}' for keywords {keywords}")
                except Exception as e:
                    logger.debug(f"Error processing RSS entry: {e}")
                    continue
            
            all_headlines.extend(feed_headlines)
            logger.info(f"RSS feed {feed_url}: {len(feed_headlines)} relevant headlines")
            
        except Exception as e:
            logger.warning(f"RSS feed error for {feed_url}: {e}")
            continue
    
    logger.info(f"RSS total headlines found: {len(all_headlines)}")
    return all_headlines

def fetch_economic_calendar(pair):
    """Fetch economic events for specific currency pairs using ForexFactory RSS only, with improved parsing."""
    logger.info(f"Fetching economic calendar for {pair}")
    calendar_feeds = [
        "https://www.forexfactory.com/ff_calendar_thisweek.xml"
    ]
    events = []
    currency_map = {
        'USDJPY': ['USD', 'JPY'],
        'BTCUSD': ['USD'],
        'USDCHF': ['USD', 'CHF'],
        'NZDJPY': ['JPY', 'NZD']
    }
    currencies = currency_map.get(pair, [])
    for feed_url in calendar_feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                # Try to extract event details from title/summary
                title = getattr(entry, 'title', '')
                summary = getattr(entry, 'summary', '')
                # ForexFactory calendar events often have currency code in title
                if any(currency in title for currency in currencies):
                    # Try to extract date/time and impact from summary
                    event = {'title': title}
                    if hasattr(entry, 'published'):
                        event['date'] = entry.published
                    if summary:
                        event['summary'] = summary
                        # Try to extract impact (e.g., 'High Impact')
                        if 'impact' in summary.lower():
                            event['impact'] = summary
                    events.append(event)
        except Exception as e:
            logger.warning(f"Economic calendar RSS error for {feed_url}: {e}")
    logger.info(f"Economic calendar events for {pair}: {len(events)}")
    return events

def save_news_cache(pair, date_str, news_data):
    cache_file = os.path.join(NEWS_CACHE_DIR, f"{pair}_{date_str}.json")
    with open(cache_file, 'w') as f:
        json.dump(news_data, f)
    logger.info(f"[news_cache] Saved news for {pair} {date_str} to {cache_file}")

def load_last_news_cache(pair):
    files = [f for f in os.listdir(NEWS_CACHE_DIR) if f.startswith(pair)]
    if not files:
        return None
    files.sort(reverse=True)
    latest_file = os.path.join(NEWS_CACHE_DIR, files[0])
    with open(latest_file, 'r') as f:
        return json.load(f)

def get_news_sentiment(keywords, from_date, to_date, pair=None, ttl_hours=12):
    logger.info(f"Starting news sentiment fetch for {keywords}, from={from_date}, to={to_date}, pair={pair}")
    all_headlines = []
    cache_key = f"{pair}_{from_date}_{to_date}" if pair else f"{from_date}_{to_date}"
    cache_file = os.path.join(NEWS_CACHE_DIR, f"{cache_key}.json")
    now = datetime.now().timestamp()
    # Try cache first with TTL
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            cache_time = cache_data.get('timestamp', 0)
            age_hours = (now - cache_time) / 3600
            if age_hours < ttl_hours:
                logger.info(f"[news_cache] Cache hit for {cache_key}, age: {age_hours:.2f}h < {ttl_hours}h TTL")
                return cache_data.get('sentiment_score', 0.0)
            else:
                logger.info(f"[news_cache] Cache expired for {cache_key}, age: {age_hours:.2f}h >= {ttl_hours}h TTL. Fetching fresh data.")
        except Exception as e:
            logger.warning(f"[news_cache] Error reading cache for {cache_key}: {e}")
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
                    logger.warning(f"NewsAPI rate limit hit for keyword '{kw}'. Trying Newsdata.io...")
                    break
                else:
                    logger.error(f"NewsAPI error for keyword '{kw}': {e}")
        if all_headlines:
            logger.info(f"NewsAPI: {len(all_headlines)} headlines found. Sample: {all_headlines[:1]}")
            sentiment_score = sum([analyzer.polarity_scores(headline)['compound'] for headline in all_headlines]) / len(all_headlines)
            logger.info(f"NewsAPI sentiment score: {sentiment_score:.4f}")
            # Save to cache with timestamp
            with open(cache_file, 'w') as f:
                json.dump({'sentiment_score': sentiment_score, 'headlines': all_headlines, 'timestamp': now}, f)
            logger.info(f"[news_cache] Saved news sentiment for {cache_key} to {cache_file}")
            return sentiment_score
    except Exception as e:
        logger.warning(f"NewsAPI failed: {e}")
    # Fallback: Try RSS as final fallback
    try:
        logger.info("Trying RSS as final fallback...")
        rss_headlines = fetch_rss_news(keywords, hours_back=24) # Fetch recent news
        if rss_headlines:
            logger.info(f"RSS found {len(rss_headlines)} headlines. Sample: {rss_headlines[:1]}")
            sentiment_scores = [analyzer.polarity_scores(headline)['compound'] for headline in rss_headlines]
            sentiment_score = sum(sentiment_scores) / len(sentiment_scores)
            # Save to cache with timestamp
            with open(cache_file, 'w') as f:
                json.dump({'sentiment_score': sentiment_score, 'headlines': rss_headlines, 'timestamp': now}, f)
            logger.info(f"[news_cache] Saved RSS sentiment for {cache_key} to {cache_file}")
            return sentiment_score
        else:
            logger.warning(f"RSS fallback found no headlines for {cache_key}")
    except Exception as e:
        logger.warning(f"RSS fallback failed: {e}")
    # If all fail, try last cache (even if expired)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            logger.info(f"[news_cache] Final fallback cache hit for {cache_key}")
            return cache_data.get('sentiment_score', 0.0)
        except Exception as e:
            logger.warning(f"[news_cache] Error reading fallback cache for {cache_key}: {e}")
    logger.warning(f"[news_cache] No news sentiment available for {cache_key}, returning 0.0")
    return 0.0

def get_news_sentiment_with_cache(keywords, from_date, to_date, pair):
    logger.info(f"IN get_news_sentiment_with_cache: Checking cache for pair={pair}, from={from_date}, to={to_date}")
    # Try to fetch news from API as usual
    try:
        sentiment = get_news_sentiment(keywords, from_date, to_date)
        if sentiment != 0.0:
            logger.info(f"[news_api] Got sentiment score {sentiment:.4f} for {pair} {to_date}")
            return sentiment
        else:
            logger.warning(f"[news_api] No news found for {pair} {to_date}. Trying cache fallback.")
            last_news = load_last_news_cache(pair)
            if last_news:
                logger.info(f"[news_cache] Using last cached news for {pair}")
                sentiment = sum([analyzer.polarity_scores(headline)['compound'] for headline in last_news]) / len(last_news)
                return sentiment
            else:
                logger.error(f"[news_cache] No cached news available for {pair}")
                return 0.0  # or N/A
    except Exception as e:
        logger.error(f"[news_api] Exception fetching news for {pair}: {e}", exc_info=True)
        last_news = load_last_news_cache(pair)
        if last_news:
            logger.info(f"[news_cache] Using last cached news for {pair} after exception")
            sentiment = sum([analyzer.polarity_scores(headline)['compound'] for headline in last_news]) / len(last_news)
            return sentiment
        else:
            logger.error(f"[news_cache] No cached news available for {pair} after exception")
            return 0.0  # or N/A

# Macroeconomic event detection

def get_macro_events(pair, from_date, to_date, ttl_hours=12):
    logger.info(f"Fetching macroeconomic events for {pair}, from={from_date}, to={to_date}")
    import json
    from datetime import datetime
    CACHE_DIR = 'logs/event_cache/'
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = f"{pair}_{from_date}_{to_date}"
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    now = datetime.now().timestamp()
    # Try cache first with TTL
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            cache_time = cache_data.get('timestamp', 0)
            age_hours = (now - cache_time) / 3600
            if age_hours < ttl_hours:
                logger.info(f"[event_cache] Cache hit for {cache_key}, age: {age_hours:.2f}h < {ttl_hours}h TTL")
                return cache_data.get('events', [])
            else:
                logger.info(f"[event_cache] Cache expired for {cache_key}, age: {age_hours:.2f}h >= {ttl_hours}h TTL. Fetching fresh data.")
        except Exception as e:
            logger.warning(f"[event_cache] Error reading cache for {cache_key}: {e}")
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
                'NZDJPY': ['JPY', 'NZD'],
            }
            currencies = currency_map.get(pair, [])
            relevant_events = [e for e in events if any(cur in e.get('country', '') or cur in e.get('event', '') for cur in currencies)]
            logger.info(f"Found {len(relevant_events)} macro events for {pair}.")
            # Save to cache with timestamp
            try:
                cache_data = {
                    'timestamp': now,
                    'events': relevant_events
                }
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f)
                logger.info(f"[event_cache] Saved macro events for {cache_key} to {cache_file}")
            except Exception as e:
                logger.warning(f"[event_cache] Error saving cache for {cache_key}: {e}")
            return relevant_events
        else:
            logger.warning(f"FMP economic calendar API returned status {resp.status_code}")
    except Exception as e:
        logger.warning(f"Failed to fetch macro events: {e}")
    # If all fail, try last cache (even if expired)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            logger.info(f"[event_cache] Final fallback cache hit for {cache_key}")
            return cache_data.get('events', [])
        except Exception as e:
            logger.warning(f"[event_cache] Error reading fallback cache for {cache_key}: {e}")
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
