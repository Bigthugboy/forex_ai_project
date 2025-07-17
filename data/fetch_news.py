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
        'JPYNZD': ['JPY', 'NZD']
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
                    logger.warning(f"NewsAPI rate limit hit for keyword '{kw}'. Trying Newsdata.io...")
                    break
                else:
                    logger.error(f"NewsAPI error for keyword '{kw}': {e}")
        if all_headlines:
            logger.info(f"NewsAPI: {len(all_headlines)} headlines found. Sample: {all_headlines[:1]}")
            sentiment_score = sum([analyzer.polarity_scores(headline)['compound'] for headline in all_headlines]) / len(all_headlines)
            logger.info(f"NewsAPI sentiment score: {sentiment_score:.4f}")
            return sentiment_score
    except Exception as e:
        logger.warning(f"NewsAPI failed: {e}")
    # Try CryptoPanic (crypto only)
    try:
        logger.info("Trying CryptoPanic...")
        # TODO: Remove hardcoded API keys after testing
        cryptopanic_key = '33aac378af50f293544f98578dee9b3ceae19162'
        CRYPTOPANIC_CACHE_DIR = 'logs/cryptopanic_cache/'
        os.makedirs(CRYPTOPANIC_CACHE_DIR, exist_ok=True)
        import time
        # Only use CryptoPanic for crypto pairs
        crypto_keywords = ['BTC', 'ETH', 'XRP', 'ADA', 'DOT', 'LINK', 'LTC', 'BCH', 'XLM', 'EOS']
        crypto_pairs = ['BTCUSD', 'ETHUSD', 'XRPUSD', 'ADAUSD', 'DOTUSD', 'LINKUSD', 'LTCUSD', 'BCHUSD', 'XLMUSD', 'EOSUSD']
        is_crypto_pair = any(crypto_kw in kw.upper() for crypto_kw in crypto_keywords + crypto_pairs)
        if not is_crypto_pair:
            logger.info("Skipping CryptoPanic for non-crypto pair")
        else:
            for kw in keywords:
                # Sanitize filename by replacing '/' with '_'
                safe_kw = kw.replace('/', '_')
                cache_file = os.path.join(CRYPTOPANIC_CACHE_DIR, f"{safe_kw}.json")
                cache_hit = False
                if os.path.exists(cache_file):
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                    last_time = cache_data.get('timestamp', 0)
                    now = time.time()
                    if now - last_time < 30:
                        cp_headlines = cache_data.get('headlines', [])
                        all_headlines.extend(cp_headlines)
                        logger.info(f"CryptoPanic: Cache hit for '{kw}' (age: {now - last_time:.1f}s, {len(cp_headlines)} headlines)")
                        cache_hit = True
                if not cache_hit:
                    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={cryptopanic_key}&currencies={kw}&filter=hot"
                    try:
                        resp = requests.get(url, timeout=10)
                        if resp.status_code == 200:
                            data = resp.json()
                            if 'results' in data:
                                cp_headlines = [article['title'] for article in data['results'] if 'title' in article]
                                all_headlines.extend(cp_headlines)
                                logger.info(f"CryptoPanic: {len(cp_headlines)} headlines for '{kw}'. Sample: {cp_headlines[:3]}")
                                # Save to cache
                                with open(cache_file, 'w') as f:
                                    json.dump({'timestamp': time.time(), 'headlines': cp_headlines}, f)
                                logger.info(f"CryptoPanic: Saved {len(cp_headlines)} headlines to cache for '{kw}'")
                        else:
                            logger.warning(f"CryptoPanic: Non-200 response for '{kw}': {resp.status_code}, message: {resp.text}")
                    except Exception as e:
                        logger.error(f"CryptoPanic error for '{kw}': {e}")
        if all_headlines:
            logger.info(f"CryptoPanic: {len(all_headlines)} headlines found. Sample: {all_headlines[:1]}")
            sentiment_score = sum([analyzer.polarity_scores(headline)['compound'] for headline in all_headlines]) / len(all_headlines)
            logger.info(f"CryptoPanic sentiment score: {sentiment_score:.4f}")
            return sentiment_score
    except Exception as e:
        logger.warning(f"CryptoPanic failed: {e}")
    
    # Try CoinGecko News (crypto only)
    try:
        logger.info("Trying CoinGecko News...")
        # Only use CoinGecko for crypto pairs
        crypto_keywords = ['BTC', 'ETH', 'XRP', 'ADA', 'DOT', 'LINK', 'LTC', 'BCH', 'XLM', 'EOS']
        crypto_pairs = ['BTCUSD', 'ETHUSD', 'XRPUSD', 'ADAUSD', 'DOTUSD', 'LINKUSD', 'LTCUSD', 'BCHUSD', 'XLMUSD', 'EOSUSD']
        is_crypto_pair = any(crypto_kw in kw.upper() for crypto_kw in crypto_keywords + crypto_pairs)
        if not is_crypto_pair:
            logger.info("Skipping CoinGecko News for non-crypto pair")
        else:
            for kw in keywords:
                # CoinGecko news API endpoint
                url = f"https://api.coingecko.com/api/v3/news?query={kw}"
                try:
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        data = resp.json()
                        if 'data' in data and isinstance(data['data'], list):
                            cg_headlines = [article.get('title', '') for article in data['data'] if article.get('title')]
                            all_headlines.extend(cg_headlines)
                            logger.info(f"CoinGecko News: {len(cg_headlines)} headlines for '{kw}'. Sample: {cg_headlines[:3]}")
                    else:
                        logger.warning(f"CoinGecko News: Non-200 response for '{kw}': {resp.status_code}, message: {resp.text}")
                except Exception as e:
                    logger.error(f"CoinGecko News error for '{kw}': {e}")
        if all_headlines:
            logger.info(f"CoinGecko News: {len(all_headlines)} headlines found. Sample: {all_headlines[:1]}")
            sentiment_score = sum([analyzer.polarity_scores(headline)['compound'] for headline in all_headlines]) / len(all_headlines)
            logger.info(f"CoinGecko News sentiment score: {sentiment_score:.4f}")
            return sentiment_score
    except Exception as e:
        logger.warning(f"CoinGecko News failed: {e}")
    # Try RSS as final fallback
    try:
        logger.info("Trying RSS as final fallback...")
        rss_headlines = fetch_rss_news(keywords, hours_back=24) # Fetch recent news
        if rss_headlines:
            logger.info(f"RSS found {len(rss_headlines)} headlines. Sample: {rss_headlines[:1]}")
            
            # Debug: Calculate and log individual sentiment scores
            sentiment_scores = []
            for i, headline in enumerate(rss_headlines):
                score = analyzer.polarity_scores(headline)['compound']
                sentiment_scores.append(score)
                logger.info(f"RSS headline {i+1}: '{headline}' -> sentiment: {score:.4f}")
            
            sentiment_score = sum(sentiment_scores) / len(sentiment_scores)
            logger.info(f"RSS individual scores: {sentiment_scores}")
            logger.info(f"RSS calculated average: {sentiment_score:.4f}")
            logger.info(f"RSS sentiment score: {sentiment_score:.4f}")
            return sentiment_score
        else:
            logger.warning("No news headlines found from any API. Returning neutral sentiment.")
            return 0.0
    except Exception as e:
        logger.warning(f"RSS failed: {e}")
        logger.warning("No news headlines found from any API. Returning neutral sentiment.")
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
