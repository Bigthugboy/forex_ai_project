import os
import pandas as pd
from config import Config
from utils.logger import get_logger
from data.fetch_market import SYMBOL_MAP
import yfinance as yf
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import json
from datetime import datetime, timedelta
import feedparser

logger = get_logger('data_fetcher', log_file='logs/data_fetcher.log')

class DataFetcher:
    """
    Centralized service for all data fetching: price, news sentiment, multi-timeframe.
    """
    def __init__(self):
        self.newsapi = NewsApiClient(api_key=Config.NEWS_API_KEY)
        self.analyzer = SentimentIntensityAnalyzer()
        self.NEWS_CACHE_DIR = 'logs/news_cache/'
        os.makedirs(self.NEWS_CACHE_DIR, exist_ok=True)
        self.CRYPTOPANIC_CACHE_DIR = 'logs/cryptopanic_cache/'
        os.makedirs(self.CRYPTOPANIC_CACHE_DIR, exist_ok=True)
        self.RSS_FEEDS = [
            "https://www.fxstreet.com/rss/news",
            "https://www.forexcrunch.com/feed/"
        ]
        self.crypto_keywords = ['BTC', 'ETH', 'XRP', 'ADA', 'DOT', 'LINK', 'LTC', 'BCH', 'XLM', 'EOS']
        self.crypto_pairs = ['BTCUSD', 'ETHUSD', 'XRPUSD', 'ADAUSD', 'DOTUSD', 'LINKUSD', 'LTCUSD', 'BCHUSD', 'XLMUSD', 'EOSUSD']

    def get_price_data(self, pair, interval='1h', lookback=Config.LOOKBACK_PERIOD):
        """Fetch historical price data for a given trading pair."""
        min_lookback = 200
        if lookback < min_lookback:
            logger.info(f"[DEBUG] Increasing lookback from {lookback} to {min_lookback} for {pair}")
            lookback = min_lookback
        symbol = SYMBOL_MAP.get(pair)
        if symbol is None:
            logger.error(f"No Yahoo symbol mapping for pair: {pair}")
            return None
        try:
            data = yf.download(symbol, period=f'{lookback}d', interval=interval, auto_adjust=True)
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [str(col[-1]) for col in data.columns.values]
                standard_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if len(data.columns) == 5:
                    data.columns = standard_cols
                data = data.dropna()
                if len(data) < 100:
                    logger.error(f"Not enough data for {pair}: only {len(data)} rows after cleaning")
                    return None
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_cols:
                    if col not in data.columns or data[col].isnull().sum() > 0:
                        logger.error(f"Missing or NaN in required price column: {col} for {pair}")
                        return None
                return data
            else:
                logger.error(f"Data is None or empty for {pair} after fetch.")
                return None
        except Exception as e:
            logger.error(f"Failed to fetch price data for {pair}: {e}")
            return None

    def get_news_sentiment(self, keywords, from_date, to_date, pair=None):
        """Fetch news sentiment score for a given pair and date range, with cache/fallback."""
        all_headlines = []
        # Try NewsAPI
        try:
            for kw in keywords:
                articles = self.newsapi.get_everything(q=kw, from_param=from_date, to=to_date, language='en', sort_by='relevancy', page_size=20)
                for article in articles['articles']:
                    all_headlines.append(article['title'])
            if all_headlines:
                sentiment_score = sum([self.analyzer.polarity_scores(h)['compound'] for h in all_headlines]) / len(all_headlines)
                return sentiment_score
        except Exception as e:
            logger.warning(f"NewsAPI failed: {e}")
        # Try CryptoPanic (crypto only)
        try:
            is_crypto_pair = any(kw.upper() in self.crypto_keywords + self.crypto_pairs for kw in keywords)
            if is_crypto_pair:
                for kw in keywords:
                    safe_kw = kw.replace('/', '_')
                    cache_file = os.path.join(self.CRYPTOPANIC_CACHE_DIR, f"{safe_kw}.json")
                    cache_hit = False
                    if os.path.exists(cache_file):
                        with open(cache_file, 'r') as f:
                            cache_data = json.load(f)
                        last_time = cache_data.get('timestamp', 0)
                        now = datetime.now().timestamp()
                        if now - last_time < 30:
                            cp_headlines = cache_data.get('headlines', [])
                            all_headlines.extend(cp_headlines)
                            cache_hit = True
                    if not cache_hit:
                        url = f"https://cryptopanic.com/api/v1/posts/?auth_token=YOUR_CRYPTOPANIC_KEY&currencies={kw}&filter=hot"
                        try:
                            resp = requests.get(url, timeout=10)
                            if resp.status_code == 200:
                                data = resp.json()
                                if 'results' in data:
                                    cp_headlines = [article['title'] for article in data['results'] if 'title' in article]
                                    all_headlines.extend(cp_headlines)
                                    with open(cache_file, 'w') as f:
                                        json.dump({'timestamp': datetime.now().timestamp(), 'headlines': cp_headlines}, f)
                        except Exception as e:
                            logger.error(f"CryptoPanic error for '{kw}': {e}")
                if all_headlines:
                    sentiment_score = sum([self.analyzer.polarity_scores(h)['compound'] for h in all_headlines]) / len(all_headlines)
                    return sentiment_score
        except Exception as e:
            logger.warning(f"CryptoPanic failed: {e}")
        # Try RSS as fallback
        try:
            rss_headlines = self.fetch_rss_news(keywords, hours_back=24)
            if rss_headlines:
                sentiment_scores = [self.analyzer.polarity_scores(h)['compound'] for h in rss_headlines]
                sentiment_score = sum(sentiment_scores) / len(sentiment_scores)
                return sentiment_score
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"RSS failed: {e}")
            return 0.0

    def fetch_rss_news(self, keywords, hours_back=24):
        """Fetch news from multiple RSS feeds for given keywords and time window."""
        all_headlines = []
        for feed_url in self.RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                if hasattr(feed, 'status') and feed.status != 200:
                    continue
                for entry in feed.entries:
                    try:
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            parsed = entry.published_parsed
                            if isinstance(parsed, tuple) and len(parsed) >= 6:
                                pub_date = datetime(parsed[0], parsed[1], parsed[2], parsed[3], parsed[4], parsed[5])
                                if pub_date > datetime.now() - timedelta(hours=hours_back):
                                    title = getattr(entry, 'title', '').lower()
                                    summary = getattr(entry, 'summary', '').lower()
                                    content = title + ' ' + summary
                                    if any(kw.lower() in content for kw in keywords):
                                        all_headlines.append(entry.title)
                    except Exception:
                        continue
            except Exception:
                continue
        return all_headlines

    def get_multi_timeframe_data(self, symbol, lookback_days=60):
        """Fetch data for all timeframes for a given symbol."""
        timeframes = {'15m': '15m', '1h': '1h', '4h': '4h'}
        multi_tf_data = {}
        for tf_name, tf_interval in timeframes.items():
            try:
                data = self.get_price_data(symbol, interval=tf_interval, lookback=lookback_days)
                if data is not None and not data.empty:
                    multi_tf_data[tf_name] = data
            except Exception as e:
                logger.error(f"Error fetching {tf_name} data for {symbol}: {e}")
        return multi_tf_data 