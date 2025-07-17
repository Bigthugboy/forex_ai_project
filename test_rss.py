#!/usr/bin/env python3
"""
Test script for RSS feed functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.fetch_news import fetch_rss_news, fetch_economic_calendar
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils.logger import get_logger

logger = get_logger('rss_test', log_file='logs/rss_test.log')

def test_rss_news():
    """Test RSS news fetching"""
    print("🧪 Testing RSS News Fetching...")
    
    # Test keywords for different pairs
    test_keywords = [
        ['USD/JPY', 'USDJPY', 'JPY'],
        ['BTC/USD', 'BTCUSD', 'Bitcoin', 'BTC'],
        ['USD/CHF', 'USDCHF', 'CHF'],
        ['JPY/NZD', 'JPYNZD', 'NZD']
    ]
    
    analyzer = SentimentIntensityAnalyzer()
    
    for i, keywords in enumerate(test_keywords):
        print(f"\n📰 Testing keywords: {keywords}")
        
        try:
            # Fetch RSS news
            headlines = fetch_rss_news(keywords, hours_back=24)
            
            if headlines:
                print(f"✅ Found {len(headlines)} headlines")
                print(f"📋 Sample headlines:")
                for j, headline in enumerate(headlines[:3]):
                    print(f"   {j+1}. {headline}")
                
                # Calculate sentiment
                sentiment_scores = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                
                print(f"📊 Average sentiment: {avg_sentiment:.4f}")
                print(f"📈 Sentiment range: {min(sentiment_scores):.4f} to {max(sentiment_scores):.4f}")
                
            else:
                print("❌ No headlines found")
                
        except Exception as e:
            print(f"❌ Error testing keywords {keywords}: {e}")

def test_economic_calendar():
    """Test economic calendar fetching"""
    print("\n📅 Testing Economic Calendar...")
    
    test_pairs = ['USDJPY', 'BTCUSD', 'USDCHF', 'JPYNZD']
    
    for pair in test_pairs:
        print(f"\n🏦 Testing economic calendar for {pair}")
        
        try:
            events = fetch_economic_calendar(pair)
            
            if events:
                print(f"✅ Found {len(events)} economic events")
                print(f"📋 Sample events:")
                for i, event in enumerate(events[:3]):
                    print(f"   {i+1}. {event['title']}")
                    if event.get('date'):
                        print(f"      Date: {event['date']}")
            else:
                print("❌ No economic events found")
                
        except Exception as e:
            print(f"❌ Error testing economic calendar for {pair}: {e}")

def test_rss_feed_availability():
    """Test if RSS feeds are accessible"""
    print("\n🔗 Testing RSS Feed Availability...")
    
    from data.fetch_news import RSS_FEEDS
    import feedparser
    
    for feed_url in RSS_FEEDS:
        print(f"\n🔍 Testing: {feed_url}")
        
        try:
            feed = feedparser.parse(feed_url)
            
            if hasattr(feed, 'status'):
                print(f"   Status: {feed.status}")
            else:
                print("   Status: Unknown")
                
            if hasattr(feed, 'entries'):
                print(f"   Entries: {len(feed.entries)}")
                
                if feed.entries:
                    print(f"   Latest entry: {feed.entries[0].title}")
                else:
                    print("   No entries found")
            else:
                print("   No entries attribute")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting RSS Feed Tests...")
    print("=" * 50)
    
    # Test RSS feed availability first
    test_rss_feed_availability()
    
    # Test RSS news fetching
    test_rss_news()
    
    # Test economic calendar
    test_economic_calendar()
    
    print("\n" + "=" * 50)
    print("✅ RSS Feed Tests Complete!")

if __name__ == "__main__":
    main() 