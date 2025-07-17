import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from data.data_fetcher import DataFetcher

class TestDataFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = DataFetcher()

    @patch('yfinance.download')
    def test_get_price_data_valid(self, mock_download):
        # Mock a valid DataFrame with at least 100 rows
        df = pd.DataFrame({
            'Open': range(100),
            'High': range(1, 101),
            'Low': range(0, 100),
            'Close': range(2, 102),
            'Volume': [100]*100
        })
        mock_download.return_value = df
        result = self.fetcher.get_price_data('USDJPY')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ['Open', 'High', 'Low', 'Close', 'Volume'])

    @patch('yfinance.download')
    def test_get_price_data_invalid_pair(self, mock_download):
        result = self.fetcher.get_price_data('INVALID')
        self.assertIsNone(result)

    @patch('yfinance.download')
    def test_get_price_data_not_enough_data(self, mock_download):
        df = pd.DataFrame({
            'Open': [1], 'High': [2], 'Low': [0], 'Close': [1.5], 'Volume': [100]
        })
        mock_download.return_value = df
        result = self.fetcher.get_price_data('USDJPY')
        self.assertIsNone(result)

    @patch('data.data_fetcher.DataFetcher.fetch_rss_news')
    @patch('data.data_fetcher.NewsApiClient')
    def test_get_news_sentiment_newsapi(self, mock_newsapi, mock_rss):
        mock_newsapi.return_value.get_everything.return_value = {
            'articles': [{'title': 'Good news'}, {'title': 'Bad news'}]
        }
        self.fetcher.newsapi = mock_newsapi.return_value
        mock_rss.return_value = []
        score = self.fetcher.get_news_sentiment(['USD'], '2023-01-01', '2023-01-02')
        self.assertIsInstance(score, float)

    @patch('data.data_fetcher.DataFetcher.fetch_rss_news')
    def test_get_news_sentiment_rss(self, mock_rss):
        mock_rss.return_value = ['Great market', 'Terrible crash']
        score = self.fetcher.get_news_sentiment(['USD'], '2023-01-01', '2023-01-02')
        self.assertIsInstance(score, float)

    def test_fetch_rss_news_empty(self):
        headlines = self.fetcher.fetch_rss_news(['nonexistentkeyword'], hours_back=1)
        self.assertIsInstance(headlines, list)

    @patch.object(DataFetcher, 'get_price_data')
    def test_get_multi_timeframe_data(self, mock_get_price):
        df = pd.DataFrame({
            'Open': [1, 2], 'High': [2, 3], 'Low': [0, 1], 'Close': [1.5, 2.5], 'Volume': [100, 200]
        })
        mock_get_price.return_value = df
        result = self.fetcher.get_multi_timeframe_data('USDJPY')
        self.assertIsInstance(result, dict)
        self.assertIn('1h', result)
        self.assertIsInstance(result['1h'], pd.DataFrame)

if __name__ == '__main__':
    unittest.main() 