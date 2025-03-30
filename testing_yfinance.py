import unittest
from datetime import datetime, timedelta
from stock_data_fetcher import (
    get_historical_prices,
    get_analyst_ratings,
    get_price_reaction,
    calculate_technical_indicators
)

class TestStockDataFetcher(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.ticker = "MSFT"  # Using Microsoft as a reliable test case
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        self.start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        self.filing_date = "2023-10-24"  # Microsoft's last earnings date

    def test_get_historical_prices(self):
        """Test fetching historical price data."""
        prices = get_historical_prices(self.ticker, self.start_date, self.end_date)
        
        # Check if we got a valid response
        self.assertIsInstance(prices, dict)
        self.assertIn("ticker", prices)
        self.assertIn("dates", prices)
        self.assertIn("close", prices)
        self.assertIn("volume", prices)
        
        # Check data types and lengths
        self.assertEqual(prices["ticker"], self.ticker)
        self.assertIsInstance(prices["dates"], list)
        self.assertIsInstance(prices["close"], list)
        self.assertIsInstance(prices["volume"], list)
        
        # Check that we have data
        self.assertGreater(len(prices["dates"]), 0)
        self.assertGreater(len(prices["close"]), 0)
        self.assertGreater(len(prices["volume"]), 0)
        
        # Check data consistency
        self.assertEqual(len(prices["dates"]), len(prices["close"]))
        self.assertEqual(len(prices["dates"]), len(prices["volume"]))

    def test_get_analyst_ratings(self):
        """Test fetching analyst ratings."""
        ratings = get_analyst_ratings(self.ticker)
        
        # Check if we got a valid response
        self.assertIsInstance(ratings, dict)
        self.assertIn("ticker", ratings)
        
        # Check data types
        self.assertEqual(ratings["ticker"], self.ticker)
        self.assertIsInstance(ratings.get("mean_target"), (float, type(None)))
        self.assertIsInstance(ratings.get("median_target"), (float, type(None)))
        self.assertIsInstance(ratings.get("buy_ratings"), int)
        self.assertIsInstance(ratings.get("hold_ratings"), int)
        self.assertIsInstance(ratings.get("sell_ratings"), int)
        
        # Check rating counts are non-negative
        self.assertGreaterEqual(ratings.get("buy_ratings", 0), 0)
        self.assertGreaterEqual(ratings.get("hold_ratings", 0), 0)
        self.assertGreaterEqual(ratings.get("sell_ratings", 0), 0)

    def test_get_price_reaction(self):
        """Test price reaction analysis around filing date."""
        reaction = get_price_reaction(self.ticker, self.filing_date, days_before=5, days_after=5)
        
        # Check if we got a valid response
        self.assertIsInstance(reaction, dict)
        self.assertIn("ticker", reaction)
        self.assertIn("filing_date", reaction)
        self.assertIn("price_change_percent", reaction)
        
        # Check data types
        self.assertEqual(reaction["ticker"], self.ticker)
        self.assertEqual(reaction["filing_date"], self.filing_date)
        self.assertIsInstance(reaction["price_change_percent"], float)
        self.assertIsInstance(reaction["volume_change_percent"], float)
        
        # Check price data
        self.assertIsInstance(reaction["pre_filing_price"], float)
        self.assertIsInstance(reaction["post_filing_price"], float)
        self.assertGreater(reaction["pre_filing_price"], 0)
        self.assertGreater(reaction["post_filing_price"], 0)

    def test_calculate_technical_indicators(self):
        """Test technical indicator calculations."""
        # First get price data
        prices = get_historical_prices(self.ticker, self.start_date, self.end_date)
        self.assertIsInstance(prices, dict)
        
        # Calculate indicators
        indicators = calculate_technical_indicators(prices)
        
        # Check if we got a valid response
        self.assertIsInstance(indicators, dict)
        self.assertIn("ticker", indicators)
        self.assertIn("sma_20", indicators)
        self.assertIn("rsi", indicators)
        self.assertIn("macd", indicators)
        self.assertIn("bollinger_bands", indicators)
        
        # Check data types and lengths
        self.assertEqual(indicators["ticker"], self.ticker)
        self.assertIsInstance(indicators["sma_20"], list)
        self.assertIsInstance(indicators["rsi"], list)
        self.assertIsInstance(indicators["macd"], list)
        self.assertIsInstance(indicators["bollinger_bands"], dict)
        
        # Check Bollinger Bands
        bb = indicators["bollinger_bands"]
        self.assertIn("upper", bb)
        self.assertIn("middle", bb)
        self.assertIn("lower", bb)
        
        # Check that we have data
        self.assertGreater(len(indicators["sma_20"]), 0)
        self.assertGreater(len(indicators["rsi"]), 0)
        self.assertGreater(len(indicators["macd"]), 0)
        
        # Check data consistency
        self.assertEqual(len(indicators["sma_20"]), len(indicators["rsi"]))
        self.assertEqual(len(indicators["sma_20"]), len(indicators["macd"]))

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with invalid ticker
        prices = get_historical_prices("INVALID_TICKER", self.start_date, self.end_date)
        self.assertEqual(prices, {})
        
        # Test with invalid dates
        prices = get_historical_prices(self.ticker, "invalid-date", self.end_date)
        self.assertEqual(prices, {})
        
        # Test with invalid filing date
        reaction = get_price_reaction(self.ticker, "invalid-date", days_before=5, days_after=5)
        self.assertEqual(reaction, {})
        
        # Test technical indicators with invalid data
        indicators = calculate_technical_indicators({})
        self.assertEqual(indicators, {})

if __name__ == '__main__':
    unittest.main()
