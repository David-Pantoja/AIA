import unittest
from datetime import datetime, timedelta
from insight_generator import InsightGenerator

class TestInsightGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.ticker = "AAPL"  # Using Apple as a reliable test case
        self.filing_date = "2023-11-02"  # Apple's Q4 2023 earnings date
        self.insight_generator = InsightGenerator()

    def test_prepare_financial_data(self):
        """Test financial data preparation."""
        financial_data = self.insight_generator._prepare_financial_data(self.ticker, self.filing_date)
        
        # Check if we got a valid response
        self.assertIsInstance(financial_data, dict)
        if financial_data:  # Only check structure if we got data
            self.assertIn("ticker", financial_data)
            self.assertIn("filing_date", financial_data)
            self.assertIn("recent_filings", financial_data)
            self.assertIn("price_reaction", financial_data)
            self.assertIn("analyst_ratings", financial_data)
            self.assertIn("technical_indicators", financial_data)
            
            # Check data types
            self.assertEqual(financial_data["ticker"], self.ticker)
            self.assertEqual(financial_data["filing_date"], self.filing_date)
            self.assertIsInstance(financial_data["recent_filings"], list)
            self.assertIsInstance(financial_data["price_reaction"], dict)
            self.assertIsInstance(financial_data["analyst_ratings"], dict)
            self.assertIsInstance(financial_data["technical_indicators"], dict)

    def test_create_analysis_prompt(self):
        """Test prompt creation."""
        # First get financial data
        financial_data = self.insight_generator._prepare_financial_data(self.ticker, self.filing_date)
        if not financial_data:
            self.skipTest("No financial data available for testing")
        
        # Create prompt
        prompt = self.insight_generator._create_analysis_prompt(financial_data)
        
        # Check prompt structure
        self.assertIsInstance(prompt, str)
        self.assertIn(self.ticker, prompt)
        self.assertIn("Financial Metrics:", prompt)
        self.assertIn("Management Discussion:", prompt)
        self.assertIn("Technical Indicators:", prompt)
        self.assertIn("Analyst Ratings:", prompt)
        self.assertIn("Price Reaction:", prompt)

    def test_generate_insights(self):
        """Test insight generation."""
        insights = self.insight_generator.generate_insights(self.ticker, self.filing_date)
        
        # Check if we got a valid response
        self.assertIsInstance(insights, dict)
        if insights:  # Only check structure if we got data
            self.assertIn("ticker", insights)
            self.assertIn("filing_date", insights)
            self.assertIn("generated_at", insights)
            self.assertIn("insights", insights)
            
            # Check data types
            self.assertEqual(insights["ticker"], self.ticker)
            self.assertEqual(insights["filing_date"], self.filing_date)
            self.assertIsInstance(insights["generated_at"], str)
            self.assertIsInstance(insights["insights"], dict)
            
            # Check insights structure
            insight_data = insights["insights"]
            self.assertIn("summary", insight_data)
            self.assertIn("financial_health", insight_data)
            self.assertIn("recommendation", insight_data)
            self.assertIn("risk_factors", insight_data)
            self.assertIn("technical_analysis", insight_data)
            self.assertIn("market_sentiment", insight_data)
            
            # Check recommendation structure
            recommendation = insight_data["recommendation"]
            self.assertIn("action", recommendation)
            self.assertIn("buy_probability", recommendation)
            self.assertIn("hold_probability", recommendation)
            self.assertIn("sell_probability", recommendation)
            self.assertIn("confidence_level", recommendation)
            
            # Check probability values
            self.assertGreaterEqual(recommendation["buy_probability"], 0)
            self.assertLessEqual(recommendation["buy_probability"], 1)
            self.assertGreaterEqual(recommendation["hold_probability"], 0)
            self.assertLessEqual(recommendation["hold_probability"], 1)
            self.assertGreaterEqual(recommendation["sell_probability"], 0)
            self.assertLessEqual(recommendation["sell_probability"], 1)
            self.assertGreaterEqual(recommendation["confidence_level"], 0)
            self.assertLessEqual(recommendation["confidence_level"], 1)

    def test_get_latest_insights(self):
        """Test getting latest insights."""
        insights = self.insight_generator.get_latest_insights(self.ticker)
        
        # Check if we got a valid response
        self.assertIsInstance(insights, dict)
        if insights:  # Only check structure if we got data
            self.assertIn("ticker", insights)
            self.assertIn("filing_date", insights)
            self.assertIn("generated_at", insights)
            self.assertIn("insights", insights)
            
            # Check data types
            self.assertEqual(insights["ticker"], self.ticker)
            self.assertIsInstance(insights["filing_date"], str)
            self.assertIsInstance(insights["generated_at"], str)
            self.assertIsInstance(insights["insights"], dict)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with invalid ticker
        insights = self.insight_generator.generate_insights("INVALID_TICKER", self.filing_date)
        self.assertEqual(insights, {})
        
        # Test with invalid date
        insights = self.insight_generator.generate_insights(self.ticker, "invalid-date")
        self.assertEqual(insights, {})
        
        # Test with invalid ticker for latest insights
        insights = self.insight_generator.get_latest_insights("INVALID_TICKER")
        self.assertEqual(insights, {})

if __name__ == '__main__':
    unittest.main() 