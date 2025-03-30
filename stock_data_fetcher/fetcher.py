import yfinance as yf
import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from cachetools import TTLCache, cached
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataFetcher:
    def __init__(self):
        # Initialize caches with 1-hour TTL
        self.price_cache = TTLCache(maxsize=100, ttl=3600)
        self.ratings_cache = TTLCache(maxsize=100, ttl=3600)
        self.indicators_cache = TTLCache(maxsize=100, ttl=3600)

    def _get_ticker(self, ticker: str) -> Optional[yf.Ticker]:
        """Get a yfinance Ticker object with error handling."""
        try:
            return yf.Ticker(ticker)
        except Exception as e:
            logger.error(f"Error fetching ticker {ticker}: {str(e)}")
            return None

    @cached(cache=TTLCache(maxsize=100, ttl=3600))
    def get_historical_prices(self, ticker: str, start_date: str, end_date: str) -> Dict:
        """Fetch historical price data for a given ticker."""
        try:
            t = self._get_ticker(ticker)
            if not t:
                return {}

            # Convert dates to datetime objects
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")

            # Fetch historical data
            hist = t.history(start=start, end=end)
            
            if hist.empty:
                logger.warning(f"No historical data found for {ticker}")
                return {}

            # Convert to dictionary format
            return {
                "ticker": ticker,
                "dates": hist.index.strftime("%Y-%m-%d").tolist(),
                "open": hist["Open"].tolist(),
                "high": hist["High"].tolist(),
                "low": hist["Low"].tolist(),
                "close": hist["Close"].tolist(),
                "volume": hist["Volume"].tolist()
            }
        except Exception as e:
            logger.error(f"Error fetching historical prices for {ticker}: {str(e)}")
            return {}

    @cached(cache=TTLCache(maxsize=100, ttl=3600))
    def get_analyst_ratings(self, ticker: str) -> Dict:
        """Fetch analyst ratings and price targets for a given ticker."""
        try:
            t = self._get_ticker(ticker)
            if not t:
                return {
                    "ticker": ticker,
                    "mean_target": None,
                    "median_target": None,
                    "buy_ratings": 0,
                    "hold_ratings": 0,
                    "sell_ratings": 0,
                    "latest_rating": None,
                    "latest_target": None
                }

            # Initialize rating counts
            rating_counts = {
                "Buy": 0,
                "Hold": 0,
                "Sell": 0
            }

            # Get analyst recommendations
            recommendations = t.recommendations
            latest_rating = None
            latest_target = None

            if recommendations is not None and not recommendations.empty:
                # Handle different possible column names for ratings
                rating_columns = ["To Grade", "Grade", "Action"]
                rating_column = next((col for col in rating_columns if col in recommendations.columns), None)
                
                if rating_column:
                    rating_counts = recommendations[rating_column].value_counts().to_dict()
                    latest_rating = recommendations.iloc[0][rating_column]
                    latest_target = recommendations.iloc[0].get("Price Target", None)

            # Get price targets and other info
            info = t.info
            mean_target = None
            median_target = None

            if info:
                mean_target = info.get("targetMeanPrice")
                median_target = info.get("targetMedianPrice")

            # If no price targets in info, try to calculate from recommendations
            if (mean_target is None or median_target is None) and recommendations is not None and not recommendations.empty:
                price_targets = recommendations["Price Target"].dropna()
                if not price_targets.empty:
                    if mean_target is None:
                        mean_target = price_targets.mean()
                    if median_target is None:
                        median_target = price_targets.median()

            return {
                "ticker": ticker,
                "mean_target": float(mean_target) if mean_target is not None else None,
                "median_target": float(median_target) if median_target is not None else None,
                "buy_ratings": int(rating_counts.get("Buy", 0)),
                "hold_ratings": int(rating_counts.get("Hold", 0)),
                "sell_ratings": int(rating_counts.get("Sell", 0)),
                "latest_rating": latest_rating,
                "latest_target": float(latest_target) if latest_target is not None else None
            }
        except Exception as e:
            logger.error(f"Error fetching analyst ratings for {ticker}: {str(e)}")
            return {
                "ticker": ticker,
                "mean_target": None,
                "median_target": None,
                "buy_ratings": 0,
                "hold_ratings": 0,
                "sell_ratings": 0,
                "latest_rating": None,
                "latest_target": None
            }

    def get_price_reaction(self, ticker: str, filing_date: str, days_before: int = 5, days_after: int = 5) -> Dict:
        """Analyze price reaction around a filing date."""
        try:
            # Convert filing date to datetime
            filing_dt = datetime.strptime(filing_date, "%Y-%m-%d")
            
            # Calculate date range
            start_date = (filing_dt - timedelta(days=days_before)).strftime("%Y-%m-%d")
            end_date = (filing_dt + timedelta(days=days_after)).strftime("%Y-%m-%d")
            
            # Get historical prices for the period
            price_data = self.get_historical_prices(ticker, start_date, end_date)
            if not price_data:
                return {}

            # Calculate price changes
            prices = price_data["close"]
            if not prices:
                return {}

            pre_filing_price = prices[0]  # First price in the period
            post_filing_price = prices[-1]  # Last price in the period
            price_change = ((post_filing_price - pre_filing_price) / pre_filing_price) * 100

            return {
                "ticker": ticker,
                "filing_date": filing_date,
                "pre_filing_price": pre_filing_price,
                "post_filing_price": post_filing_price,
                "price_change_percent": price_change,
                "volume_change_percent": ((price_data["volume"][-1] - price_data["volume"][0]) / price_data["volume"][0]) * 100
            }
        except Exception as e:
            logger.error(f"Error analyzing price reaction for {ticker}: {str(e)}")
            return {}

    def calculate_technical_indicators(self, price_data: Dict) -> Dict:
        """Calculate technical indicators from price data using ta-lib."""
        try:
            if not price_data or "close" not in price_data:
                return {}

            # Convert price data to numpy arrays for ta-lib
            close = np.array(price_data["close"], dtype=float)
            high = np.array(price_data["high"], dtype=float)
            low = np.array(price_data["low"], dtype=float)
            volume = np.array(price_data["volume"], dtype=float)

            # Calculate technical indicators
            indicators = {
                "ticker": price_data["ticker"],
                "sma_20": talib.SMA(close, timeperiod=20).tolist(),
                "sma_50": talib.SMA(close, timeperiod=50).tolist(),
                "sma_200": talib.SMA(close, timeperiod=200).tolist(),
                "rsi": talib.RSI(close, timeperiod=14).tolist(),
                "macd": talib.MACD(close)[0].tolist(),  # MACD line
                "macd_signal": talib.MACD(close)[1].tolist(),  # Signal line
                "macd_hist": talib.MACD(close)[2].tolist(),  # MACD histogram
                "bollinger_bands": {
                    "upper": talib.BBANDS(close)[0].tolist(),
                    "middle": talib.BBANDS(close)[1].tolist(),
                    "lower": talib.BBANDS(close)[2].tolist()
                }
            }

            return indicators
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return {} 