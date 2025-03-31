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
                    "latest_target": None,
                    "raw_recommendations": {
                        "strongBuy": 0,
                        "buy": 0,
                        "hold": 0,
                        "sell": 0,
                        "strongSell": 0
                    }
                }

            # Get analyst recommendations
            recommendations = t.get_recommendations()
            latest_rating = None
            latest_target = None

            if recommendations is not None and not recommendations.empty:
                # Calculate rating counts from the recommendations DataFrame
                # Weight strongBuy and strongSell more heavily
                rating_counts = {
                    "Buy": int(recommendations["buy"].sum() + recommendations["strongBuy"].sum() * 1.5),
                    "Hold": int(recommendations["hold"].sum()),
                    "Sell": int(recommendations["sell"].sum() + recommendations["strongSell"].sum() * 1.5)
                }
                
                # Get the latest rating with strong recommendations weighted more heavily
                latest_row = recommendations.iloc[0]
                buy_score = latest_row["buy"] + latest_row["strongBuy"] * 1.5
                hold_score = latest_row["hold"]
                sell_score = latest_row["sell"] + latest_row["strongSell"] * 1.5
                
                if buy_score > hold_score and buy_score > sell_score:
                    latest_rating = "Strong Buy" if latest_row["strongBuy"] > latest_row["buy"] else "Buy"
                elif hold_score > buy_score and hold_score > sell_score:
                    latest_rating = "Hold"
                elif sell_score > buy_score and sell_score > hold_score:
                    latest_rating = "Strong Sell" if latest_row["strongSell"] > latest_row["sell"] else "Sell"
                
                # Get the latest target from info
                latest_target = recommendations.iloc[0].get("priceTarget", None)

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
                "latest_target": float(latest_target) if latest_target is not None else None,
                "raw_recommendations": {
                    "strongBuy": int(recommendations["strongBuy"].sum()) if recommendations is not None and not recommendations.empty else 0,
                    "buy": int(recommendations["buy"].sum()) if recommendations is not None and not recommendations.empty else 0,
                    "hold": int(recommendations["hold"].sum()) if recommendations is not None and not recommendations.empty else 0,
                    "sell": int(recommendations["sell"].sum()) if recommendations is not None and not recommendations.empty else 0,
                    "strongSell": int(recommendations["strongSell"].sum()) if recommendations is not None and not recommendations.empty else 0
                }
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
                "latest_target": None,
                "raw_recommendations": {
                    "strongBuy": 0,
                    "buy": 0,
                    "hold": 0,
                    "sell": 0,
                    "strongSell": 0
                }
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