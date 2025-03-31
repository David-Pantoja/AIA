import yfinance as yf
import pandas as pd
import numpy as np
# Import talib conditionally so we don't fail if it's not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available. Using pandas for technical indicators.")
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from cachetools import TTLCache, cached
import logging
import requests
import time

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

            # Fetch historical data directly using string dates (like in tmp.py)
            hist = t.history(
                start=start_date,
                end=end_date,
                interval="1d",        # Daily data
                auto_adjust=True,     # Auto-adjust OHLC
                actions=True,         # Include dividends and splits
                back_adjust=True,     # Back-adjust data to mimic true historical prices
                repair=True,          # Detect and repair currency unit mixups
                rounding=False,       # Keep precision as suggested by Yahoo
                timeout=20            # Longer timeout for reliability
            )
            
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
    def get_analyst_ratings(self, ticker: str, cutoff_date: Optional[datetime] = None) -> Dict:
        """Fetch analyst ratings and price targets for a given ticker."""
        max_retries = 3
        retry_delay = 5  # Base delay in seconds
        
        for attempt in range(max_retries):
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
                    # Filter recommendations by cutoff date if provided
                    if cutoff_date:
                        # Convert cutoff_date to pandas Timestamp for comparison
                        # Make sure it's timezone naive if the recommendations index is naive
                        cutoff_ts = pd.Timestamp(cutoff_date)
                        
                        # Safely handle timezone comparison with proper type checking
                        if not recommendations.empty:
                            idx_tzinfo = getattr(recommendations.index[0], 'tzinfo', None)
                            cutoff_tzinfo = getattr(cutoff_ts, 'tzinfo', None)
                            
                            # Synchronize timezone info
                            if idx_tzinfo is None and cutoff_tzinfo is not None:
                                cutoff_ts = cutoff_ts.replace(tzinfo=None)
                            elif idx_tzinfo is not None and cutoff_tzinfo is None:
                                try:
                                    cutoff_ts = cutoff_ts.tz_localize(idx_tzinfo)
                                except:
                                    # If localization fails, make both naive
                                    cutoff_ts = cutoff_ts.replace(tzinfo=None)
                                    
                        # Use scalar comparison for recommendations filtering
                        filtered_rows = []
                        for idx, row in recommendations.iterrows():
                            try:
                                if idx <= cutoff_ts:
                                    filtered_rows.append(row)
                            except TypeError:
                                # If comparison fails, convert to string and compare dates
                                idx_date = str(idx).split()[0] if hasattr(idx, '__str__') else ''
                                cutoff_date_str = cutoff_ts.strftime('%Y-%m-%d')
                                if idx_date <= cutoff_date_str:
                                    filtered_rows.append(row)
                                    
                        if filtered_rows:
                            # Create new DataFrame with filtered rows
                            try:
                                recommendations = pd.DataFrame(filtered_rows, index=[r.name for r in filtered_rows])
                            except:
                                # Fallback if r.name is not available
                                recommendations = pd.DataFrame(filtered_rows)
                        else:
                            recommendations = pd.DataFrame()
                    
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
                
            except requests.exceptions.Timeout:
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f"Timeout fetching analyst ratings for {ticker} on attempt {attempt + 1}/{max_retries}, waiting {wait_time} seconds...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                    
            except requests.exceptions.ConnectionError as e:
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f"Connection error fetching analyst ratings for {ticker} on attempt {attempt + 1}/{max_retries}: {e}, waiting {wait_time} seconds...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                    
            except TypeError as e:
                # Handle yfinance API errors (like invalid parameters)
                logger.warning(f"yfinance API error for {ticker}: {e}")
                break  # Don't retry for API errors
                    
            except Exception as e:
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f"Error fetching analyst ratings for {ticker} on attempt {attempt + 1}/{max_retries}: {e}, waiting {wait_time} seconds...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
        
        # Return default values if all retries failed
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
            # Convert filing date to datetime (only for calculating date ranges)
            filing_dt = datetime.strptime(filing_date, "%Y-%m-%d")
            
            # Calculate date range as strings
            start_date = (filing_dt - timedelta(days=days_before)).strftime("%Y-%m-%d")
            end_date = (filing_dt + timedelta(days=days_after)).strftime("%Y-%m-%d")
            
            # Get historical prices for the period using string dates
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
        """Calculate technical indicators from price data. Falls back to pandas if talib is unavailable."""
        try:
            if not price_data or "close" not in price_data:
                return {}

            # Convert price data to numpy arrays for calculations
            close = np.array(price_data["close"], dtype=float)
            high = np.array(price_data["high"], dtype=float)
            low = np.array(price_data["low"], dtype=float)
            volume = np.array(price_data["volume"], dtype=float)

            indicators = {"ticker": price_data["ticker"]}
            
            if TALIB_AVAILABLE:
                # Use ta-lib if available
                try:
                    indicators.update({
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
                    })
                    logger.info("Successfully used TA-Lib for technical indicators")
                    return indicators
                except Exception as e:
                    logger.warning(f"TA-Lib failed, falling back to pandas implementation: {str(e)}")
                    # Fall through to pandas implementation
            
            # Use pandas implementation
            logger.info("Using pandas implementation for technical indicators")
            
            # Convert numpy arrays to pandas Series for easier calculations
            close_series = pd.Series(close)
            
            # Calculate indicators using pandas
            indicators.update({
                "sma_20": close_series.rolling(window=20).mean().tolist(),
                "sma_50": close_series.rolling(window=50).mean().tolist(),
                "sma_200": close_series.rolling(window=200).mean().tolist(),
            })
            
            # Calculate RSI
            delta = close_series.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators["rsi"] = rsi.tolist()
            
            # Calculate MACD (12,26,9)
            exp1 = close_series.ewm(span=12, adjust=False).mean()
            exp2 = close_series.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            hist = macd - signal
            
            indicators["macd"] = macd.tolist()
            indicators["macd_signal"] = signal.tolist()
            indicators["macd_hist"] = hist.tolist()
            
            # Calculate Bollinger Bands
            sma_20 = close_series.rolling(window=20).mean()
            std_20 = close_series.rolling(window=20).std()
            upper_band = sma_20 + (std_20 * 2)
            lower_band = sma_20 - (std_20 * 2)
            
            indicators["bollinger_bands"] = {
                "upper": upper_band.tolist(),
                "middle": sma_20.tolist(),
                "lower": lower_band.tolist()
            }

            return indicators
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return {} 