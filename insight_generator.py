import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import openai
from dotenv import load_dotenv
import os
from sec_edgar_fetcher import fetch_filings
from stock_data_fetcher import StockDataFetcher
from cachetools import TTLCache, cached

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class InsightGenerator:
    def __init__(self):
        """Initialize the InsightGenerator with OpenAI API key and caches."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        openai.api_key = self.openai_api_key
        
        # Initialize caches with 1-hour TTL
        self.insight_cache = TTLCache(maxsize=100, ttl=3600)
        self.stock_fetcher = StockDataFetcher()

    def _prepare_financial_data(self, ticker: str, filing_date: str) -> Dict:
        """Prepare financial data for analysis by combining SEC filings and market data."""
        try:
            # Convert filing date to datetime for calculations
            filing_dt = datetime.strptime(filing_date, "%Y-%m-%d")
            
            # Calculate date ranges for historical data
            start_date = (filing_dt - timedelta(days=365)).strftime("%Y-%m-%d")
            end_date = (filing_dt + timedelta(days=30)).strftime("%Y-%m-%d")
            
            print(f"\nFetching data for {ticker} as of {filing_date}")
            print(f"Date range: {start_date} to {end_date}")
            
            # Get all filings in a single pass
            print("\nFetching all SEC filings...")
            all_filings = fetch_filings(ticker, "ALL", date_range=(start_date, filing_date))
            
            if not all_filings:
                logger.warning(f"No recent filings found for {ticker}")
                return {}

            # Group filings by type for logging
            filings_by_type = {}
            for filing in all_filings:
                filing_type = filing.get("form_type", "Unknown")
                if filing_type not in filings_by_type:
                    filings_by_type[filing_type] = []
                filings_by_type[filing_type].append(filing)

            # Print summary of filings found
            for filing_type, filings in filings_by_type.items():
                print(f"\nFound {len(filings)} {filing_type} filings:")
                for filing in filings:
                    print(f"- {filing['filing_date']}: {filing.get('financials', {}).get('Revenue', 'N/A')} Revenue")

            # Get price reaction data
            print("\nFetching price reaction data...")
            price_reaction = self.stock_fetcher.get_price_reaction(ticker, filing_date)
            if price_reaction:
                print(f"Price change: {price_reaction.get('price_change_percent', 'N/A')}%")
                print(f"Volume change: {price_reaction.get('volume_change_percent', 'N/A')}%")
            
            # Get analyst ratings
            print("\nFetching analyst ratings...")
            analyst_ratings = self.stock_fetcher.get_analyst_ratings(ticker)
            if analyst_ratings:
                print(f"Mean target: ${analyst_ratings.get('mean_target', 'N/A')}")
                print(f"Latest rating: {analyst_ratings.get('latest_rating', 'N/A')}")
                print(f"Buy/Hold/Sell: {analyst_ratings.get('buy_ratings', 0)}/{analyst_ratings.get('hold_ratings', 0)}/{analyst_ratings.get('sell_ratings', 0)}")
            
            # Get historical prices for technical analysis
            print("\nFetching historical price data...")
            price_data = self.stock_fetcher.get_historical_prices(ticker, start_date, end_date)
            if price_data:
                print(f"Found {len(price_data.get('dates', []))} price points")
            
            # Calculate technical indicators
            print("\nCalculating technical indicators...")
            technical_indicators = self.stock_fetcher.calculate_technical_indicators(price_data)
            
            # Get price and volume data at filing date
            filing_price = price_data.get("close", [])[-1] if price_data.get("close") else None
            filing_volume = price_data.get("volume", [])[-1] if price_data.get("volume") else None
            
            # Get moving averages at filing date
            sma_20 = technical_indicators.get("sma_20", [])[-1] if technical_indicators.get("sma_20") else None
            sma_50 = technical_indicators.get("sma_50", [])[-1] if technical_indicators.get("sma_50") else None
            
            # Get RSI and MACD at filing date
            rsi = technical_indicators.get("rsi", [])[-1] if technical_indicators.get("rsi") else None
            macd = technical_indicators.get("macd", [])[-1] if technical_indicators.get("macd") else None
            
            # Get Bollinger Bands at filing date
            bb = technical_indicators.get("bollinger_bands", {})
            bb_upper = bb.get("upper", [])[-1] if bb.get("upper") else None
            bb_lower = bb.get("lower", [])[-1] if bb.get("lower") else None

            if technical_indicators:
                print(f"RSI: {rsi:.2f}")
                print(f"MACD: {macd:.2f}")
                print(f"SMA 20: ${sma_20:.2f}")
                print(f"SMA 50: ${sma_50:.2f}")

            return {
                "ticker": ticker,
                "filing_date": filing_date,
                "analysis_date": filing_date,
                "recent_filings": all_filings,
                "price_reaction": price_reaction,
                "analyst_ratings": analyst_ratings,
                "technical_indicators": {
                    "filing_price": filing_price,
                    "filing_volume": filing_volume,
                    "sma_20": sma_20,
                    "sma_50": sma_50,
                    "rsi": rsi,
                    "macd": macd,
                    "bollinger_bands": {
                        "upper": bb_upper,
                        "lower": bb_lower
                    }
                },
                "price_data": price_data
            }
        except Exception as e:
            logger.error(f"Error preparing financial data: {str(e)}")
            return {}

    def _create_analysis_prompt(self, financial_data: Dict) -> str:
        """Create a structured prompt for the GPT model."""
        ticker = financial_data["ticker"]
        filing_date = financial_data["filing_date"]
        analysis_date = financial_data["analysis_date"]
        
        # Extract key metrics from all filings
        all_filings = financial_data["recent_filings"]
        print(f"\nProcessing {len(all_filings)} filings for analysis...")
        
        # Group filings by type
        filings_by_type = {}
        for filing in all_filings:
            filing_type = filing.get("form_type", "Unknown")
            if filing_type not in filings_by_type:
                filings_by_type[filing_type] = []
            filings_by_type[filing_type].append(filing)
        
        # Print filing summary
        for filing_type, filings in filings_by_type.items():
            print(f"\n{filing_type} Filings:")
            for filing in filings:
                print(f"- {filing['filing_date']}: {filing.get('financials', {}).get('Revenue', 'N/A')} Revenue")
        
        # Use the most recent filing for financial metrics
        latest_filing = all_filings[0] if all_filings else {}
        financial_metrics = latest_filing.get("financials", {})
        
        # Extract management discussion
        mda_text = financial_metrics.get("Management Discussion & Analysis summary", "")
        
        # Format technical indicators
        tech_indicators = financial_data.get("technical_indicators", {})
        
        # Format analyst ratings
        ratings = financial_data.get("analyst_ratings", {})
        
        # Format price reaction
        price_reaction = financial_data.get("price_reaction", {})
        
        prompt = f"""You are a financial analyst assistant. Analyze the following financial data for {ticker} as of {filing_date}:

Financial Metrics:
{json.dumps(financial_metrics, indent=2)}

Management Discussion:
{mda_text}

Market Data as of {filing_date}:
Price: ${tech_indicators.get('filing_price', 'N/A')}
Volume: {tech_indicators.get('filing_volume', 'N/A'):,}
RSI: {tech_indicators.get('rsi', 'N/A')}
MACD: {tech_indicators.get('macd', 'N/A')}
SMA 20: ${tech_indicators.get('sma_20', 'N/A')}
SMA 50: ${tech_indicators.get('sma_50', 'N/A')}
Bollinger Bands:
  Upper: ${tech_indicators.get('bollinger_bands', {}).get('upper', 'N/A')}
  Lower: ${tech_indicators.get('bollinger_bands', {}).get('lower', 'N/A')}

Analyst Ratings as of {filing_date}:
Mean Target: ${ratings.get('mean_target', 'N/A')}
Median Target: ${ratings.get('median_target', 'N/A')}
Latest Rating: {ratings.get('latest_rating', 'N/A')}
Latest Target: ${ratings.get('latest_target', 'N/A')}
Buy Ratings: {ratings.get('buy_ratings', 0)}
Hold Ratings: {ratings.get('hold_ratings', 0)}
Sell Ratings: {ratings.get('sell_ratings', 0)}

Price Reaction:
Pre-Filing Price: ${price_reaction.get('pre_filing_price', 'N/A')}
Post-Filing Price: ${price_reaction.get('post_filing_price', 'N/A')}
Price Change: {price_reaction.get('price_change_percent', 'N/A')}%
Volume Change: {price_reaction.get('volume_change_percent', 'N/A')}%

Recent Filings Summary:
{chr(10).join(f"- {f['form_type']} ({f['filing_date']}): {f.get('financials', {}).get('Revenue', 'N/A')} Revenue" for f in all_filings[:5])}

Provide:
1. A concise summary of key performance indicators
2. Your assessment of the company's financial health
3. A probabilistic recommendation (buy/hold/sell) with confidence levels
4. Key risk factors to consider

Format your response as JSON with the following structure:
{{
  "summary": "",
  "financial_health": "",
  "recommendation": {{
    "action": "",
    "buy_probability": 0.0,
    "hold_probability": 0.0,
    "sell_probability": 0.0,
    "confidence_level": 0.0
  }},
  "risk_factors": [],
  "technical_analysis": {{
    "trend": "",
    "momentum": "",
    "support_resistance": ""
  }},
  "market_sentiment": {{
    "analyst_consensus": "",
    "price_momentum": "",
    "volume_trend": ""
  }}
}}"""

        return prompt

    @cached(cache=TTLCache(maxsize=100, ttl=3600))
    def generate_insights(self, ticker: str, filing_date: str) -> Dict:
        """Generate investment insights for a given ticker and filing date."""
        try:
            # Prepare financial data
            financial_data = self._prepare_financial_data(ticker, filing_date)
            if not financial_data:
                return {}

            # Create analysis prompt
            prompt = self._create_analysis_prompt(financial_data)
            
            # Call OpenAI API
            try:
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a professional financial analyst. Provide detailed, data-driven analysis with clear recommendations."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                if not response or not response.choices:
                    logger.error("Empty response from OpenAI API")
                    return {}

                # Parse the response
                content = response.choices[0].message.content.strip()
                try:
                    insights = json.loads(content)
                    return {
                        "ticker": ticker,
                        "filing_date": filing_date,
                        "generated_at": datetime.now().isoformat(),
                        "insights": insights
                    }
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing OpenAI response: {e}")
                    return {}

            except openai.BadRequestError as e:
                logger.error(f"OpenAI API error: {e}")
                return {}
            except openai.AuthenticationError as e:
                logger.error(f"OpenAI API authentication error: {e}")
                return {}
            except openai.RateLimitError as e:
                logger.error(f"OpenAI API rate limit error: {e}")
                return {}
            except openai.APIError as e:
                logger.error(f"OpenAI API error: {e}")
                return {}

        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return {}

    def get_latest_insights(self, ticker: str) -> Dict:
        """Get the latest insights for a given ticker."""
        try:
            # Get the most recent filing date
            recent_filings = fetch_filings(ticker, "10-Q")
            if not recent_filings:
                logger.warning(f"No recent filings found for {ticker}")
                return {}

            latest_filing_date = recent_filings[0]["filing_date"]
            return self.generate_insights(ticker, latest_filing_date)

        except Exception as e:
            logger.error(f"Error getting latest insights: {str(e)}")
            return {} 