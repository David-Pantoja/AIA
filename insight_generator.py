import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import openai
from dotenv import load_dotenv
import os
from sec_edgar_fetcher import fetch_filings
from sec_edgar_fetcher.fetcher import SECFetcher
from stock_data_fetcher.fetcher import StockDataFetcher
from cachetools import TTLCache, cached

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class InsightGenerator:
    def __init__(self, openai_api_key: str, use_yfinance: bool = True, use_SEC: bool = True):
        """Initialize the InsightGenerator with OpenAI API key and data source flags."""
        self.openai_api_key = openai_api_key
        self.use_yfinance = use_yfinance
        self.use_SEC = use_SEC
        self.sec_fetcher = SECFetcher()
        self.stock_fetcher = StockDataFetcher()
        openai.api_key = openai_api_key

    def _prepare_financial_data(self, ticker: str, filing_date: str, quarters: int = 8, max_search: int = 100) -> Dict:
        """Prepare financial data for analysis by combining SEC filings and market data.
        
        Args:
            ticker: The stock ticker symbol
            filing_date: The filing date to analyze
            quarters: Number of quarterly/annual reports to find (10-Q + 10-K count)
            max_search: Maximum number of iterations before terminating early
        """
        try:
            # Convert filing date to datetime for calculations
            filing_dt = datetime.strptime(filing_date, "%Y-%m-%d")
            
            # Calculate date ranges for historical data
            start_date = (filing_dt - timedelta(days=365)).strftime("%Y-%m-%d")
            end_date = (filing_dt + timedelta(days=30)).strftime("%Y-%m-%d")
            
            print(f"\nFetching data for {ticker} as of {filing_date}")
            print(f"Date range: {start_date} to {end_date}")
            print(f"Target quarters: {quarters}")
            print(f"Max search iterations: {max_search}")
            
            # Initialize data variables
            all_filings = []
            price_reaction = None
            analyst_ratings = None
            price_data = None
            technical_indicators = None

            # Get SEC filings if enabled
            if self.use_SEC:
                print("\nFetching all SEC filings...")
                all_filings = fetch_filings(
                    ticker, 
                    "ALL", 
                    date_range=(start_date, filing_date),
                    quarters=quarters,
                    max_search=max_search
                )
                
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

            # Get market data if yfinance is enabled
            if self.use_yfinance:
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
                
                if technical_indicators:
                    print(f"RSI: {technical_indicators.get('rsi', [])[-1]:.2f}")
                    print(f"MACD: {technical_indicators.get('macd', [])[-1]:.2f}")
                    print(f"SMA 20: ${technical_indicators.get('sma_20', [])[-1]:.2f}")
                    print(f"SMA 50: ${technical_indicators.get('sma_50', [])[-1]:.2f}")

            return {
                "ticker": ticker,
                "filing_date": filing_date,
                "analysis_date": filing_date,
                "recent_filings": all_filings,
                "price_reaction": price_reaction,
                "analyst_ratings": analyst_ratings,
                "technical_indicators": technical_indicators,
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
        
        # Initialize prompt sections
        prompt_sections = []
        
        # Add SEC data section if enabled
        if self.use_SEC:
            all_filings = financial_data["recent_filings"]
            if all_filings:
                # Use the most recent filing for financial metrics
                latest_filing = all_filings[0]
                financial_metrics = latest_filing.get("financials", {})
                mda_text = financial_metrics.get("Management Discussion & Analysis summary", "")
                
                prompt_sections.append(f"""Financial Metrics:
{json.dumps(financial_metrics, indent=2)}

Management Discussion:
{mda_text}

Recent Filings Summary:
{chr(10).join(f"- {f['form_type']} ({f['filing_date']}): {f.get('financials', {}).get('Revenue', 'N/A')} Revenue" for f in all_filings[:5])}""")
        
        # Add market data section if yfinance is enabled
        if self.use_yfinance:
            tech_indicators = financial_data.get("technical_indicators", {})
            price_reaction = financial_data.get("price_reaction", {})
            
            prompt_sections.append(f"""Market Data as of {filing_date}:
Price: ${tech_indicators.get('filing_price', 'N/A')}
Volume: {tech_indicators.get('filing_volume', 'N/A')}
RSI: {tech_indicators.get('rsi', 'N/A')}
MACD: {tech_indicators.get('macd', 'N/A')}
SMA 20: ${tech_indicators.get('sma_20', 'N/A')}
SMA 50: ${tech_indicators.get('sma_50', 'N/A')}
Bollinger Bands:
  Upper: ${tech_indicators.get('bollinger_bands', {}).get('upper', 'N/A')}
  Lower: ${tech_indicators.get('bollinger_bands', {}).get('lower', 'N/A')}

Price Reaction:
Pre-Filing Price: ${price_reaction.get('pre_filing_price', 'N/A')}
Post-Filing Price: ${price_reaction.get('post_filing_price', 'N/A')}
Price Change: {price_reaction.get('price_change_percent', 'N/A')}%
Volume Change: {price_reaction.get('volume_change_percent', 'N/A')}%""")
        
        # Build the complete prompt
        prompt = f"""You are a financial analyst assistant. Analyze the following financial data for {ticker} as of {filing_date}:

{chr(10).join(prompt_sections)}

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

    def generate_insight(self, ticker: str, filing_date: str, quarters: int = 8, max_search: int = 100) -> Dict:
        """Generate investment insight for a given ticker and filing date.
        
        Args:
            ticker: The stock ticker symbol
            filing_date: The filing date to analyze
            quarters: Number of quarterly/annual reports to find (10-Q + 10-K count)
            max_search: Maximum number of iterations before terminating early
        """
        try:
            # Prepare financial data
            financial_data = self._prepare_financial_data(ticker, filing_date, quarters, max_search)
            if not financial_data:
                return {}

            # Create analysis prompt (now only includes SEC data)
            prompt = self._create_analysis_prompt(financial_data)

            # Get GPT analysis
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst assistant. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )

            # Parse the response
            try:
                analysis = json.loads(response.choices[0].message.content)
                
                # If yfinance is enabled, validate against analyst ratings
                if self.use_yfinance and financial_data.get("analyst_ratings"):
                    yf_ratings = financial_data["analyst_ratings"]
                    raw_recommendations = yf_ratings.get("raw_recommendations", {})
                    total_ratings = sum(raw_recommendations.values())
                    
                    if total_ratings > 0:
                        # Calculate probabilities for all 5 ratings
                        yf_probs = {
                            "strongBuy": raw_recommendations.get("strongBuy", 0) / total_ratings,
                            "buy": raw_recommendations.get("buy", 0) / total_ratings,
                            "hold": raw_recommendations.get("hold", 0) / total_ratings,
                            "sell": raw_recommendations.get("sell", 0) / total_ratings,
                            "strongSell": raw_recommendations.get("strongSell", 0) / total_ratings
                        }
                        
                        # Map model probabilities to the 5 ratings
                        model_probs = {
                            "strongBuy": analysis["recommendation"]["buy_probability"] * 0.3,  # 30% of buy probability
                            "buy": analysis["recommendation"]["buy_probability"] * 0.7,      # 70% of buy probability
                            "hold": analysis["recommendation"]["hold_probability"],
                            "sell": analysis["recommendation"]["sell_probability"] * 0.7,    # 70% of sell probability
                            "strongSell": analysis["recommendation"]["sell_probability"] * 0.3  # 30% of sell probability
                        }
                        
                        # Add yfinance comparison to the analysis
                        analysis["yfinance_comparison"] = {
                            "yfinance_probabilities": yf_probs,
                            "model_probabilities": model_probs,
                            "difference": {
                                k: model_probs[k] - yf_probs[k] for k in yf_probs.keys()
                            }
                        }
                
                return analysis
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing GPT response: {str(e)}")
                return {}

        except Exception as e:
            logger.error(f"Error generating insight: {str(e)}")
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
            return self.generate_insight(ticker, latest_filing_date)

        except Exception as e:
            logger.error(f"Error getting latest insights: {str(e)}")
            return {} 