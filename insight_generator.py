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

    def _prepare_financial_data(self, ticker: str, quarters: int = 8, max_search: int = 100) -> Dict:
        """Prepare financial data for analysis by combining SEC filings and market data.
        
        Args:
            ticker: The stock ticker symbol
            quarters: Number of quarterly/annual reports to find (10-Q + 10-K count)
            max_search: Maximum number of iterations before terminating early
        """
        try:
            print(f"\nFetching data for {ticker}")
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

            # Get market data if enabled
            if self.use_yfinance:
                print("\nFetching market data...")
                fetcher = StockDataFetcher()
                
                # Get analyst ratings
                analyst_ratings = fetcher.get_analyst_ratings(ticker)
                if analyst_ratings:
                    print(f"Latest rating: {analyst_ratings.get('latest_rating', 'N/A')}")
                    print(f"Price target: {analyst_ratings.get('latest_target', 'N/A')}")
                
                # Get historical prices for technical indicators
                price_data = fetcher.get_historical_prices(
                    ticker,
                    start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                    end_date=datetime.now().strftime("%Y-%m-%d")
                )
                
                if price_data:
                    technical_indicators = fetcher.calculate_technical_indicators(price_data)
                    print("Calculated technical indicators")

            return {
                "filings": all_filings,
                "price_reaction": price_reaction,
                "analyst_ratings": analyst_ratings,
                "price_data": price_data,
                "technical_indicators": technical_indicators
            }
            
        except Exception as e:
            logger.error(f"Error preparing financial data: {str(e)}")
            return {}

    def _create_analysis_prompt(self, financial_data: Dict, ticker: str) -> str:
        """Create a structured prompt for the GPT model."""
        # Initialize prompt sections
        prompt_sections = []
        
        # Add SEC data section if enabled
        if self.use_SEC and "filings" in financial_data:
            all_filings = financial_data["filings"]
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
            price_data = financial_data.get("price_data", {})
            
            if price_data and "close" in price_data:
                latest_price = price_data["close"][-1] if price_data["close"] else "N/A"
                latest_volume = price_data["volume"][-1] if price_data["volume"] else "N/A"
            else:
                latest_price = "N/A"
                latest_volume = "N/A"
            
            prompt_sections.append(f"""Market Data:
Price: ${latest_price}
Volume: {latest_volume}
RSI: {tech_indicators.get('rsi', 'N/A')}
MACD: {tech_indicators.get('macd', 'N/A')}
SMA 20: ${tech_indicators.get('sma_20', 'N/A')}
SMA 50: ${tech_indicators.get('sma_50', 'N/A')}
Bollinger Bands:
  Upper: ${tech_indicators.get('bollinger_bands', {}).get('upper', 'N/A')}
  Lower: ${tech_indicators.get('bollinger_bands', {}).get('lower', 'N/A')}""")
        
        # Build the complete prompt
        prompt = f"""You are a financial analyst assistant. Analyze the following financial data for {ticker}:

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

    def generate_insight(self, ticker: str, quarters: int = 8, max_search: int = 100) -> Dict:
        """Generate investment insight for a given ticker.
        
        Args:
            ticker: The stock ticker symbol
            quarters: Number of quarterly/annual reports to find (10-Q + 10-K count)
            max_search: Maximum number of iterations before terminating early
        """
        try:
            # Prepare financial data
            financial_data = self._prepare_financial_data(ticker, quarters, max_search)
            if not financial_data:
                return {}

            # Create analysis prompt
            prompt = self._create_analysis_prompt(financial_data, ticker)

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
                content = response.choices[0].message.content.strip()
                logger.info(f"Raw GPT response: {content[:200]}...")  # Log first 200 chars
                
                # Try to parse the response as JSON
                analysis = json.loads(content)
                
                # Validate required fields
                required_fields = ["summary", "financial_health", "recommendation", "risk_factors", "technical_analysis", "market_sentiment"]
                missing_fields = [field for field in required_fields if field not in analysis]
                if missing_fields:
                    logger.error(f"Missing required fields in GPT response: {missing_fields}")
                    return {}
                
                # Validate recommendation structure
                recommendation = analysis.get("recommendation", {})
                required_rec_fields = ["action", "buy_probability", "hold_probability", "sell_probability", "confidence_level"]
                missing_rec_fields = [field for field in required_rec_fields if field not in recommendation]
                if missing_rec_fields:
                    logger.error(f"Missing required fields in recommendation: {missing_rec_fields}")
                    return {}
                
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
                logger.error(f"Raw response content: {content}")
                return {}
            except Exception as e:
                logger.error(f"Unexpected error processing GPT response: {str(e)}")
                return {}

        except Exception as e:
            logger.error(f"Error generating insight: {str(e)}")
            return {}

    def get_latest_insights(self, ticker: str) -> Dict:
        """Get the latest insights for a given ticker.
        
        Args:
            ticker: The stock ticker symbol
            
        Returns:
            Dict containing the analysis results
        """
        try:
            # Generate insights with default parameters
            return self.generate_insight(ticker)
            
        except Exception as e:
            logger.error(f"Error getting latest insights: {str(e)}")
            return {} 