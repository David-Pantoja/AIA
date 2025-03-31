#single stock insight generator
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

    def _prepare_financial_data(self, ticker: str, quarters: int = 8, max_search: int = 100, 
                               use_SEC_override: Optional[bool] = None, 
                               use_yfinance_override: Optional[bool] = None) -> Dict:
        """Prepare financial data for analysis by combining SEC filings and market data.
        
        Args:
            ticker: The stock ticker symbol
            quarters: Number of quarterly/annual reports to find (10-Q + 10-K count)
            max_search: Maximum number of iterations before terminating early
            use_SEC_override: Force enable/disable SEC data fetching for this call.
            use_yfinance_override: Force enable/disable yfinance data fetching for this call.
        """
        try:
            # Determine actual data source flags based on overrides
            fetch_sec = self.use_SEC if use_SEC_override is None else use_SEC_override
            fetch_yfinance = self.use_yfinance if use_yfinance_override is None else use_yfinance_override
            
            print(f"\nFetching data for {ticker} (SEC: {fetch_sec}, yFinance: {fetch_yfinance})")
            if fetch_sec:
                print(f"Target quarters: {quarters}")
                print(f"Max search iterations: {max_search}")
            
            # Initialize data variables
            all_filings = []
            price_reaction = None
            analyst_ratings = None
            price_data = None
            technical_indicators = None

            # Get SEC filings if enabled
            if fetch_sec:
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
            if fetch_yfinance:
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

    def _create_analysis_prompt(self, financial_data: Dict, ticker: str, is_blind: bool = False) -> str:
        """Create a structured prompt for the GPT model. If is_blind is True, ask for a recommendation without specific data."""
        
        if is_blind:
            # Prompt for blind analysis (no specific data provided)
            prompt = f"""You are a financial analyst assistant. Provide a general investment recommendation for the ticker {ticker}. 
            Base your assessment ONLY on your general knowledge of this company, its sector, and overall market conditions. 
            Do NOT assume you have been given specific financial data, filings, or technical indicators for this ticker.

            Provide a probabilistic recommendation (buy/hold/sell) with confidence levels.

            Format your response as JSON with the following structure:
            {{
              "summary": "General assessment based on common knowledge.",
              "financial_health": "General assessment based on common knowledge.",
              "recommendation": {{
                "action": "",
                "buy_probability": 0.0,
                "hold_probability": 0.0,
                "sell_probability": 0.0,
                "confidence_level": 0.0 
              }},
              "risk_factors": ["General risks associated with this type of company/sector."],
              "technical_analysis": {{}},
              "market_sentiment": {{}}
            }}
            Ensure the probabilities sum to 1.0."""
            return prompt

        # Initialize prompt sections
        prompt_sections = []
        
        # Determine if SEC and yFinance data should be used based on instance settings
        use_sec_data = self.use_SEC and financial_data.get("filings")
        use_yfinance_data = self.use_yfinance and (financial_data.get("technical_indicators") or financial_data.get("price_data"))

        # Add SEC data section if enabled and available
        if use_sec_data:
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
        
        # Add market data section if yfinance is enabled and available
        if use_yfinance_data:
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
        
        # Build the complete prompt for non-blind analysis
        data_presence_message = "Based on the provided data" if prompt_sections else "Based on general knowledge (no specific data provided)"
        prompt = f"""You are a financial analyst assistant. Analyze the following financial data for {ticker}:
{chr(10).join(prompt_sections) if prompt_sections else 'No specific financial data provided.'}

{data_presence_message}, provide:
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

    def generate_insight(self, ticker: str, quarters: int = 8, max_search: int = 100,
                       use_SEC_override: Optional[bool] = None, 
                       use_yfinance_override: Optional[bool] = None) -> Dict:
        """Generate investment insight for a given ticker. Can be run in 'blind' mode using overrides.
        
        Args:
            ticker: The stock ticker symbol
            quarters: Number of quarterly/annual reports to find (10-Q + 10-K count)
            max_search: Maximum number of iterations before terminating early
            use_SEC_override: Force disable SEC data for this call (for blind mode).
            use_yfinance_override: Force disable yfinance data for this call (for blind mode).
        """
        try:
            # Determine if this is a blind run
            is_blind_run = use_SEC_override is False and use_yfinance_override is False
            
            # Prepare financial data (respecting overrides)
            financial_data = self._prepare_financial_data(
                ticker, quarters, max_search, 
                use_SEC_override=use_SEC_override, 
                use_yfinance_override=use_yfinance_override
            )
            
            # Check if data prep failed (only relevant for non-blind runs, blind runs expect empty data)
            if not financial_data and not is_blind_run:
                 logger.warning(f"Financial data preparation failed for {ticker} in non-blind mode.")
                 # We might still proceed if only one source failed, but let's return empty for now
                 # if the goal requires both sources. If partial data is acceptable, adjust this check.
                 # For now, returning {} if preparation fails non-blindly.
                 # Blind runs should proceed even with empty `financial_data`.
                 return {}

            # Create analysis prompt (passing is_blind flag)
            prompt = self._create_analysis_prompt(financial_data, ticker, is_blind=is_blind_run)

            # Get GPT-4 analysis for insights
            response = openai.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": """You are an expert financial analyst with deep knowledge of market analysis, 
                    technical indicators, and fundamental analysis. Your task is to provide comprehensive investment insights 
                    based on the provided data. Always respond with valid JSON and ensure all probabilities sum to 1.0."""},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Slightly higher temperature for more nuanced analysis
                max_tokens=2000,  # Increased for more detailed analysis
                presence_penalty=0.6,  # Encourage diverse analysis
                frequency_penalty=0.6  # Reduce repetition
            )

            # Parse the response
            try:
                content = response.choices[0].message.content.strip()
                logger.info(f"Raw GPT-4 response ({'blind' if is_blind_run else 'standard'}): {content[:200]}...")
                
                # Extract JSON from markdown code blocks if present
                if content.startswith("```"):
                    # Remove markdown code block markers
                    content = content.replace("```json", "").replace("```", "").strip()
                
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
                
                # Validate probability distribution
                probs = {
                    "buy": recommendation.get("buy_probability", 0),
                    "hold": recommendation.get("hold_probability", 0),
                    "sell": recommendation.get("sell_probability", 0)
                }
                total = sum(probs.values())
                if abs(total - 1.0) > 0.0001:
                    logger.warning(f"Probabilities do not sum to 1.0 (sum: {total})")
                    # Normalize probabilities
                    for key in probs:
                        probs[key] /= total
                    recommendation["buy_probability"] = probs["buy"]
                    recommendation["hold_probability"] = probs["hold"]
                    recommendation["sell_probability"] = probs["sell"]
                
                # If yfinance is enabled, validate against analyst ratings
                if self.use_yfinance and financial_data.get("analyst_ratings"):
                    yf_ratings = financial_data["analyst_ratings"]
                    raw_recommendations = yf_ratings.get("raw_recommendations", {})
                    total_ratings = sum(raw_recommendations.values())
                    
                    if total_ratings > 0:
                        # Calculate yfinance probabilities
                        yf_probs = {
                            "strongBuy": raw_recommendations.get("strongBuy", 0) / total_ratings,
                            "buy": raw_recommendations.get("buy", 0) / total_ratings,
                            "hold": raw_recommendations.get("hold", 0) / total_ratings,
                            "sell": raw_recommendations.get("sell", 0) / total_ratings,
                            "strongSell": raw_recommendations.get("strongSell", 0) / total_ratings
                        }
                        
                        # Add yfinance comparison to the analysis
                        analysis["yfinance_comparison"] = {
                            "yfinance_probabilities": yf_probs,
                            "model_probabilities": {
                                "strongBuy": recommendation["buy_probability"] * 0.6,
                                "buy": recommendation["buy_probability"] * 0.4,
                                "hold": recommendation["hold_probability"],
                                "sell": recommendation["sell_probability"] * 0.4,
                                "strongSell": recommendation["sell_probability"] * 0.6
                            }
                        }
                
                # Add metadata
                analysis["ticker"] = ticker
                analysis["generated_at"] = datetime.now().isoformat()
                analysis["is_blind"] = is_blind_run # Add flag to indicate blind run

                # Add filing info only if SEC data was used and available
                if not is_blind_run and self.use_SEC and financial_data.get("filings"):
                     analysis["filing_date"] = financial_data["filings"][0]["filing_date"]
                     analysis["filings"] = financial_data["filings"]

                return analysis
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GPT response as JSON ({'blind' if is_blind_run else 'standard'}): {e}")
                logger.error(f"Raw response content: {content}")
                return {}
            except Exception as e:
                logger.error(f"Error processing GPT response ({'blind' if is_blind_run else 'standard'}): {e}")
                return {}

        except Exception as e:
            logger.error(f"Error generating insight for {ticker} ({'blind' if is_blind_run else 'standard'}): {e}")
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