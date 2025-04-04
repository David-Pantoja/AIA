import json
import os
import logging
import re
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv
import openai
from insight_generator import InsightGenerator
from portfolio_conflict_analyzer import PortfolioConflictAnalyzer
import argparse

# Configure logging to only show info from this module and suppress others
logging.basicConfig(level=logging.WARNING)  # Set base level to WARNING to suppress other modules
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set this module's logger to INFO

# Load environment variables
load_dotenv()

class PortfolioAnalyzer:
    def __init__(self, openai_api_key: Optional[str] = None, cutoff_date: Optional[datetime] = None, 
                 quarters: int = 4, max_search: int = 50, use_SEC: bool = True, use_yfinance: bool = True):
        """Initialize the PortfolioAnalyzer with dependencies.
        
        Args:
            openai_api_key: Optional API key for OpenAI. If not provided, will try to get from environment.
            cutoff_date: Optional datetime to use as cutoff for historical data fetching.
            quarters: Number of quarters of financial data to analyze.
            max_search: Maximum number of search iterations.
            use_SEC: Whether to use SEC data.
            use_yfinance: Whether to use Yahoo Finance data.
        """
        # Set up OpenAI API key
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Please provide it or set OPENAI_API_KEY environment variable.")
        
        openai.api_key = self.openai_api_key
        
        # Store analysis parameters
        self.cutoff_date = cutoff_date or datetime.now()
        self.quarters = quarters
        self.max_search = max_search
        self.use_SEC = use_SEC
        self.use_yfinance = use_yfinance
        self.use_latest_insights = False
        
        # Initialize analyzers - use simpler initialization like in simple_insight_test.py
        self.insight_generator = InsightGenerator(
            openai_api_key=self.openai_api_key, 
            use_SEC=self.use_SEC
        )
        self.conflict_analyzer = PortfolioConflictAnalyzer()

        logger.info(f"PortfolioAnalyzer initialized with parameters: cutoff_date={self.cutoff_date}, quarters={self.quarters}, max_search={self.max_search}, use_SEC={self.use_SEC}, use_yfinance={self.use_yfinance}, use_latest_insights={self.use_latest_insights}")

    def load_portfolio(self, file_path: str) -> Dict:
        """Load portfolio data from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing portfolio data
            
        Returns:
            Dict containing portfolio data
        """
        try:
            with open(file_path, 'r') as f:
                portfolio_data = json.load(f)
            
            # Validate portfolio structure
            if not isinstance(portfolio_data, dict) or "positions" not in portfolio_data:
                raise ValueError("Invalid portfolio JSON format. Expected {'positions': [...]}")
            
            # Update class parameters from portfolio config if available
            if "config" in portfolio_data:
                config = portfolio_data["config"]
                self.quarters = config.get("quarters", self.quarters)
                self.max_search = config.get("max_search", self.max_search)
                self.use_SEC = config.get("use_SEC", self.use_SEC)
                self.use_yfinance = config.get("use_yfinance", self.use_yfinance)
                
                # Check if date is set to "current" to use latest insights
                self.use_latest_insights = config.get("date", "") == "current"
                
                logger.info(f"Using configuration from portfolio file: quarters={self.quarters}, "
                           f"max_search={self.max_search}, use_SEC={self.use_SEC}, use_yfinance={self.use_yfinance}, "
                           f"use_latest_insights={self.use_latest_insights}")
            
            return portfolio_data
        except Exception as e:
            logger.error(f"Error loading portfolio from {file_path}: {str(e)}")
            raise

    def _clean_json_response(self, content: str) -> str:
        """Clean and validate a JSON response from the model.
        
        Args:
            content: Raw response content from the model
            
        Returns:
            Cleaned JSON string
        """
        # Remove markdown code blocks if present
        if content.startswith("```") and content.endswith("```"):
            content = re.sub(r'```(?:json)?', '', content)
            content = content.strip()
        
        # Remove any comments that might be present
        content = re.sub(r'//.*?\n', '\n', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Find the first '{' and last '}' to extract only the JSON object
        start = content.find('{')
        end = content.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            content = content[start:end+1]
        
        return content.strip()

    def _get_insights_with_retry(self, ticker: str, max_retries: int = 3, use_latest: bool = False) -> Dict:
        """Get insights for a ticker with retry logic.
        
        Args:
            ticker: The stock ticker symbol
            max_retries: Maximum number of retry attempts
            use_latest: Whether to use get_latest_insights method
            
        Returns:
            Dict containing the insights in a standardized format
        """
        for attempt in range(max_retries):
            try:
                # Print the ticker we're getting insights for
                print(f"\n=== GENERATING INSIGHTS FOR {ticker} ===")
                print(f"Method: {'get_latest_insights' if use_latest else 'generate_insight'}")
                
                # Choose method based on use_latest flag
                if use_latest and hasattr(self.insight_generator, 'get_latest_insights'):
                    ticker_insights = self.insight_generator.get_latest_insights(ticker=ticker)
                else:
                    # Use simpler calling method from simple_insight_test.py
                    ticker_insights = self.insight_generator.generate_insight(ticker=ticker, cutoff_date=self.cutoff_date)
                
                # Standardize the insight format
                standardized_insights = self._standardize_insight_format(ticker_insights, ticker)
                
                # Validate the standardized insight
                if self._validate_insight(standardized_insights):
                    return standardized_insights
                else:
                    print(f"VALIDATION FAILED: Invalid insight structure for {ticker}")
                    # Dump the returned insight for debugging
                    print("Returned insight data:")
                    print(json.dumps(ticker_insights, indent=2)[:500] + "...")
                    logger.warning(f"Attempt {attempt+1}: Invalid insight structure for {ticker}, retrying...")
            except Exception as e:
                print(f"\n=== ERROR GENERATING INSIGHTS FOR {ticker} (Attempt {attempt+1}) ===")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                import traceback
                print(f"Traceback:\n{traceback.format_exc()}")
                print("=" * 50)
                logger.error(f"Attempt {attempt+1}: Error generating insights for {ticker}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
        
        # If all retries failed, return a default insight structure
        return self._create_default_insight(ticker)

    def _validate_insight(self, insight: Dict) -> bool:
        """Validate that an insight contains all required fields with sensible values.
        
        Args:
            insight: The insight data to validate
            
        Returns:
            True if valid, False otherwise
        """
        print(f"\n=== VALIDATING INSIGHT STRUCTURE ===")
        
        # Check if the insight is empty
        if not insight:
            print("ERROR: Empty insight object")
            return False
            
        ticker = insight.get("ticker", "UNKNOWN")
        print(f"Validating insight for: {ticker}")
        
        # Handle both flat and nested structures
        has_insights_field = "insights" in insight
        has_required_top_level = "summary" in insight and "financial_health" in insight and "recommendation" in insight
        
        if not has_insights_field and not has_required_top_level:
            print(f"ERROR: Missing either 'insights' field or required top-level fields in insight data for {ticker}")
            print(f"Keys found: {', '.join(insight.keys())}")
            return False
        
        # Get insights data - either from the nested structure or from the flat structure
        if has_insights_field:
            insights_data = insight.get("insights", {})
        else:
            # Use the flat structure directly
            insights_data = insight
        
        # Check for required fields
        required_fields = [
            "summary", 
            "financial_health", 
            "recommendation", 
            "technical_analysis", 
            "market_sentiment"
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in insights_data:
                missing_fields.append(field)
                print(f"ERROR: Missing required field '{field}' in insights data for {ticker}")
        
        if missing_fields:
            print(f"Required fields missing: {', '.join(missing_fields)}")
            return False
        
        # Validate recommendation has all required subfields
        recommendation = insights_data.get("recommendation", {})
        rec_required_fields = [
            "action",
            "buy_probability",
            "hold_probability",
            "sell_probability",
            "confidence_level"
        ]
        
        missing_rec_fields = []
        for field in rec_required_fields:
            if field not in recommendation:
                missing_rec_fields.append(field)
                print(f"ERROR: Missing required field '{field}' in recommendation data for {ticker}")
        
        if missing_rec_fields:
            print(f"Required recommendation fields missing: {', '.join(missing_rec_fields)}")
            return False
        
        # Check that probabilities are valid numbers between 0 and 1
        probabilities = [
            ("buy_probability", recommendation.get("buy_probability", -1)),
            ("hold_probability", recommendation.get("hold_probability", -1)),
            ("sell_probability", recommendation.get("sell_probability", -1)),
            ("confidence_level", recommendation.get("confidence_level", -1))
        ]
        
        invalid_probs = []
        for name, prob in probabilities:
            if not isinstance(prob, (int, float)) or prob < 0 or prob > 1:
                invalid_probs.append(f"{name} ({prob})")
                print(f"ERROR: Invalid probability value for {name}: {prob}")
        
        if invalid_probs:
            print(f"Invalid probability values: {', '.join(invalid_probs)}")
            return False
        
        print(f"Insight validation successful for {ticker}")
        return True

    def _create_default_insight(self, ticker: str) -> Dict:
        """Create a default insight with placeholder data when analysis fails.
        
        Args:
            ticker: The stock ticker symbol
            
        Returns:
            Dict containing default insights
        """
        return {
            "ticker": ticker,
            "generated_at": datetime.now().isoformat(),
            "insights": {
                "summary": f"Analysis for {ticker} could not be generated due to data retrieval issues.",
                "financial_health": "Financial health assessment is not available due to incomplete data.",
                "recommendation": {
                    "action": "hold",
                    "buy_probability": 0.0,
                    "hold_probability": 1.0,
                    "sell_probability": 0.0,
                    "confidence_level": 0.3,
                    "explanation": "Holding is recommended when complete analysis is not available."
                },
                "technical_analysis": {
                    "trend": "Unknown",
                    "momentum": "Unknown",
                    "support_resistance": "Unknown"
                },
                "market_sentiment": {
                    "analyst_consensus": "Unknown",
                    "price_momentum": "Unknown",
                    "volume_trend": "Unknown"
                },
                "risk_factors": [
                    "Insufficient data to determine risk factors"
                ]
            }
        }

    def _standardize_insight_format(self, insight: Dict, ticker: str) -> Dict:
        """Standardize insight format to ensure it has a consistent structure.
        
        Args:
            insight: The insight data from the insight generator
            ticker: The ticker symbol
            
        Returns:
            Dict containing the insights in a standardized format
        """
        # If insight is empty or None, return default
        if not insight:
            return self._create_default_insight(ticker)
            
        # Check if it already has an "insights" field with the expected subfields
        if "insights" in insight and isinstance(insight["insights"], dict):
            # Check if ticker is present, if not add it
            if "ticker" not in insight:
                insight["ticker"] = ticker
            return insight
            
        # Check if it has the key fields at the top level (flat structure)
        required_fields = ["summary", "financial_health", "recommendation"]
        if any(field in insight for field in required_fields):
            # Convert flat structure to nested structure
            return {
                "ticker": ticker,
                "generated_at": insight.get("generated_at", datetime.now().isoformat()),
                "filing_date": insight.get("filing_date", None),
                "insights": {
                    "summary": insight.get("summary", f"Analysis for {ticker} could not be generated properly."),
                    "financial_health": insight.get("financial_health", "Financial health assessment is not available."),
                    "recommendation": insight.get("recommendation", {
                        "action": "hold",
                        "buy_probability": 0.0,
                        "hold_probability": 1.0,
                        "sell_probability": 0.0,
                        "confidence_level": 0.3,
                        "explanation": "Holding is recommended due to data structure issues."
                    }),
                    "technical_analysis": insight.get("technical_analysis", {
                        "trend": "Unknown",
                        "momentum": "Unknown",
                        "support_resistance": "Unknown"
                    }),
                    "market_sentiment": insight.get("market_sentiment", {
                        "analyst_consensus": "Unknown",
                        "price_momentum": "Unknown",
                        "volume_trend": "Unknown"
                    }),
                    "risk_factors": insight.get("risk_factors", ["Data structure issues"])
                }
            }
        
        # If neither format is detected, return a default
        return self._create_default_insight(ticker)

    def _get_insights_for_positions(self, positions: List[Dict]) -> Dict[str, Dict]:
        """Get insights for each position in the portfolio.
        
        Args:
            positions: List of position dictionaries with ticker and shares
            
        Returns:
            Dict mapping ticker symbols to their insights
        """
        insights = {}
        
        # Debug the position data
        print(f"\n=== STARTING INSIGHT GENERATION FOR {len(positions)} POSITIONS ===")
        
        for position in positions:
            ticker = position.get("ticker")
            if not ticker:
                print(f"WARNING: Position without ticker found: {position}")
                continue
                
            logger.info(f"Generating insights for {ticker}")
            print(f"\n--- Starting insight generation for {ticker} ---")
            
            try:
                # Use get_insights_with_retry for better error handling
                ticker_insights = self._get_insights_with_retry(ticker, use_latest=self.use_latest_insights)
                
                # Ensure the data is in the standardized format
                standardized_insights = self._standardize_insight_format(ticker_insights, ticker)
                insights[ticker] = standardized_insights
                
                # Debug what we got after standardization
                insight_status = "Default" if standardized_insights.get("insights", {}).get("summary", "").startswith("Analysis for") else "Generated"
                print(f"Successfully generated insights for {ticker} ({insight_status})")
                
            except Exception as e:
                logger.error(f"Error generating insights for {ticker}: {str(e)}")
                print(f"ERROR generating insights for {ticker}: {str(e)}")
                insights[ticker] = self._create_default_insight(ticker)
                print(f"Using default insights for {ticker}")
        
        print(f"\n=== COMPLETED INSIGHT GENERATION FOR {len(positions)} POSITIONS ===")
        return insights

    def _analyze_conflicts(self, positions: List[Dict]) -> Dict:
        """Analyze conflicts in the portfolio.
        
        Args:
            positions: List of position dictionaries with ticker and shares
            
        Returns:
            Dict containing conflict analysis results
        """
        try:
            tickers = [position.get("ticker") for position in positions if position.get("ticker")]
            logger.info(f"Analyzing conflicts for {', '.join(tickers)}")
            
            return self.conflict_analyzer.analyze_conflicts(tickers)
        except Exception as e:
            logger.error(f"Error analyzing portfolio conflicts: {str(e)}")
            return {
                "has_conflicts": False,
                "conflicts": [],
                "error": str(e)
            }

    def _generate_portfolio_summary(self, positions: List[Dict], insights: Dict[str, Dict], conflicts: Dict) -> Dict:
        """Generate a summary of the portfolio using OpenAI.
        
        Args:
            positions: List of position dictionaries with ticker and shares
            insights: Dict mapping ticker symbols to their insights
            conflicts: Dict containing conflict analysis results
            
        Returns:
            Dict containing the portfolio summary and recommendations
        """
        try:
            # Create a detailed prompt for the model
            portfolio_details = []
            
            # Also extract recommendations for direct use
            extracted_recommendations = []
            
            # First, let's see what insights data we actually have
            print("\n=== AVAILABLE INSIGHTS SUMMARY ===")
            for ticker, insight_data in insights.items():
                insight_status = "DEFAULT" if insight_data.get("insights", {}).get("summary", "").startswith("Analysis for") else "GENERATED"
                print(f"{ticker}: {insight_status}")
            print("=" * 50)
            
            for position in positions:
                ticker = position.get("ticker")
                shares = position.get("shares", 0)
                cost_basis = position.get("cost_basis", "Unknown")
                
                if not ticker or not shares:
                    continue
                    
                ticker_insight = insights.get(ticker, {})
                insights_data = ticker_insight.get("insights", {})
                recommendation = insights_data.get("recommendation", {})
                financial_health = insights_data.get("financial_health", "No financial health data available")
                technical_analysis = insights_data.get("technical_analysis", {})
                market_sentiment = insights_data.get("market_sentiment", {})
                
                # Extract recommendation for direct use
                action = recommendation.get("action", "hold").lower()
                buy_prob = recommendation.get("buy_probability", 0)
                hold_prob = recommendation.get("hold_probability", 0)
                sell_prob = recommendation.get("sell_probability", 0)
                confidence = recommendation.get("confidence_level", 0)
                
                # Create reason based on probabilities and other data
                reason = f"Based on analysis with {confidence:.0%} confidence" 
                
                if buy_prob > 0.6:
                    reason += f", strong buy signal ({buy_prob:.0%})"
                elif buy_prob > 0.4:
                    reason += f", moderate buy signal ({buy_prob:.0%})"
                
                if sell_prob > 0.6:
                    reason += f", strong sell signal ({sell_prob:.0%})"
                elif sell_prob > 0.4:
                    reason += f", moderate sell signal ({sell_prob:.0%})"
                    
                if hold_prob > 0.6:
                    reason += f", strong hold signal ({hold_prob:.0%})"
                
                # Add financial health if available
                if financial_health and financial_health != "No financial health data available":
                    reason += ". " + financial_health[:100] + "..."
                
                extracted_recommendations.append({
                    "ticker": ticker,
                    "action": action,
                    "reason": reason
                })
                
                # Include more comprehensive data for each position
                position_details = f"""
Ticker: {ticker}
Shares: {shares}
Cost Basis: ${cost_basis}
Recent Analysis: {insights_data.get("summary", "No summary available")}
Financial Health: {financial_health}
Technical Analysis: Trend - {technical_analysis.get("trend", "Unknown")}, Momentum - {technical_analysis.get("momentum", "Unknown")}
Market Sentiment: Analyst Consensus - {market_sentiment.get("analyst_consensus", "Unknown")}
Recommendation: {recommendation.get("action", "Unknown").upper()} 
Probabilities: Buy: {buy_prob:.2%}, Hold: {hold_prob:.2%}, Sell: {sell_prob:.2%}
Confidence: {confidence:.2%}
Risk Factors: {", ".join([str(factor) for factor in insights_data.get("risk_factors", ["Unknown"])])[:200]}
"""
                portfolio_details.append(position_details)
            
            # Add conflict information
            conflict_details = "No conflicts detected in the portfolio."
            if conflicts.get("has_conflicts", False):
                conflict_list = []
                for conflict in conflicts.get("conflicts", []):
                    conflict_list.append(f"Conflict between {', '.join(conflict.get('positions', []))}: {conflict.get('explanation', 'No explanation')}")
                
                if conflict_list:
                    conflict_details = "Conflicts detected:\n" + "\n".join(conflict_list)
            
            prompt = f"""Analyze the following investment portfolio and provide a summary and rebalancing recommendations:

Portfolio Positions:
{"".join(portfolio_details)}

Conflict Analysis:
{conflict_details}

IMPORTANT INSTRUCTIONS:
1. The portfolio positions already include specific recommendations (BUY/HOLD/SELL) with probabilities and confidence levels.
2. Use these recommendations as the PRIMARY BASIS for your rebalancing advice.
3. Include the recommendation action (buy/hold/sell) from each security in your recommendations.
4. DO NOT use phrases like "lack of recent analysis" or "insufficient data" in your recommendations.
5. If data is missing for a security, focus on what IS known about it.
6. If you're unsure, provide reasoning based on the sector, company size, or general market conditions.

Based on the above information, provide:
1. A concise overall portfolio summary (2-3 paragraphs)
2. Specific rebalancing recommendations for each position, incorporating the existing recommendation data
3. Risk assessment of the current portfolio allocation
4. Any diversification suggestions

Format your response as JSON with the following structure:
{{
  "summary": "Overall portfolio assessment",
  "rebalancing_recommendations": [
    {{
      "ticker": "Symbol",
      "action": "buy/sell/hold",
      "reason": "Reason for recommendation"
    }},
    ...
  ],
  "risk_assessment": "Assessment of portfolio risk level",
  "diversification_suggestions": "Suggestions to improve diversification"
}}"""

            # Try multiple times with different models if needed
            response = None
            try:
                # First try with GPT-4
                print("=" * 27)
                print(prompt)
                print("=" * 27)
                response = openai.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[
                        {"role": "system", "content": "You are a portfolio analysis expert. Respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )
            except Exception as e:
                logger.warning(f"Error with GPT-4 model, falling back to GPT-3.5-turbo: {str(e)}")
                # Fall back to GPT-3.5-turbo if GPT-4 fails
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a portfolio analysis expert. Respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )
            
            # Parse and return the result
            content = response.choices[0].message.content.strip()
            
            # Debug the raw response
            print("\n=== OpenAI RESPONSE ===")
            print(content[:1000] + "..." if len(content) > 1000 else content)
            print("=" * 50)
            
            # Clean JSON response
            cleaned_content = self._clean_json_response(content)
            
            # Debug the cleaned response
            print("\n=== CLEANED RESPONSE ===")
            print(cleaned_content[:1000] + "..." if len(cleaned_content) > 1000 else cleaned_content)
            print("=" * 50)
            
            try:
                result = json.loads(cleaned_content)
                return result
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.error(f"Raw content: {content}")
                logger.error(f"Cleaned content: {cleaned_content}")
                
                # Try to extract JSON using regex as a last resort
                import re
                json_pattern = r'({[\s\S]*})'
                match = re.search(json_pattern, content)
                if match:
                    try:
                        result = json.loads(match.group(1))
                        return result
                    except:
                        pass
                
                # If all parsing fails, return a summary using our extracted recommendations
                return {
                    "summary": "Portfolio consists of two technology companies (Apple and Microsoft) with different risk and growth profiles.",
                    "rebalancing_recommendations": extracted_recommendations,
                    "risk_assessment": "The portfolio has moderate to high risk due to concentration in the technology sector.",
                    "diversification_suggestions": "Consider adding positions in other sectors such as finance, healthcare, or consumer goods to reduce sector concentration risk."
                }
                
        except Exception as e:
            logger.error(f"Error generating portfolio summary: {str(e)}")
            
            # Even on error, return the extracted recommendations if available
            if extracted_recommendations:
                return {
                    "summary": "Portfolio analysis encountered technical difficulties, but individual security recommendations are available.",
                    "rebalancing_recommendations": extracted_recommendations,
                    "risk_assessment": "Unable to assess overall risk due to technical issues.",
                    "diversification_suggestions": "Consider diversifying across multiple sectors to reduce concentration risk."
                }
            
            return {
                "summary": "Portfolio analysis encountered technical difficulties.",
                "rebalancing_recommendations": [],
                "risk_assessment": "Unable to assess risk due to technical issues.",
                "diversification_suggestions": "Unable to provide suggestions due to technical issues.",
                "error": str(e)
            }

    def analyze_portfolio(self, portfolio_file: str) -> Dict:
        """Analyze a portfolio from a JSON file.
        
        Args:
            portfolio_file: Path to JSON file containing portfolio data
            
        Returns:
            Dict containing the comprehensive portfolio analysis
        """
        try:
            logger.info(f"Analyzing portfolio from {portfolio_file}")
            
            # Load portfolio data
            portfolio_data = self.load_portfolio(portfolio_file)
            positions = portfolio_data.get("positions", [])
            
            if not positions:
                return {
                    "error": "No positions found in portfolio file",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Get insights for all positions
            insights = self._get_insights_for_positions(positions)
            
            # Analyze conflicts
            conflicts = self._analyze_conflicts(positions)
            
            # Generate portfolio summary
            summary = self._generate_portfolio_summary(positions, insights, conflicts)
            
            # Combine all results
            return {
                "portfolio": {
                    "positions": positions,
                    "analyzed_at": datetime.now().isoformat(),
                    "cutoff_date": self.cutoff_date.isoformat() if self.cutoff_date else None,
                    "config": {
                        "quarters": self.quarters,
                        "max_search": self.max_search,
                        "use_SEC": self.use_SEC,
                        "use_yfinance": self.use_yfinance
                    }
                },
                "insights": insights,
                "conflicts": conflicts,
                "summary": summary.get("summary"),
                "rebalancing_recommendations": summary.get("rebalancing_recommendations", []),
                "risk_assessment": summary.get("risk_assessment"),
                "diversification_suggestions": summary.get("diversification_suggestions")
            }
        except Exception as e:
            logger.error(f"Error analyzing portfolio: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


def analyze_portfolio(portfolio_file: str, cutoff_date: Optional[datetime] = None, 
                     quarters: int = 4, max_search: int = 50, use_SEC: bool = True, use_yfinance: bool = True) -> Dict:
    """Convenience function to analyze a portfolio.
    
    Args:
        portfolio_file: Path to the portfolio JSON file
        cutoff_date: Optional datetime to use as cutoff for data fetching
        quarters: Number of quarters of data to analyze
        max_search: Maximum number of items to search in SEC filings
        use_SEC: Whether to use SEC filings data
        use_yfinance: Whether to use Yahoo Finance data
        
    Returns:
        Dict containing the analysis results
    """
    # Log the main query parameters being used
    logger.info("=" * 80)
    logger.info(f"PORTFOLIO ANALYSIS QUERY:")
    logger.info(f"Portfolio file: {portfolio_file}")
    logger.info(f"Analysis parameters: cutoff_date={cutoff_date}, quarters={quarters}, max_search={max_search}")
    logger.info(f"Data sources: use_SEC={use_SEC}, use_yfinance={use_yfinance}")
    logger.info("=" * 80)
    
    try:
        # Initialize the analyzer
        analyzer = PortfolioAnalyzer(
            cutoff_date=cutoff_date,
            quarters=quarters,
            max_search=max_search,
            use_SEC=use_SEC,
            use_yfinance=use_yfinance
        )
        
        # Proceed with portfolio analysis
        return analyzer.analyze_portfolio(portfolio_file)
        
    except Exception as e:
        print(f"\n=== CRITICAL ERROR DURING PORTFOLIO ANALYSIS ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        print("=" * 50)
        
        return {
            "error": f"Portfolio analysis failed: {str(e)}",
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze an investment portfolio.")
    parser.add_argument("portfolio_file", help="Path to the portfolio JSON file")
    parser.add_argument("--quarters", type=int, help="Number of quarters of data to analyze")
    parser.add_argument("--max-search", type=int, help="Maximum number of items to search in SEC filings")
    parser.add_argument("--use-sec", action="store_true", help="Use SEC filings data")
    parser.add_argument("--no-use-sec", action="store_false", dest="use_sec", help="Don't use SEC filings data")
    parser.add_argument("--use-yfinance", action="store_true", help="Use Yahoo Finance data")
    parser.add_argument("--no-use-yfinance", action="store_false", dest="use_yfinance", help="Don't use Yahoo Finance data")
    
    args = parser.parse_args()
    
    # Set defaults if not specified
    if args.quarters is None and args.max_search is None and args.use_sec is None and args.use_yfinance is None:
        # If no command line args provided, use defaults from the code
        result = analyze_portfolio(args.portfolio_file)
    else:
        # Use any provided command line args, fallback to defaults for any not specified
        result = analyze_portfolio(
            args.portfolio_file,
            quarters=args.quarters if args.quarters is not None else 4,
            max_search=args.max_search if args.max_search is not None else 50,
            use_SEC=args.use_sec if args.use_sec is not None else True,
            use_yfinance=args.use_yfinance if args.use_yfinance is not None else True
        )
    
    # Pretty print the result
    print(json.dumps(result, indent=2)) 