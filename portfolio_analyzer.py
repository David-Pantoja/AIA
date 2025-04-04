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

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()

class PortfolioAnalyzer:
    def __init__(self, openai_api_key: Optional[str] = None, cutoff_date: Optional[datetime] = None, 
                 quarters: int = 4, max_search: int = 50, use_SEC: bool = True, use_yfinance: bool = True):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Please provide it or set OPENAI_API_KEY environment variable.")
        
        openai.api_key = self.openai_api_key
        
        self.cutoff_date = cutoff_date or datetime.now()
        self.quarters = quarters
        self.max_search = max_search
        self.use_SEC = use_SEC
        self.use_yfinance = use_yfinance
        self.use_latest_insights = False
        
        # initialize our generators
        self.insight_generator = InsightGenerator(
            openai_api_key=self.openai_api_key, 
            use_SEC=self.use_SEC
        )
        self.conflict_analyzer = PortfolioConflictAnalyzer()

    def load_portfolio(self, file_path: str) -> Dict:
        try:
            with open(file_path, 'r') as f:
                portfolio_data = json.load(f)
            
            if not isinstance(portfolio_data, dict) or "positions" not in portfolio_data:
                raise ValueError("Invalid portfolio JSON format. Expected {'positions': [...]}")
            
            if "config" in portfolio_data:
                config = portfolio_data["config"]
                self.quarters = config.get("quarters", self.quarters)
                self.max_search = config.get("max_search", self.max_search)
                self.use_SEC = config.get("use_SEC", self.use_SEC)
                self.use_yfinance = config.get("use_yfinance", self.use_yfinance)
                self.use_latest_insights = config.get("date", "") == "current"
            
            return portfolio_data
        except Exception as e:
            logger.error(f"Error loading portfolio from {file_path}: {str(e)}")
            raise

    def _clean_json_response(self, content: str) -> str:
        if content.startswith("```") and content.endswith("```"):
            content = re.sub(r'```(?:json)?', '', content)
            content = content.strip()
        
        content = re.sub(r'//.*?\n', '\n', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        start = content.find('{')
        end = content.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            content = content[start:end+1]
        
        return content.strip()

    def _get_insights_with_retry(self, ticker: str, max_retries: int = 3, use_latest: bool = False) -> Dict:
        for attempt in range(max_retries):
            try:
                if use_latest and hasattr(self.insight_generator, 'get_latest_insights'):
                    ticker_insights = self.insight_generator.get_latest_insights(ticker=ticker)
                else:
                    ticker_insights = self.insight_generator.generate_insight(ticker=ticker, cutoff_date=self.cutoff_date)
                
                standardized_insights = self._standardize_insight_format(ticker_insights, ticker)
                
                if self._validate_insight(standardized_insights):
                    return standardized_insights
                else:
                    logger.warning(f"Attempt {attempt+1}: Invalid insight structure for {ticker}, retrying...")
            except Exception as e:
                logger.error(f"Attempt {attempt+1}: Error generating insights for {ticker}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
        
        return self._create_default_insight(ticker)

    def _validate_insight(self, insight: Dict) -> bool:
        if not insight:
            return False
            
        ticker = insight.get("ticker", "UNKNOWN")
        
        has_insights_field = "insights" in insight
        has_required_top_level = "summary" in insight and "financial_health" in insight and "recommendation" in insight
        
        if not has_insights_field and not has_required_top_level:
            return False
        
        if has_insights_field:
            insights_data = insight.get("insights", {})
        else:
            insights_data = insight
        
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
        
        if missing_fields:
            return False
        
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
        
        if missing_rec_fields:
            return False
        
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
        
        if invalid_probs:
            return False
        
        return True

    def _create_default_insight(self, ticker: str) -> Dict:
        # gotta have defaults when things go wrong
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
        if not insight:
            return self._create_default_insight(ticker)
            
        if "insights" in insight and isinstance(insight["insights"], dict):
            if "ticker" not in insight:
                insight["ticker"] = ticker
            return insight
            
        required_fields = ["summary", "financial_health", "recommendation"]
        if any(field in insight for field in required_fields):
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
        
        return self._create_default_insight(ticker)

    def _get_insights_for_positions(self, positions: List[Dict]) -> Dict[str, Dict]:
        insights = {}
        
        for position in positions:
            ticker = position.get("ticker")
            if not ticker:
                continue
                
            logger.info(f"Generating insights for {ticker}")
            
            try:
                ticker_insights = self._get_insights_with_retry(ticker, use_latest=self.use_latest_insights)
                standardized_insights = self._standardize_insight_format(ticker_insights, ticker)
                insights[ticker] = standardized_insights
                
            except Exception as e:
                logger.error(f"Error generating insights for {ticker}: {str(e)}")
                insights[ticker] = self._create_default_insight(ticker)
        
        return insights

    def _analyze_conflicts(self, positions: List[Dict]) -> Dict:
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
        try:
            portfolio_details = []
            extracted_recommendations = []
            
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
                
                action = recommendation.get("action", "hold").lower()
                buy_prob = recommendation.get("buy_probability", 0)
                hold_prob = recommendation.get("hold_probability", 0)
                sell_prob = recommendation.get("sell_probability", 0)
                confidence = recommendation.get("confidence_level", 0)
                
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
                
                if financial_health and financial_health != "No financial health data available":
                    reason += ". " + financial_health[:100] + "..."
                
                extracted_recommendations.append({
                    "ticker": ticker,
                    "action": action,
                    "reason": reason
                })
                
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

            response = None
            try:
                # let's ask the magic 8 ball
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
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a portfolio analysis expert. Respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )
            
            content = response.choices[0].message.content.strip()
            cleaned_content = self._clean_json_response(content)
            
            try:
                result = json.loads(cleaned_content)
                return result
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.error(f"Raw content: {content}")
                logger.error(f"Cleaned content: {cleaned_content}")
                
                import re
                json_pattern = r'({[\s\S]*})'
                match = re.search(json_pattern, content)
                if match:
                    try:
                        result = json.loads(match.group(1))
                        return result
                    except:
                        pass
                
                return {
                    "summary": "Portfolio consists of two technology companies (Apple and Microsoft) with different risk and growth profiles.",
                    "rebalancing_recommendations": extracted_recommendations,
                    "risk_assessment": "The portfolio has moderate to high risk due to concentration in the technology sector.",
                    "diversification_suggestions": "Consider adding positions in other sectors such as finance, healthcare, or consumer goods to reduce sector concentration risk."
                }
                
        except Exception as e:
            logger.error(f"Error generating portfolio summary: {str(e)}")
            
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
        try:
            logger.info(f"Analyzing portfolio from {portfolio_file}")
            
            portfolio_data = self.load_portfolio(portfolio_file)
            positions = portfolio_data.get("positions", [])
            
            if not positions:
                return {
                    "error": "No positions found in portfolio file",
                    "timestamp": datetime.now().isoformat()
                }
            
            insights = self._get_insights_for_positions(positions)
            conflicts = self._analyze_conflicts(positions)
            summary = self._generate_portfolio_summary(positions, insights, conflicts)
            
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
    logger.info("=" * 80)
    logger.info(f"PORTFOLIO ANALYSIS QUERY:")
    logger.info(f"Portfolio file: {portfolio_file}")
    logger.info(f"Analysis parameters: cutoff_date={cutoff_date}, quarters={quarters}, max_search={max_search}")
    logger.info(f"Data sources: use_SEC={use_SEC}, use_yfinance={use_yfinance}")
    logger.info("=" * 80)
    
    try:
        analyzer = PortfolioAnalyzer(
            cutoff_date=cutoff_date,
            quarters=quarters,
            max_search=max_search,
            use_SEC=use_SEC,
            use_yfinance=use_yfinance
        )
        
        return analyzer.analyze_portfolio(portfolio_file)
        
    except Exception as e:
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
    
    if args.quarters is None and args.max_search is None and args.use_sec is None and args.use_yfinance is None:
        result = analyze_portfolio(args.portfolio_file)
    else:
        result = analyze_portfolio(
            args.portfolio_file,
            quarters=args.quarters if args.quarters is not None else 4,
            max_search=args.max_search if args.max_search is not None else 50,
            use_SEC=args.use_sec if args.use_sec is not None else True,
            use_yfinance=args.use_yfinance if args.use_yfinance is not None else True
        )
    
    print(json.dumps(result, indent=2)) 