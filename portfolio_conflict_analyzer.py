import yfinance as yf
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
import openai
from dotenv import load_dotenv
import os
import re

# Configure logging to only show errors
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

class PortfolioConflictAnalyzer:
    def __init__(self):
        """Initialize the PortfolioConflictAnalyzer with OpenAI API key."""
        openai.api_key = openai_api_key

    def _get_company_info(self, ticker: str) -> Dict:
        """Get company information using yfinance.
        
        Args:
            ticker: The stock ticker symbol
            
        Returns:
            Dict containing company information or empty dict if failed
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                "ticker": ticker,
                "longName": info.get('longName', 'Unknown'),
                "longBusinessSummary": info.get('longBusinessSummary', 'No summary available'),
                "legalType": info.get('legalType', 'Unknown')
            }
        except Exception as e:
            logger.error(f"Error getting company info for {ticker}: {str(e)}")
            return {}

    def _clean_json_response(self, content: str) -> str:
        """Clean and validate JSON response from the model.
        
        Args:
            content: Raw response content from the model
            
        Returns:
            Cleaned JSON string
        """
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
        
        # Remove any leading/trailing whitespace
        content = content.strip()
        
        # Handle multiple opening braces
        if content.startswith("{{"):
            content = content[1:]
        elif content.startswith("{"):
            # Count opening braces
            open_braces = len(re.findall(r'\{', content))
            close_braces = len(re.findall(r'\}', content))
            
            # If we have more opening braces than closing, remove the extra ones
            if open_braces > close_braces:
                content = content[open_braces - close_braces:]
        
        # Handle multiple closing braces
        if content.endswith("}}"):
            content = content[:-1]
        
        # Remove any trailing whitespace or newlines
        content = content.rstrip()
        
        return content

    def _create_analysis_prompt(self, positions: List[str], company_info: List[Dict]) -> str:
        """Create a prompt for the GPT model to analyze portfolio conflicts.
        
        Args:
            positions: List of ticker symbols
            company_info: List of dictionaries containing company information
            
        Returns:
            String containing the formatted prompt
        """
        # Format company information for the prompt
        company_details = "\n".join([
            f"{info['ticker']}:\n"
            f"Name: {info['longName']}\n"
            f"Type: {info['legalType']}\n"
            f"Summary: {info['longBusinessSummary']}\n"
            for info in company_info
        ])
        
        return f"""Analyze the following portfolio positions for potential conflicts:

Portfolio Positions: {', '.join(positions)}

Company Information:
{company_details}

Based on the company information provided, identify any conflicts in this portfolio. A conflict could be:
1. Sector contradictions (e.g., investing in both fossil fuels and renewable energy)
2. Directional conflicts (e.g., betting both for and against interest rates)
3. Factor exposure conflicts (e.g., mixing value and growth stocks)
4. Inflation outlook contradictions
5. Currency contradictions
6. Volatility contradictions
7. Direct hedging contradictions
8. Leveraged ETF time decay conflicts

IMPORTANT: Respond ONLY with valid JSON in the following format, with no additional text or explanation:
{{
  "has_conflicts": true/false,
  "conflicts": [
    {{
      "positions": ["ticker1", "ticker2"],
      "reason": "Brief description of the conflict type",
      "explanation": "Detailed explanation of why these positions conflict"
    }}
  ]
}}

If there are no conflicts, return ONLY:
{{
  "has_conflicts": false,
  "conflicts": []
}}"""

    def analyze_conflicts(self, positions: List[str]) -> Dict:
        """Analyze a portfolio for conflicts between positions.
        
        Args:
            positions: List of ticker symbols to analyze
            
        Returns:
            Dict containing the analysis results
        """
        try:
            # Get company information for each position
            company_info = []
            for ticker in positions:
                info = self._get_company_info(ticker)
                if info:  # Only add if we successfully got the info
                    company_info.append(info)
            
            if not company_info:
                logger.error("Failed to get company information for any positions")
                return {
                    "has_conflicts": False,
                    "conflicts": [],
                    "error": "Failed to get company information"
                }
            
            # Create and send prompt to OpenAI
            prompt = self._create_analysis_prompt(positions, company_info)
            
            response = openai.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": "You are a portfolio analysis expert. Respond ONLY with valid JSON, no additional text or explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse the response
            try:
                content = response.choices[0].message.content.strip()
                
                # Clean the response
                cleaned_content = self._clean_json_response(content)
                
                analysis = json.loads(cleaned_content)
                
                # Add metadata
                analysis["positions"] = positions
                analysis["analyzed_at"] = datetime.now().isoformat()
                
                return analysis
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GPT response as JSON: {e}")
                logger.error(f"Raw response content: {content}")
                logger.error(f"Cleaned response content: {cleaned_content}")
                return {
                    "has_conflicts": False,
                    "conflicts": [],
                    "error": "Failed to parse analysis response"
                }
                
        except Exception as e:
            logger.error(f"Error analyzing portfolio conflicts: {str(e)}")
            return {
                "has_conflicts": False,
                "conflicts": [],
                "error": str(e)
            }

def analyze_portfolio_conflicts(positions: List[str]) -> Dict:
    """Convenience function to analyze portfolio conflicts.
    
    Args:
        positions: List of ticker symbols to analyze
        
    Returns:
        Dict containing the analysis results
    """
    analyzer = PortfolioConflictAnalyzer()
    return analyzer.analyze_conflicts(positions)

if __name__ == "__main__":
    # Example usage
    positions = ["MSFT", "TLT"]
    
    if not openai_api_key:
        print("Error: OpenAI API key not found in environment variables")
        exit(1)
        
    result = analyze_portfolio_conflicts(positions)
    # Only print the final result
    print(json.dumps(result, indent=2)) 