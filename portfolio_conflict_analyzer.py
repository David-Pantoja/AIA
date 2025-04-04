import yfinance as yf
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
import openai
from dotenv import load_dotenv
import os
import re

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

class PortfolioConflictAnalyzer:
    def __init__(self):
        openai.api_key = openai_api_key

    def _get_company_info(self, ticker: str) -> Dict:
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
        # fix messy json from the ai
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
        
        content = content.strip()
        
        if content.startswith("{{"):
            content = content[1:]
        elif content.startswith("{"):
            open_braces = len(re.findall(r'\{', content))
            close_braces = len(re.findall(r'\}', content))
            
            if open_braces > close_braces:
                content = content[open_braces - close_braces:]
        
        if content.endswith("}}"):
            content = content[:-1]
        
        content = content.rstrip()
        
        return content

    def _create_analysis_prompt(self, positions: List[str], company_info: List[Dict]) -> str:
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
        # where the magic happens
        try:
            company_info = []
            for ticker in positions:
                info = self._get_company_info(ticker)
                if info:
                    company_info.append(info)
            
            if not company_info:
                logger.error("Failed to get company information for any positions")
                return {
                    "has_conflicts": False,
                    "conflicts": [],
                    "error": "Failed to get company information"
                }
            
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
            
            try:
                content = response.choices[0].message.content.strip()
                
                cleaned_content = self._clean_json_response(content)
                
                analysis = json.loads(cleaned_content)
                
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
    # just a wrapper
    analyzer = PortfolioConflictAnalyzer()
    return analyzer.analyze_conflicts(positions)

if __name__ == "__main__":
    positions = ["MSFT", "TLT"]
    
    if not openai_api_key:
        exit(1)
        
    result = analyze_portfolio_conflicts(positions)