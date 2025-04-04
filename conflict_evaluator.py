import json
import logging
from typing import Dict, List, Tuple
from datetime import datetime
import openai
from dotenv import load_dotenv
import os
from portfolio_conflict_analyzer import analyze_portfolio_conflicts

# set up some basic logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# environment setup
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables")

def load_conflicted_portfolios() -> List[Dict]:
    # load our test cases
    try:
        with open('conflict_eval_input.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading conflicted portfolios: {e}")
        return []

def analyze_tickers_only(tickers: List[str]) -> Dict:
    # simple version that only uses ticker symbols
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a portfolio analysis expert. Respond ONLY with valid JSON, no additional text or explanation."},
                {"role": "user", "content": f"""Analyze these portfolio positions for potential conflicts: {', '.join(tickers)}

A conflict could be:
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
}}"""}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        content = response.choices[0].message.content.strip()
        return json.loads(content)
        
    except Exception as e:
        logger.error(f"Error in ticker-only analysis: {e}")
        return {"has_conflicts": False, "conflicts": [], "error": str(e)}

def evaluate_analyzer() -> Dict:
    # check our analyzer against known test cases
    known_conflicts = load_conflicted_portfolios()
    if not known_conflicts:
        return {"error": "Failed to load known conflicts"}
    
    results = {
        "total_cases": len(known_conflicts),
        "detailed_results": [],
        "summary": {
            "full_analyzer": {"correct": 0, "incorrect": 0},
            "ticker_only": {"correct": 0, "incorrect": 0}
        }
    }
    
    for case in known_conflicts:
        positions = case["positions"]
        known_has_conflicts = case.get("has_conflict", True)  # Default to True for backward compatibility
        
        full_analysis = analyze_portfolio_conflicts(positions)
        ticker_analysis = analyze_tickers_only(positions)
        
        full_correct = full_analysis["has_conflicts"] == known_has_conflicts
        ticker_correct = ticker_analysis["has_conflicts"] == known_has_conflicts
        
        if full_correct:
            results["summary"]["full_analyzer"]["correct"] += 1
        else:
            results["summary"]["full_analyzer"]["incorrect"] += 1
            
        if ticker_correct:
            results["summary"]["ticker_only"]["correct"] += 1
        else:
            results["summary"]["ticker_only"]["incorrect"] += 1
        
        results["detailed_results"].append({
            "positions": positions,
            "known_has_conflicts": known_has_conflicts,
            "full_analysis": {
                "has_conflicts": full_analysis["has_conflicts"],
                "correct": full_correct,
                "conflicts": full_analysis.get("conflicts", [])
            },
            "ticker_analysis": {
                "has_conflicts": ticker_analysis["has_conflicts"],
                "correct": ticker_correct,
                "conflicts": ticker_analysis.get("conflicts", [])
            }
        })
    
    # calc stats
    total = results["total_cases"]
    results["summary"]["full_analyzer"]["accuracy"] = (
        results["summary"]["full_analyzer"]["correct"] / total * 100
    )
    results["summary"]["ticker_only"]["accuracy"] = (
        results["summary"]["ticker_only"]["correct"] / total * 100
    )
    
    results["evaluated_at"] = datetime.now().isoformat()
    
    return results

if __name__ == "__main__":
    results = evaluate_analyzer()
    
    # save results to file for later
    output_file = f"conflict_eval_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2) 