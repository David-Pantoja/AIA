import json
import logging
from typing import Dict, List, Tuple
from datetime import datetime
import openai
from dotenv import load_dotenv
import os
from portfolio_conflict_analyzer import analyze_portfolio_conflicts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables")

def load_conflicted_portfolios() -> List[Dict]:
    """Load the known conflicted portfolios from JSON file."""
    try:
        with open('conflicted_portfolios.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading conflicted portfolios: {e}")
        return []

def analyze_tickers_only(tickers: List[str]) -> Dict:
    """Analyze portfolio conflicts using only ticker symbols.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        Dict containing the analysis results
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
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
    """Evaluate the portfolio conflict analyzer against known conflicts."""
    # Load known conflicts
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
    
    # Evaluate each case
    for case in known_conflicts:
        positions = case["positions"]
        known_has_conflicts = case.get("has_conflict", True)  # Default to True for backward compatibility
        
        # Get results from both methods
        full_analysis = analyze_portfolio_conflicts(positions)
        ticker_analysis = analyze_tickers_only(positions)
        
        # Compare results
        full_correct = full_analysis["has_conflicts"] == known_has_conflicts
        ticker_correct = ticker_analysis["has_conflicts"] == known_has_conflicts
        
        # Update summary counts
        if full_correct:
            results["summary"]["full_analyzer"]["correct"] += 1
        else:
            results["summary"]["full_analyzer"]["incorrect"] += 1
            
        if ticker_correct:
            results["summary"]["ticker_only"]["correct"] += 1
        else:
            results["summary"]["ticker_only"]["incorrect"] += 1
        
        # Store detailed results
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
    
    # Calculate accuracy percentages
    total = results["total_cases"]
    results["summary"]["full_analyzer"]["accuracy"] = (
        results["summary"]["full_analyzer"]["correct"] / total * 100
    )
    results["summary"]["ticker_only"]["accuracy"] = (
        results["summary"]["ticker_only"]["correct"] / total * 100
    )
    
    # Add timestamp
    results["evaluated_at"] = datetime.now().isoformat()
    
    return results

if __name__ == "__main__":
    # Run evaluation
    results = evaluate_analyzer()
    
    # Print summary
    print("\nEvaluation Summary:")
    print("-" * 50)
    print(f"Total cases evaluated: {results['total_cases']}")
    print("\nFull Analyzer Results:")
    print(f"Correct: {results['summary']['full_analyzer']['correct']}")
    print(f"Incorrect: {results['summary']['full_analyzer']['incorrect']}")
    print(f"Accuracy: {results['summary']['full_analyzer']['accuracy']:.2f}%")
    print("\nTicker-Only Analysis Results:")
    print(f"Correct: {results['summary']['ticker_only']['correct']}")
    print(f"Incorrect: {results['summary']['ticker_only']['incorrect']}")
    print(f"Accuracy: {results['summary']['ticker_only']['accuracy']:.2f}%")
    
    # Print detailed results for incorrect cases
    print("\nDetailed Results for Incorrect Cases:")
    print("-" * 50)
    for case in results["detailed_results"]:
        if not case["full_analysis"]["correct"] or not case["ticker_analysis"]["correct"]:
            print(f"\nPositions: {', '.join(case['positions'])}")
            print(f"Known has conflicts: {case['known_has_conflicts']}")
            print("Full Analysis:")
            print(f"  Has conflicts: {case['full_analysis']['has_conflicts']}")
            print(f"  Correct: {case['full_analysis']['correct']}")
            print("Ticker Analysis:")
            print(f"  Has conflicts: {case['ticker_analysis']['has_conflicts']}")
            print(f"  Correct: {case['ticker_analysis']['correct']}")
    
    # Save full results to file
    output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {output_file}") 