import os
import json
from datetime import datetime
from insight_generator import InsightGenerator
from typing import Dict, List, Tuple
import pandas as pd
import yfinance as yf

# Configuration
USE_SEC = True  # Set to False to disable SEC data
USE_YFINANCE = True  # Set to False to disable yfinance data
QUARTERS = 4  # Number of quarterly/annual reports to find
MAX_SEARCH = 300  # Maximum number of search iterations

# List of tickers to benchmark
TICKERS = [
    "AAPL","MSFT",
     
 ] 
tmp = ["GOOGL", "AMZN", "META", # Tech
    "JPM", "BAC", "GS", "MS", "V",  # Financials
    "JNJ", "PFE", "UNH", "MRK", "ABBV",  # Healthcare
    "XOM", "CVX", "COP", "SLB", "EOG",  # Energy
    "PG", "KO", "WMT", "MCD", "DIS"  # Consumer
]

def validate_probability_distribution(probs: Dict[str, float]) -> bool:
    """Validate that probabilities sum to 1.0 within a small tolerance."""
    total = sum(probs.values())
    return abs(total - 1.0) < 0.0001

def calculate_mse(yf_probs: Dict[str, float], model_probs: Dict[str, float]) -> float:
    """Calculate Mean Square Error between two probability distributions."""
    if not validate_probability_distribution(yf_probs) or not validate_probability_distribution(model_probs):
        return float('inf')
    
    mse = 0.0
    for action in ["strongBuy", "buy", "hold", "sell", "strongSell"]:
        yf_prob = yf_probs.get(action, 0)
        model_prob = model_probs.get(action, 0)
        mse += (yf_prob - model_prob) ** 2
    return mse / 5  # Divide by number of actions

def process_ticker(ticker: str) -> Dict:
    """Process a single ticker and return results."""
    print(f"\nProcessing {ticker}")
    
    # Initialize the insight generator
    generator = InsightGenerator(
        os.getenv("OPENAI_API_KEY"),
        use_yfinance=USE_YFINANCE,
        use_SEC=USE_SEC
    )
    
    # Generate insight
    insight = generator.generate_insight(
        ticker,
        quarters=QUARTERS,
        max_search=MAX_SEARCH
    )
    
    if not insight:
        print(f"Failed to generate insights for {ticker}")
        return None
    
    # Extract relevant data
    result = {
        "ticker": ticker,
        "model_probabilities": {},
        "mse": None,
        "yfinance_probabilities": None,
        "quarterly_reports_found": 0
    }
    
    # Count quarterly reports if SEC data is available
    if USE_SEC and "filings" in insight:
        result["quarterly_reports_found"] = sum(
            1 for f in insight.get("filings", [])
            if f.get("form_type") in ["10-Q", "10-K"]
        )
        print(f"Found {result['quarterly_reports_found']} quarterly reports for {ticker}")
    
    # Get yfinance comparison if available
    if USE_YFINANCE and "yfinance_comparison" in insight:
        yf_comparison = insight["yfinance_comparison"]
        result["yfinance_probabilities"] = yf_comparison["yfinance_probabilities"]
        result["model_probabilities"] = yf_comparison["model_probabilities"]
        result["mse"] = calculate_mse(
            yf_comparison["yfinance_probabilities"],
            yf_comparison["model_probabilities"]
        )
    
    return result

def save_results(results: List[Dict], filename: str = None):
    """Save results to a CSV file."""
    if not results:
        print("\nNo results to save.")
        return
        
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.csv"
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Define columns in order of preference
    columns = [
        "ticker", "quarterly_reports_found",
        "mse", "yfinance_probabilities", "model_probabilities"
    ]
    
    # Only include columns that exist in the DataFrame
    available_columns = [col for col in columns if col in df.columns]
    if not available_columns:
        print("\nNo valid columns found in results.")
        return
        
    df = df[available_columns]
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total tickers processed: {len(results)}")
    if "mse" in df.columns:
        print(f"Average MSE: {df['mse'].mean():.6f}")
    if "quarterly_reports_found" in df.columns:
        print(f"Average quarterly reports found: {df['quarterly_reports_found'].mean():.2f}")
    if "mse" in df.columns:
        print(f"Success rate: {(df['mse'].notna().sum() / len(df)) * 100:.1f}%")

def main():
    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OpenAI API key not found in environment variables")
        return
    
    print("Benchmarking Configuration:")
    print(f"USE_SEC: {USE_SEC}")
    print(f"USE_YFINANCE: {USE_YFINANCE}")
    print(f"QUARTERS: {QUARTERS}")
    print(f"MAX_SEARCH: {MAX_SEARCH}")
    print(f"Number of tickers: {len(TICKERS)}")
    
    # Process each ticker
    results = []
    
    for ticker in TICKERS:
        try:
            result = process_ticker(ticker)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
    
    # Save results
    save_results(results)

if __name__ == "__main__":
    main() 