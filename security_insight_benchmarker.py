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
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",  # Tech - Large Cap
    "NVDA", "AMD", "INTC", "TSM", "MU",  # Semiconductors
    "CRM", "ADBE", "ORCL", "IBM", "SAP",  # Enterprise Software
    "CSCO", "AVGO", "QCOM", "TXN", "NXPI",  # Networking/Hardware
    "PYPL", "SQ", "ADYEY", "V", "MA",  # Fintech
    
    "JPM", "BAC", "GS", "MS", "WFC",  # Banking
    "BLK", "BX", "KKR", "C", "AXP",  # Financial Services
    "PGR", "ALL", "TRV", "CB", "MET",  # Insurance
    
    "JNJ", "PFE", "UNH", "MRK", "ABBV",  # Healthcare/Pharma
    "MDT", "TMO", "DHR", "ABT", "ISRG",  # Medical Devices/Equipment
    
    "XOM", "CVX", "COP", "SLB", "EOG",  # Oil & Gas
    "NEE", "DUK", "SO", "AEP", "PCG",  # Utilities
    
    "PG", "KO", "WMT", "MCD", "DIS",  # Consumer Staples/Entertainment
    "NKE", "SBUX", "LULU", "TGT", "HD",  # Retail
    "PEP", "KHC", "GIS", "K", "CAG",  # Food & Beverage
    
    "TSLA", "GM", "F", "TM", "VWAGY",  # Auto
    "BA", "LMT", "RTX", "GE", "HON",  # Aerospace/Industrial
    
    "NFLX", "CMCSA", "CHTR", "ROKU", "SPOT",  # Media/Streaming
    "T", "VZ", "TMUS", "LBRDK", "DISH",  # Telecom
    
    "AMGN", "GILD", "BIIB", "MRNA", "REGN",  # Biotech
    "ABNB", "UBER", "LYFT", "DASH", "BKNG",  # Travel/Mobility
    
    "TWLO", "NET", "ZS", "OKTA", "CRWD",  # Cloud Security
    "NOW", "TEAM", "ZM", "DOCU", "WDAY"   # SaaS
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
    """Process a single ticker and return results including standard and blind analysis."""
    print(f"\nProcessing {ticker}")
    
    # Initialize the insight generator
    generator = InsightGenerator(
        os.getenv("OPENAI_API_KEY"),
        use_yfinance=USE_YFINANCE,
        use_SEC=USE_SEC
    )
    
    # --- Standard Insight Generation ---
    print("\nGenerating standard insight...")
    insight = generator.generate_insight(
        ticker,
        quarters=QUARTERS,
        max_search=MAX_SEARCH
    )
    
    if not insight:
        print(f"Failed to generate standard insights for {ticker}")
        return None # Return None if standard insight fails
    
    # --- Blind Insight Generation ---
    print("\nGenerating blind insight...")
    blind_insight = generator.generate_insight(
        ticker,
        quarters=0,  # No quarters needed for blind
        max_search=0, # No search needed for blind
        use_SEC_override=False,
        use_yfinance_override=False
    )

    if not blind_insight:
        print(f"Warning: Failed to generate blind insights for {ticker}. Proceeding without blind data.")
        blind_model_probs = None
        blind_mse = None
    else:
        # Extract blind probabilities
        blind_recommendation = blind_insight.get("recommendation", {})
        blind_model_probs = {
            "strongBuy": blind_recommendation.get("buy_probability", 0) * 0.6, # Apportion based on confidence maybe?
            "buy": blind_recommendation.get("buy_probability", 0) * 0.4,
            "hold": blind_recommendation.get("hold_probability", 0),
            "sell": blind_recommendation.get("sell_probability", 0) * 0.4,
            "strongSell": blind_recommendation.get("sell_probability", 0) * 0.6
        }
        # Normalize blind probabilities after apportionment
        total_blind_prob = sum(blind_model_probs.values())
        if total_blind_prob > 0:
             blind_model_probs = {k: v / total_blind_prob for k, v in blind_model_probs.items()}
        else:
             blind_model_probs = {k: 0.0 for k in blind_model_probs}


    # Initialize result dict
    result = {
        "ticker": ticker,
        "model_probabilities": None,
        "mse": None,
        "yfinance_probabilities": None,
        "quarterly_reports_found": 0,
        "blind_model_probabilities": blind_model_probs, # Initialize with potentially None
        "blind_mse": None
    }
    
    # --- Process Standard Insight Results ---
    # Count quarterly reports if SEC data was used in standard insight
    if USE_SEC and "filings" in insight and insight["filings"]:
        result["quarterly_reports_found"] = sum(
            1 for f in insight.get("filings", [])
            if f.get("form_type") in ["10-Q", "10-K"]
        )
        print(f"Found {result['quarterly_reports_found']} quarterly reports for {ticker}")
    
    # Get yfinance comparison if available from standard insight
    if USE_YFINANCE and "yfinance_comparison" in insight:
        yf_comparison = insight["yfinance_comparison"]
        yf_probs = yf_comparison.get("yfinance_probabilities")
        model_probs = yf_comparison.get("model_probabilities")

        if yf_probs:
             result["yfinance_probabilities"] = yf_probs
             # Calculate standard MSE
             if model_probs:
                 result["model_probabilities"] = model_probs
                 result["mse"] = calculate_mse(yf_probs, model_probs)
             
             # Calculate blind MSE if blind probs exist
             if blind_model_probs:
                 result["blind_mse"] = calculate_mse(yf_probs, blind_model_probs)
        else:
            print(f"Warning: yfinance probabilities missing for {ticker}")
    elif USE_YFINANCE:
        print(f"Warning: yfinance_comparison block missing in standard insight for {ticker}")

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
    
    # Define columns in order of preference, including new blind columns
    columns = [
        "ticker", "quarterly_reports_found",
        "mse", "yfinance_probabilities", "model_probabilities",
        "blind_mse", "blind_model_probabilities"
    ]
    
    # Only include columns that exist in the DataFrame
    available_columns = [col for col in columns if col in df.columns]
    if not available_columns:
        print("\nNo valid columns found in results.")
        return
        
    df = df[available_columns]
    
    # Convert probability dicts to strings for CSV
    for col in ["yfinance_probabilities", "model_probabilities", "blind_model_probabilities"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
            
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")
    
    # Print summary statistics, including blind MSE
    print("\nSummary Statistics:")
    print(f"Total tickers processed: {len(results)}")
    if "mse" in df.columns:
        print(f"Average MSE (Standard): {df['mse'].mean():.6f}")
    if "blind_mse" in df.columns:
        print(f"Average MSE (Blind): {df['blind_mse'].mean():.6f}")
    if "quarterly_reports_found" in df.columns:
        print(f"Average quarterly reports found: {df['quarterly_reports_found'].mean():.2f}")
    if "mse" in df.columns:
        print(f"Success rate (Standard): {(df['mse'].notna().sum() / len(df)) * 100:.1f}%")
    if "blind_mse" in df.columns:
        print(f"Success rate (Blind): {(df['blind_mse'].notna().sum() / len(df)) * 100:.1f}%")

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