#!/usr/bin/env python3
"""
Script to directly test the insight generator and examine its output structure.
"""

import json
import os
import sys
import logging
from datetime import datetime
from insight_generator import InsightGenerator
from dotenv import load_dotenv

# Configure logging to see SEC data fetch details
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("ERROR: OpenAI API key not found in environment variables.")
    print("Please set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

def test_insight_generator(ticker="AAPL", use_SEC=True, use_yfinance=True, quarters=4, max_search=200):
    """Test the insight generator with a specific ticker.
    
    Args:
        ticker: Stock ticker symbol to analyze
        use_SEC: Whether to use SEC filings data (default: True, matching portfolio_analyzer)
        use_yfinance: Whether to use Yahoo Finance data
        quarters: Number of quarters of data to analyze
        max_search: Maximum number of SEC filings to search through
    """
    print(f"\n=== TESTING INSIGHT GENERATOR ===")
    print(f"Testing insight generation for {ticker}...")
    logger.info("=" * 80)
    logger.info(f"INSIGHT GENERATOR TEST:")
    logger.info(f"Ticker: {ticker}")
    logger.info(f"Parameters: use_SEC={use_SEC}, use_yfinance={use_yfinance}, quarters={quarters}, max_search={max_search}")
    logger.info("=" * 80)
    
    # Provide a warning if using SEC data
    if use_SEC:
        print("\n⚠️ WARNING: SEC data fetching is enabled, which may be slower.")
        print("If you need faster testing, set use_SEC=False.\n")
    
    try:
        # Initialize the insight generator
        cutoff_date = datetime.now()
        print(f"Using cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
        
        insight_generator = InsightGenerator(
            openai_api_key=openai_api_key,
            use_SEC=use_SEC,
            use_yfinance=use_yfinance,
            cutoff_date=cutoff_date
        )
        
        # Generate insights
        print(f"\nGenerating insights for {ticker}...")
        start_time = datetime.now()
        
        insights = insight_generator.generate_insight(
            ticker=ticker,
            quarters=quarters,
            max_search=max_search,
            use_SEC_override=use_SEC,  # Explicitly pass the override
            use_yfinance_override=use_yfinance,
            cutoff_date=cutoff_date
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"Insight generation completed in {duration:.2f} seconds")
        
        # Check if the insight generation was successful
        if not insights:
            print(f"\nERROR: Failed to generate insights for {ticker}")
            return None
        
        # Examine the structure
        print("\nInsight structure:")
        print(f"Type: {type(insights)}")
        print(f"Top-level keys: {list(insights.keys())}")
        
        # Check if SEC data was included
        if use_SEC:
            if "filings" in insights:
                print(f"\nSEC filings included: Yes ({len(insights['filings'])} filings)")
                if insights['filings']:
                    print(f"Most recent filing: {insights['filings'][0].get('form_type')} on {insights['filings'][0].get('filing_date')}")
            elif "filing_date" in insights:
                print(f"\nSEC filing date included: {insights['filing_date']}")
            else:
                print("\nSEC filings data: Not found in results")
        
        # Check if it has a nested 'insights' field
        if "insights" in insights:
            print("\nNested structure detected ('insights' field is present)")
            insights_data = insights["insights"]
            print(f"Insights field keys: {list(insights_data.keys())}")
            
            # Check recommendation structure
            if "recommendation" in insights_data:
                rec = insights_data["recommendation"]
                print(f"\nRecommendation keys: {list(rec.keys())}")
                print(f"Action: {rec.get('action')}")
                print(f"Buy probability: {rec.get('buy_probability')}")
                print(f"Hold probability: {rec.get('hold_probability')}")
                print(f"Sell probability: {rec.get('sell_probability')}")
                print(f"Confidence: {rec.get('confidence_level')}")
        else:
            print("\nFlat structure detected (no 'insights' field)")
            
            # Check if key insight fields are at top level
            if "recommendation" in insights:
                rec = insights["recommendation"]
                print(f"\nRecommendation keys: {list(rec.keys())}")
                print(f"Action: {rec.get('action')}")
                print(f"Buy probability: {rec.get('buy_probability')}")
                print(f"Hold probability: {rec.get('hold_probability')}")
                print(f"Sell probability: {rec.get('sell_probability')}")
                print(f"Confidence: {rec.get('confidence_level')}")
        
        # Save insights to a file
        output_file = f"{ticker}_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(insights, f, indent=2)
        
        print(f"\nInsights saved to {output_file}")
        print(f"Test successful for {ticker}")
        print("=" * 50)
        
        return insights
    
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the insight generator with various settings")
    parser.add_argument("--ticker", "-t", default="AAPL", help="Ticker symbol to analyze")
    parser.add_argument("--no-sec", dest="use_sec", action="store_false", 
                        help="Disable SEC data (default: enabled)")
    parser.add_argument("--no-yfinance", dest="use_yfinance", action="store_false", 
                        help="Disable Yahoo Finance data (default: enabled)")
    parser.add_argument("--quarters", "-q", type=int, default=4, 
                        help="Number of quarters of data to analyze")
    parser.add_argument("--max-search", "-m", type=int, default=200, 
                        help="Maximum SEC filings to search through")
    
    # Default to using SEC data
    parser.set_defaults(use_sec=True)
    
    args = parser.parse_args()
    
    test_insight_generator(
        ticker=args.ticker,
        use_SEC=args.use_sec,
        use_yfinance=args.use_yfinance,
        quarters=args.quarters,
        max_search=args.max_search
    ) 