#!/usr/bin/env python3
"""
Simple script to run the portfolio analyzer on the sample portfolio.
"""

import json
from datetime import datetime
from portfolio_analyzer import analyze_portfolio

def parse_date(date_str):
    """Parse a date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        print(f"Warning: Could not parse date '{date_str}'. Using current date instead.")
        return datetime.now()

def main():
    # Set the portfolio file path
    portfolio_file = "sample_portfolioA.json"
    
    # Load the portfolio file to get the date and configuration
    try:
        with open(portfolio_file, 'r') as f:
            portfolio_data = json.load(f)
        
        # Extract the date - check multiple possible field names
        date_str = None
        for field in ["date", "created_at", "creation_date"]:
            if field in portfolio_data:
                date_str = portfolio_data[field]
                print(f"Found portfolio date in field '{field}': {date_str}")
                break
                
        cutoff_date = parse_date(date_str) if date_str else datetime.now()
        
        # Extract configuration if available
        config = portfolio_data.get("config", {})
        quarters = config.get("quarters", 4)
        max_search = config.get("max_search", 50)
        use_SEC = config.get("use_SEC", True)
        use_yfinance = config.get("use_yfinance", True)
        
        print(f"Using configuration: quarters={quarters}, max_search={max_search}, "
              f"use_SEC={use_SEC}, use_yfinance={use_yfinance}")
        
    except Exception as e:
        print(f"Error reading portfolio file: {e}")
        print("Using default values for analysis parameters.")
        cutoff_date = datetime.now()
        quarters = 4
        max_search = 50
        use_SEC = True
        use_yfinance = True
    
    print(f"Starting analysis of {portfolio_file} with cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
    
    # Run the analysis with parameters from the portfolio file
    try:
        result = analyze_portfolio(
            portfolio_file, 
            cutoff_date=cutoff_date,
            quarters=quarters,
            max_search=max_search,
            use_SEC=use_SEC,
            use_yfinance=use_yfinance
        )
        
        # Print a simple summary
        print("\nAnalysis complete!")
        print(f"Portfolio positions: {len(result.get('portfolio', {}).get('positions', []))}")
        
        # Print conflict information
        conflicts = result.get("conflicts", {})
        if conflicts.get("has_conflicts", False):
            print("\nConflicts detected:")
            for conflict in conflicts.get("conflicts", []):
                print(f"- Conflict between {', '.join(conflict.get('positions', []))}")
                print(f"  Reason: {conflict.get('reason', 'Unknown')}")
        else:
            print("\nNo conflicts detected in the portfolio.")

        # Print recommendations
        print("\nRebalancing Recommendations:")
        recs = result.get("rebalancing_recommendations", [])
        if recs:
            for rec in recs:
                print(f"- {rec.get('ticker')}: {rec.get('action', '').upper()} - {rec.get('reason', 'No reason provided')}")
        else:
            print("No specific rebalancing recommendations.")
        
        # Print risk assessment
        print("\nRisk Assessment:")
        print(result.get("risk_assessment", "No risk assessment available."))
        
        # Print diversification suggestions
        print("\nDiversification Suggestions:")
        print(result.get("diversification_suggestions", "No diversification suggestions available."))
        
        # Save the full analysis to a file
        output_file = f"sample_portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nFull analysis saved to {output_file}")
    except Exception as e:
        print(f"Error during portfolio analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 