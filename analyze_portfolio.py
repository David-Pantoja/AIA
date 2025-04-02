#!/usr/bin/env python3
"""
Command-line tool to analyze an investment portfolio.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from portfolio_analyzer import analyze_portfolio

def parse_date(date_str):
    """Parse a date string in YYYY-MM-DD format."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        print(f"Error: Invalid date format '{date_str}'. Expected YYYY-MM-DD.")
        sys.exit(1)

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Analyze an investment portfolio.")
    parser.add_argument(
        "portfolio_file", 
        help="Path to the portfolio JSON file"
    )
    parser.add_argument(
        "--cutoff-date", 
        "-d", 
        help="Cutoff date for historical data (YYYY-MM-DD). Default is today."
    )
    parser.add_argument(
        "--output", 
        "-o", 
        help="Output file to save analysis results (JSON format). If not specified, will save to portfolio_analysis_YYYY-MM-DD.json"
    )
    parser.add_argument(
        "--summary-only", 
        "-s", 
        action="store_true", 
        help="Only output the summary and recommendations"
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true", 
        help="Show detailed information during analysis"
    )
    
    args = parser.parse_args()
    
    # Validate file exists
    if not os.path.isfile(args.portfolio_file):
        print(f"Error: Portfolio file '{args.portfolio_file}' not found.")
        sys.exit(1)
    
    # Set cutoff date
    cutoff_date = parse_date(args.cutoff_date) or datetime.now()
    
    # Configure logging based on verbose flag
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    print(f"Analyzing portfolio from {args.portfolio_file}...")
    print(f"Using cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
    
    # Analyze portfolio
    results = analyze_portfolio(args.portfolio_file, cutoff_date=cutoff_date)
    
    # Create default output filename if not specified
    output_file = args.output
    if not output_file:
        date_str = datetime.now().strftime("%Y-%m-%d")
        portfolio_name = os.path.splitext(os.path.basename(args.portfolio_file))[0]
        output_file = f"{portfolio_name}_analysis_{date_str}.json"
        
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error: Failed to create directory '{output_dir}': {e}")
            sys.exit(1)
    
    # Save the results to JSON file
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Full analysis saved to {output_file}")
    except IOError as e:
        print(f"Error: Failed to write to output file '{output_file}': {e}")
        sys.exit(1)
    
    # Print summary information
    if args.summary_only:
        print_summary_only(results)
    else:
        print_full_summary(results)
    
    return 0

def print_summary_only(results):
    """Print just the summary and recommendations."""
    print("\nPORTFOLIO SUMMARY:")
    print("=" * 80)
    print(results.get("summary", "No summary available."))
    
    print("\nREBALANCING RECOMMENDATIONS:")
    print("=" * 80)
    recommendations = results.get("rebalancing_recommendations", [])
    if recommendations:
        for rec in recommendations:
            print(f"- {rec.get('ticker')}: {rec.get('action').upper()} - {rec.get('reason')}")
    else:
        print("No rebalancing recommendations.")
    
    print("\nRISK ASSESSMENT:")
    print("=" * 80)
    print(results.get("risk_assessment", "No risk assessment available."))
    
    print("\nDIVERSIFICATION SUGGESTIONS:")
    print("=" * 80)
    print(results.get("diversification_suggestions", "No diversification suggestions available."))

def print_full_summary(results):
    """Print detailed analysis results."""
    # Portfolio information
    portfolio = results.get("portfolio", {})
    positions = portfolio.get("positions", [])
    analyzed_at = portfolio.get("analyzed_at", "Unknown")
    
    print("\nPORTFOLIO ANALYSIS SUMMARY:")
    print("=" * 80)
    print(f"Analyzed at: {analyzed_at}")
    print(f"Positions analyzed: {len(positions)}")
    
    # Print all positions
    print("\nPOSITIONS:")
    print("-" * 80)
    for position in positions:
        ticker = position.get("ticker", "Unknown")
        shares = position.get("shares", 0)
        cost_basis = position.get("cost_basis", 0)
        
        # Get insight for this position if available
        insight = results.get("insights", {}).get(ticker, {})
        recommendation = insight.get("insights", {}).get("recommendation", {})
        action = recommendation.get("action", "Unknown")
        
        print(f"{ticker}: {shares} shares @ ${cost_basis:.2f} - Recommendation: {action}")
    
    # Conflicts
    conflicts = results.get("conflicts", {})
    has_conflicts = conflicts.get("has_conflicts", False)
    
    print("\nCONFLICTS:")
    print("-" * 80)
    if has_conflicts:
        for conflict in conflicts.get("conflicts", []):
            print(f"Conflict between {', '.join(conflict.get('positions', []))}")
            print(f"  Reason: {conflict.get('reason')}")
            print(f"  Explanation: {conflict.get('explanation')}")
    else:
        print("No conflicts detected in the portfolio.")
    
    # Print summary sections
    print_summary_only(results)

if __name__ == "__main__":
    sys.exit(main()) 