#!/usr/bin/env python3
"""
Minimal script to call the insight generator and print the results.
"""

import json
from insight_generator import InsightGenerator
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Get ticker
ticker = input("Enter ticker symbol (default: AAPL): ") or "AAPL"
print(f"Generating insights for {ticker}...")

# Generate insights - disable SEC data for faster results
generator = InsightGenerator(api_key, use_SEC=True)
insights = generator.generate_insight(ticker=ticker)

# Print and save results
print(f"\nGot insight with keys: {list(insights.keys())}")

# Save to JSON file
with open(f"{ticker}_insights.json", 'w') as f:
    json.dump(insights, f, indent=2)
print(f"Saved to {ticker}_insights.json")

# Print recommendation
if "insights" in insights:
    rec = insights["insights"].get("recommendation", {})
else:
    rec = insights.get("recommendation", {})

# Print full json
print("\nRecommendation:")
try:
    print(json.dumps(rec, indent=2))
    
    print("\nFull insights:")
    print(json.dumps(insights, indent=2))
except Exception as e:
    print(f"Error printing JSON: {str(e)}")
    print("Raw data:")
    print(insights)