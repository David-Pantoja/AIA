import os
from dotenv import load_dotenv
from insight_generator import InsightGenerator
import numpy as np
import json
from datetime import datetime

# Load environment variables
load_dotenv()

def format_insights(insights):
    """Format insights for pretty printing."""
    if not insights:
        return "No insights available"
    
    insight_data = insights.get("insights", {})
    recommendation = insight_data.get("recommendation", {})
    
    return f"""
Investment Analysis for {insights.get('ticker', 'Unknown')}
Generated on: {insights.get('generated_at', 'Unknown')}
Based on filing date: {insights.get('filing_date', 'Unknown')}

Summary:
{insight_data.get('summary', 'No summary available')}

Financial Health:
{insight_data.get('financial_health', 'No financial health assessment available')}

Recommendation:
Action: {recommendation.get('action', 'Unknown')}
Strong Buy Probability: {recommendation.get('strong_buy_probability', 0):.2%}
Buy Probability: {recommendation.get('buy_probability', 0):.2%}
Hold Probability: {recommendation.get('hold_probability', 0):.2%}
Sell Probability: {recommendation.get('sell_probability', 0):.2%}
Strong Sell Probability: {recommendation.get('strong_sell_probability', 0):.2%}
Confidence Level: {recommendation.get('confidence_level', 0):.2%}

Technical Analysis:
{insight_data.get('technical_analysis', {}).get('trend', 'No trend analysis available')}
Momentum: {insight_data.get('technical_analysis', {}).get('momentum', 'No momentum analysis available')}
Support/Resistance: {insight_data.get('technical_analysis', {}).get('support_resistance', 'No support/resistance analysis available')}

Market Sentiment:
Analyst Consensus: {insight_data.get('market_sentiment', {}).get('analyst_consensus', 'No analyst consensus available')}
Price Momentum: {insight_data.get('market_sentiment', {}).get('price_momentum', 'No price momentum analysis available')}
Volume Trend: {insight_data.get('market_sentiment', {}).get('volume_trend', 'No volume trend analysis available')}

Risk Factors:
{chr(10).join(f"- {risk}" for risk in insight_data.get('risk_factors', ['No risk factors identified']))}
"""

def validate_probability_distribution(probs: dict) -> bool:
    """Validate that probabilities sum to 1 and are between 0 and 1."""
    total = sum(probs.values())
    return (abs(total - 1.0) < 1e-6 and  # Allow for small floating point errors
            all(0 <= p <= 1 for p in probs.values()))

def calculate_mse(yf_probs: dict, model_probs: dict) -> float:
    """Calculate mean square error between two probability distributions."""
    # Ensure both distributions have the same keys
    keys = set(yf_probs.keys()) & set(model_probs.keys())
    if not keys:
        return float('inf')
    
    # Calculate MSE
    squared_errors = [(yf_probs[k] - model_probs[k])**2 for k in keys]
    return np.mean(squared_errors)

def main():
    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OpenAI API key not found in environment variables")
        return

    # Test with a sample ticker and filing date
    ticker = "AAPL"
    filing_date = "2024-02-01"  # Example filing date

    # Test different combinations of data sources
    test_configs = [
        {"use_SEC": True, "use_yfinance": True, "name": "SEC + Market Data"},
        {"use_SEC": True, "use_yfinance": False, "name": "SEC Only"},
        {"use_SEC": False, "use_yfinance": True, "name": "Market Data Only"}
    ]

    for config in test_configs:
        print(f"\n{'='*80}")
        print(f"Testing with {config['name']}")
        print(f"{'='*80}")

        # Initialize the insight generator with current config
        generator = InsightGenerator(
            openai_api_key,
            use_yfinance=config["use_yfinance"],
            use_SEC=config["use_SEC"]
        )

        print(f"\nGenerating insights for {ticker} as of {filing_date}...")
        insight = generator.generate_insight(ticker, filing_date)

        if not insight:
            print("Failed to generate insights")
            continue

        # Print the analysis
        print("\nAnalysis Results:")
        print(json.dumps(insight, indent=2))

        # Check if we have yfinance comparison
        if "yfinance_comparison" in insight:
            yf_comparison = insight["yfinance_comparison"]
            yf_probs = yf_comparison["yfinance_probabilities"]
            model_probs = yf_comparison["model_probabilities"]
            
            # Validate probability distributions
            print("\nValidating probability distributions:")
            print(f"yfinance distribution valid: {validate_probability_distribution(yf_probs)}")
            print(f"model distribution valid: {validate_probability_distribution(model_probs)}")
            
            # Calculate and print MSE
            mse = calculate_mse(yf_probs, model_probs)
            print(f"\nMean Square Error between distributions: {mse:.6f}")
            
            # Print detailed comparison
            print("\nDetailed Probability Comparison:")
            print(f"{'Action':<12} {'yfinance':<12} {'Model':<12} {'Difference':<12}")
            print("-" * 48)
            for action in ["strongBuy", "buy", "hold", "sell", "strongSell"]:
                yf_prob = yf_probs.get(action, 0)
                model_prob = model_probs.get(action, 0)
                diff = yf_comparison["difference"].get(action, 0)
                print(f"{action:<12} {yf_prob:>11.4f} {model_prob:>11.4f} {diff:>11.4f}")
        else:
            print("\nNo yfinance comparison available")

if __name__ == "__main__":
    main()