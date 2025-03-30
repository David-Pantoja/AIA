from insight_generator import InsightGenerator
import json
from datetime import datetime

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
Buy Probability: {recommendation.get('buy_probability', 0):.2%}
Hold Probability: {recommendation.get('hold_probability', 0):.2%}
Sell Probability: {recommendation.get('sell_probability', 0):.2%}
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

def main():
    # Example tickers to analyze
    tickers = ["AAPL"]#, "MSFT", "GOOGL"]
    
    # Initialize the insight generator
    generator = InsightGenerator()
    
    # Analyze each ticker
    for ticker in tickers:
        print(f"\n{'='*80}")
        print(f"Analyzing {ticker}...")
        
        # Get insights
        insights = generator.get_latest_insights(ticker)
        
        # Print formatted insights
        print(format_insights(insights))
        
        # Save raw data to JSON file
        output_file = f"{ticker}_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(insights, f, indent=2)
        print(f"\nRaw insights saved to {output_file}")

if __name__ == "__main__":
    main()