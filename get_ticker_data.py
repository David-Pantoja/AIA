import json
from datetime import datetime
from insight_generator import InsightGenerator
from stock_data_fetcher import StockDataFetcher
from sec_edgar_fetcher import fetch_filings

def format_insights(insights):
    """Format insights for pretty printing."""
    if not insights:
        return "No insights available"
    
    insight_data = insights.get("insights", {})
    recommendation = insight_data.get("recommendation", {})
    
    return f"""
Investment Insights for {insights.get('ticker', 'Unknown')}
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
    # Get ticker symbol from user
    ticker = input("Enter ticker symbol (e.g., AAPL): ").strip().upper()
    
    # Set cutoff date to July 18, 2024
    cutoff_date = datetime(2024, 7, 18)
    
    # Initialize fetchers
    insight_generator = InsightGenerator(cutoff_date=cutoff_date)
    stock_fetcher = StockDataFetcher()
    
    print(f"\nFetching data for {ticker} as of {cutoff_date.strftime('%Y-%m-%d')}...")
    
    # Get latest insights
    print("\nGenerating investment insights...")
    insights = insight_generator.get_latest_insights(ticker, cutoff_date=cutoff_date)
    print(format_insights(insights))
    
    # Get analyst ratings
    print("\nFetching analyst ratings...")
    ratings = stock_fetcher.get_analyst_ratings(ticker, cutoff_date=cutoff_date)
    print(f"""
Analyst Ratings:
Mean Target Price: ${ratings.get('mean_target', 'N/A')}
Median Target Price: ${ratings.get('median_target', 'N/A')}
Latest Rating: {ratings.get('latest_rating', 'N/A')}
Latest Target: ${ratings.get('latest_target', 'N/A')}
Buy Ratings: {ratings.get('buy_ratings', 0)}
Hold Ratings: {ratings.get('hold_ratings', 0)}
Sell Ratings: {ratings.get('sell_ratings', 0)}
""")
    
    # Get recent SEC filings
    print("\nFetching recent SEC filings...")
    for filing_type in ["10-Q", "10-K", "8-K"]:
        filings = fetch_filings(ticker, filing_type, cutoff_date=cutoff_date)
        if filings:
            print(f"\nRecent {filing_type} Filings:")
            for filing in filings[:3]:  # Show last 3 filings
                print(f"- {filing['filing_date']}: {filing.get('financials', {}).get('Revenue', 'N/A')} Revenue")
    
    # Save raw data to JSON file
    output_file = f"{ticker}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    data = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "insights": insights,
        "ratings": ratings,
        "filings": {
            "10-Q": fetch_filings(ticker, "10-Q", cutoff_date=cutoff_date),
            "10-K": fetch_filings(ticker, "10-K", cutoff_date=cutoff_date),
            "8-K": fetch_filings(ticker, "8-K", cutoff_date=cutoff_date)
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nRaw data saved to {output_file}")

if __name__ == "__main__":
    main() 