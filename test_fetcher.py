#!/usr/bin/env python
import json
from datetime import datetime, timedelta
from stock_data_fetcher.fetcher import StockDataFetcher

def main():
    """Test how StockDataFetcher handles cutoff dates"""
    
    # Initialize the fetcher
    fetcher = StockDataFetcher()
    
    # Choose a stock ticker to test with
    ticker = "AAPL"
    
    # Set up different cutoff dates for testing
    today = datetime.now()
    cutoff_recent = today - timedelta(days=30)  # 30 days ago
    cutoff_past = today - timedelta(days=180)   # 6 months ago
    
    print(f"Testing fetcher.py cutoff date handling with {ticker}")
    print(f"Current date: {today.strftime('%Y-%m-%d')}")
    print(f"Recent cutoff: {cutoff_recent.strftime('%Y-%m-%d')}")
    print(f"Past cutoff: {cutoff_past.strftime('%Y-%m-%d')}")
    print("-" * 80)
    
    # Test 1: Get historical prices with today as cutoff
    print("\n1. Historical prices with today as cutoff:")
    start_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    prices_today = fetcher.get_historical_prices(ticker, start_date, end_date)
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Got {len(prices_today.get('dates', []))} days of price data")
    if prices_today.get('dates'):
        print(f"  Date range received: {prices_today['dates'][0]} to {prices_today['dates'][-1]}")
    
    # Test 2: Get historical prices with past cutoff
    print("\n2. Historical prices with past cutoff:")
    start_date = (cutoff_past - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = cutoff_past.strftime("%Y-%m-%d")
    prices_past = fetcher.get_historical_prices(ticker, start_date, end_date)
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Got {len(prices_past.get('dates', []))} days of price data")
    if prices_past.get('dates'):
        print(f"  Date range received: {prices_past['dates'][0]} to {prices_past['dates'][-1]}")
    
    # Test 3: Get analyst ratings with today as cutoff
    print("\n3. Analyst ratings with today as cutoff:")
    ratings_today = fetcher.get_analyst_ratings(ticker, cutoff_date=today)
    print(f"  Mean target: {ratings_today.get('mean_target')}")
    print(f"  Raw recommendations: {json.dumps(ratings_today.get('raw_recommendations', {}), indent=2)}")
    
    # Test 4: Get analyst ratings with past cutoff
    print("\n4. Analyst ratings with past cutoff:")
    ratings_past = fetcher.get_analyst_ratings(ticker, cutoff_date=cutoff_past)
    print(f"  Mean target: {ratings_past.get('mean_target')}")
    print(f"  Raw recommendations: {json.dumps(ratings_past.get('raw_recommendations', {}), indent=2)}")
    
    # Test 5: Compare the analyst ratings between different cutoffs
    print("\n5. Comparison of analyst ratings between different cutoffs:")
    print(f"  Current cutoff - Buy ratings: {ratings_today.get('buy_ratings', 0)}, Hold: {ratings_today.get('hold_ratings', 0)}, Sell: {ratings_today.get('sell_ratings', 0)}")
    print(f"  Past cutoff - Buy ratings: {ratings_past.get('buy_ratings', 0)}, Hold: {ratings_past.get('hold_ratings', 0)}, Sell: {ratings_past.get('sell_ratings', 0)}")

if __name__ == "__main__":
    main() 