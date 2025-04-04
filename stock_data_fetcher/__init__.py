from .fetcher import StockDataFetcher
from typing import Dict

def get_historical_prices(ticker: str, start_date: str, end_date: str) -> Dict:
    fetcher = StockDataFetcher()
    return fetcher.get_historical_prices(ticker, start_date, end_date)

def get_analyst_ratings(ticker: str) -> Dict:
    fetcher = StockDataFetcher()
    return fetcher.get_analyst_ratings(ticker)

def get_price_reaction(ticker: str, filing_date: str, days_before: int = 5, days_after: int = 5) -> Dict:
    fetcher = StockDataFetcher()
    return fetcher.get_price_reaction(ticker, filing_date, days_before, days_after)

def calculate_technical_indicators(price_data: Dict) -> Dict:
    fetcher = StockDataFetcher()
    return fetcher.calculate_technical_indicators(price_data) 