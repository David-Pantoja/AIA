# Main module interface 
from .fetcher import SECFetcher
from .parser import parse_filings
from typing import List, Dict, Optional

def fetch_filings(ticker: str, filing_type: str, date_range: Optional[tuple] = None) -> List[Dict]:
    """Fetch and parse SEC filings for a given ticker, filing type, and optional date range."""
    fetcher = SECFetcher()
    filings = fetcher.fetch_filings(ticker, filing_type, date_range)
    return parse_filings(filings) 