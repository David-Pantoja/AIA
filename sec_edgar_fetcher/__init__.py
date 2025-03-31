# Main module interface 
from .fetcher import SECFetcher
from .parser import parse_filings
from typing import List, Dict, Optional

def fetch_filings(ticker: str, filing_type: str, date_range: Optional[tuple] = None,
                 quarters: int = None, max_search: int = None) -> List[Dict]:
    """Fetch and parse SEC filings for a given ticker, filing type, and optional date range.
    
    Args:
        ticker: The stock ticker symbol
        filing_type: Type of filing to fetch ("10-K", "10-Q", "8-K", or "ALL")
        date_range: Optional tuple of (start_date, end_date)
        quarters: Number of quarterly/annual reports to find (10-Q + 10-K count)
        max_search: Maximum number of iterations before terminating early
    """
    fetcher = SECFetcher()
    filings = fetcher.fetch_filings(ticker, filing_type, date_range, quarters, max_search)
    return parse_filings(filings) 