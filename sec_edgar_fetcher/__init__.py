from .fetcher import SECFetcher
from .parser import parse_filings
from typing import List, Dict, Optional
from datetime import datetime

def fetch_filings(ticker: str, filing_type: str, date_range: Optional[tuple] = None,
                 quarters: int = None, max_search: int = None,
                 cutoff_date: Optional[datetime] = None) -> List[Dict]:

    fetcher = SECFetcher()
    filings = fetcher.fetch_filings(ticker, filing_type, date_range, quarters, max_search, cutoff_date)
    return parse_filings(filings) 