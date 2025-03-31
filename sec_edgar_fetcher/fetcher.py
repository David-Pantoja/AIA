# SEC API interaction logic 

import requests
from typing import Dict, List, Optional
from .utils import rate_limited, retry_on_http_error
from dotenv import load_dotenv
import os
from datetime import datetime
import concurrent.futures
import time
from ratelimit import limits, sleep_and_retry
import backoff
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SECFetcher:
    def __init__(self):
        self.base_url = "https://data.sec.gov"
        self.headers = {
            "User-Agent": "SECFetcher/1.0 ({email}) Python/3.11.5".format(email=os.getenv("email")),
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov"
        }
        # Initialize rate limiter for 10 requests per second
        self.rate_limiter = limits(calls=10, period=1)(sleep_and_retry(lambda: None))
        # Add a small delay between requests to be more conservative
        self.request_delay = 0.2  # 200ms between requests
        self.max_retries = 3
        self.retry_delay = 5  # 5 seconds between retries

    def _make_request(self, url: str, timeout: int = 10) -> Optional[requests.Response]:
        """Make a rate-limited request with retry logic."""
        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting
                self.rate_limiter()
                # Add a small delay between requests
                time.sleep(self.request_delay)
                
                response = requests.get(url, headers=self.headers, timeout=timeout)
                
                # Handle rate limit errors
                if response.status_code == 429:  # Too Many Requests
                    logger.warning(f"Rate limit hit on attempt {attempt + 1}/{self.max_retries}, waiting {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    continue
                
                # Handle other error status codes
                if response.status_code >= 400:
                    logger.warning(f"HTTP {response.status_code} error on attempt {attempt + 1}/{self.max_retries}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return None
                
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return None
        
        logger.error(f"Failed to make request after {self.max_retries} attempts")
        return None

    @rate_limited
    @retry_on_http_error
    def get_cik(self, ticker: str) -> Optional[str]:
        """Fetch CIK for a given ticker symbol."""
        url = "https://www.sec.gov/files/company_tickers.json"
        response = self._make_request(url)
        if not response or response.status_code != 200:
            return None
        try:
            data = response.json()
            for company in data.values():
                if company["ticker"] == ticker:
                    return str(company["cik_str"]).zfill(10)
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {e}")
            return None
        return None

    def _get_form_type_from_document(self, doc_content: str) -> Optional[str]:
        """Extract form type from document content."""
        if "CONFORMED SUBMISSION TYPE: 8-K" in doc_content:
            return "8-K"
        elif "CONFORMED SUBMISSION TYPE: 10-K" in doc_content:
            return "10-K"
        elif "CONFORMED SUBMISSION TYPE: 10-Q" in doc_content:
            return "10-Q"
        return None

    def _get_form_type_from_filename(self, filename: str) -> Optional[str]:
        """Extract form type from filename patterns."""
        filename = filename.lower()
        if any(pattern in filename for pattern in ['_8k.htm', '_8k_htm.xml', '8-k', '8k']):
            return "8-K"
        elif any(pattern in filename for pattern in ['_10k.htm', '_10k_htm.xml', '10-k', '10k']):
            return "10-K"
        elif any(pattern in filename for pattern in ['_10q.htm', '_10q_htm.xml', '10-q', '10q']):
            return "10-Q"
        return None

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
    def _process_submission(self, cik: str, item: Dict, current_date: datetime) -> Optional[Dict]:
        """Process a single submission and return its metadata if valid."""
        try:
            if item.get("type") != "folder.gif":
                return None

            accession_number = item.get("name")
            last_modified = item.get("last-modified", "")
            
            if not accession_number or not last_modified:
                return None

            # Parse the last-modified date
            date_parts = last_modified.split()[0].split('-')
            if len(date_parts) != 3:
                return None

            year = int(date_parts[0])
            month = int(date_parts[1])
            day = int(date_parts[2])
            
            # Skip future dates and invalid dates
            filing_date = datetime(year, month, day)
            
            # For 2025 filings, only skip if the date is in the future
            if year == 2025 and filing_date > current_date:
                return None
            # For other years, skip if before 2023
            elif year < 2023:
                return None

            # Get the form type from the filing metadata
            form_type = None
            form_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/index.json"
            
            form_response = self._make_request(form_url)
            if not form_response or form_response.status_code != 200:
                return None

            form_data = form_response.json()
            form_items = form_data.get("directory", {}).get("item", [])
            
            # First try to get form type from the main document
            main_doc = None
            for form_item in form_items:
                name = form_item.get("name", "").lower()
                if name.endswith('.txt') and not name.endswith('-index.txt'):
                    main_doc = name
                    break
            
            if main_doc:
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{main_doc}"
                doc_response = self._make_request(doc_url)
                if doc_response and doc_response.status_code == 200:
                    form_type = self._get_form_type_from_document(doc_response.text)
            
            # If no form type found in main document, try file patterns
            if not form_type:
                for form_item in form_items:
                    name = form_item.get("name", "")
                    form_type = self._get_form_type_from_filename(name)
                    if form_type:
                        break
            
            # If still no form type, try index file
            if not form_type:
                for form_item in form_items:
                    name = form_item.get("name", "").lower()
                    if name.endswith('-index.txt'):
                        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{name}"
                        index_response = self._make_request(index_url)
                        if index_response and index_response.status_code == 200:
                            form_type = self._get_form_type_from_document(index_response.text)
                            if form_type:
                                break
            
            if form_type in ["10-K", "10-Q", "8-K"]:
                return {
                    "form_type": form_type,
                    "filing_date": f"{year}-{month:02d}-{day:02d}",
                    "accession_number": accession_number
                }
            
            return None
        except Exception as e:
            logger.error(f"Error processing submission: {e}")
            return None

    @rate_limited
    @retry_on_http_error
    def get_submissions(self, cik: str, max_items: int = 50, quarters: int = None, max_search: int = None) -> List[Dict]:
        """Fetch submission metadata for a given CIK using parallel processing.
        
        Args:
            cik: The CIK number for the company
            max_items: Maximum number of items to process (default: 50)
            quarters: Number of quarterly/annual reports to find (10-Q + 10-K count)
            max_search: Maximum number of iterations before terminating early
        """
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/index.json"
        response = self._make_request(url)
        if not response or response.status_code != 200:
            return []
        
        try:
            data = response.json()
            items = data.get("directory", {}).get("item", [])
            logger.info(f"Found {len(items)} items in directory")
            
            # Sort items by last-modified date in reverse order
            items.sort(key=lambda x: x.get("last-modified", ""), reverse=True)
            
            # Process submissions in parallel with fewer workers
            submissions = []
            current_date = datetime.now()
            iteration_count = 0
            quarterly_report_count = 0
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Create futures for each item
                future_to_item = {
                    executor.submit(self._process_submission, cik, item, current_date): item 
                    for item in items
                }
                
                # Process completed futures
                for future in concurrent.futures.as_completed(future_to_item):
                    result = future.result()
                    iteration_count += 1
                    
                    # Check if we've reached max_search
                    if max_search and iteration_count >= max_search:
                        logger.info(f"Reached max_search limit of {max_search} iterations")
                        break
                    
                    if result:
                        submissions.append(result)
                        # Count quarterly/annual reports
                        if result["form_type"] in ["10-Q", "10-K"]:
                            quarterly_report_count += 1
                            logger.info(f"Found {quarterly_report_count} quarterly/annual reports")
                            
                            # Check if we've reached the quarters target
                            if quarters and quarterly_report_count >= quarters:
                                logger.info(f"Reached target of {quarters} quarterly/annual reports")
                                break
            
            # Filter submissions to only include those up to the point where we stopped
            if max_search and iteration_count >= max_search:
                submissions = submissions[:max_search]
            
            logger.info(f"\nFound {len(submissions)} valid submissions")
            logger.info(f"Total quarterly/annual reports: {quarterly_report_count}")
            return submissions
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"Submissions JSON Decode Error: {e}")
            return []

    @rate_limited
    @retry_on_http_error
    def get_filing_document(self, cik: str, accession_number: str) -> str:
        """Fetch the filing document for a given CIK and accession number."""
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number.replace('-', '')}/index.json"
        try:
            response = self._make_request(url)
            if not response or response.status_code != 200:
                return ""
                
            data = response.json()
            items = data.get("directory", {}).get("item", [])
            
            # Look for the main filing document
            document_url = None
            for item in items:
                name = item.get("name", "").lower()
                if name.endswith('.txt') and not name.endswith('-index.txt'):
                    document_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number.replace('-', '')}/{name}"
                    break
            
            if document_url:
                response = self._make_request(document_url)
                if response and response.status_code == 200:
                    return response.text
            return ""
        except Exception as e:
            logger.error(f"Error fetching document: {e}")
            return ""

    def fetch_filings(self, ticker: str, filing_type: str, date_range: Optional[tuple] = None, 
                     quarters: int = None, max_search: int = None) -> List[Dict]:
        """Fetch filings for a given ticker, filing type, and optional date range.
        
        Args:
            ticker: The stock ticker symbol
            filing_type: Type of filing to fetch ("10-K", "10-Q", "8-K", or "ALL")
            date_range: Optional tuple of (start_date, end_date)
            quarters: Number of quarterly/annual reports to find (10-Q + 10-K count)
            max_search: Maximum number of iterations before terminating early
        """
        cik = self.get_cik(ticker)
        if not cik:
            return []

        # Get all submissions once
        submissions = self.get_submissions(cik, quarters=quarters, max_search=max_search)
        logger.info(f"\nFound {len(submissions)} total submissions")
        
        # Process all submissions in parallel with fewer workers
        filings = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Create futures for each submission
            future_to_submission = {
                executor.submit(
                    lambda s: {
                        "accession_number": s["accession_number"],
                        "filing_date": s["filing_date"],
                        "form_type": s["form_type"],
                        "document": self.get_filing_document(cik, s["accession_number"])
                    } if (filing_type == "ALL" or s["form_type"] == filing_type) else None,
                    submission
                ): submission 
                for submission in submissions
            }
            
            # Process completed futures
            for future in concurrent.futures.as_completed(future_to_submission):
                result = future.result()
                if result:
                    filings.append(result)
        
        logger.info(f"\nProcessed {len(filings)} {filing_type if filing_type != 'ALL' else 'total'} filings")
        return filings 