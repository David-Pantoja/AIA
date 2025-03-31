# SEC API interaction logic 

import requests
from typing import Dict, List, Optional
from .utils import rate_limited, retry_on_http_error
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
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
        # Update headers based on working test script
        self.headers = {
            "User-Agent": "David Pantoja davidpantoja2727@gmail.com", # Updated User-Agent
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov" # Updated Host
        }
        # Initialize rate limiter for 5 requests per second (more conservative)
        self.rate_limiter = limits(calls=5, period=1)(sleep_and_retry(lambda: None))
        # Add a larger base delay between requests
        self.request_delay = 0.5  # 500ms between requests
        self.max_retries = 5  # Increased from 3
        self.retry_delay = 10  # Increased from 5 seconds
        self.consecutive_rate_limits = 0  # Track consecutive rate limits
        self.last_request_time = 0  # Track last request time
        self.min_request_interval = 0.5  # Minimum time between requests

    def _adaptive_delay(self):
        """Calculate adaptive delay based on rate limit history."""
        if self.consecutive_rate_limits > 0:
            # Exponential backoff based on consecutive rate limits
            delay = self.request_delay * (2 ** (self.consecutive_rate_limits - 1))
            # Cap the maximum delay at 10 seconds
            return min(delay, 10.0)
        return self.request_delay

    def _make_request(self, url: str, timeout: int = 30) -> Optional[requests.Response]:
        """Make a rate-limited request with retry logic."""
        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting
                self.rate_limiter()
                
                # Calculate adaptive delay
                delay = self._adaptive_delay()
                
                # Ensure minimum time between requests
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < self.min_request_interval:
                    time.sleep(self.min_request_interval - time_since_last)
                
                # Add the adaptive delay
                time.sleep(delay)
                
                # Update last request time
                self.last_request_time = time.time()
                
                # Increase timeout for larger documents
                if 'Archives/edgar/data' in url and not url.endswith('.json'):
                    timeout = 60  # 60 seconds for document fetches
                
                logger.debug(f"Making request to: {url} with delay: {delay:.2f}s")
                response = requests.get(url, headers=self.headers, timeout=timeout)
                
                # Handle rate limit errors
                if response.status_code == 429:  # Too Many Requests
                    self.consecutive_rate_limits += 1
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limit hit on attempt {attempt + 1}/{self.max_retries}, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Reset consecutive rate limits on successful request
                    self.consecutive_rate_limits = 0
                
                # Handle 404 errors with detailed logging
                if response.status_code == 404:
                    logger.warning(f"404 error on attempt {attempt + 1}/{self.max_retries} for URL: {url}")
                    logger.warning(f"Response content: {response.text[:500]}")  # Log first 500 chars of response
                    return None
                
                # Handle other error status codes
                if response.status_code >= 400:
                    logger.warning(f"HTTP {response.status_code} error on attempt {attempt + 1}/{self.max_retries} for URL: {url}")
                    logger.warning(f"Response content: {response.text[:500]}")
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        time.sleep(wait_time)
                        continue
                    return None
                
                return response
                
            except requests.exceptions.Timeout:
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries} for URL: {url}, waiting {wait_time} seconds...")
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                    continue
                return None
                
            except requests.exceptions.ConnectionError as e:
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Connection error on attempt {attempt + 1}/{self.max_retries} for URL: {url}: {e}, waiting {wait_time} seconds...")
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                    continue
                return None
                
            except requests.exceptions.RequestException as e:
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Request error on attempt {attempt + 1}/{self.max_retries} for URL: {url}: {e}, waiting {wait_time} seconds...")
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                    continue
                return None
        
        logger.error(f"Failed to make request to {url} after {self.max_retries} attempts")
        return None

    @rate_limited
    @retry_on_http_error
    def get_cik(self, ticker: str) -> Optional[str]:
        """Fetch CIK for a given ticker symbol."""
        # This URL is for the CIK lookup, Host should be www.sec.gov for this specific request
        cik_lookup_url = "https://www.sec.gov/files/company_tickers.json"
        cik_lookup_headers = self.headers.copy()
        cik_lookup_headers["Host"] = "www.sec.gov"
        
        logger.info(f"Fetching CIK for ticker {ticker} from {cik_lookup_url}")
        
        # Use exponential backoff for retries
        max_retries = 5
        base_delay = 2  # Start with 2 seconds delay
        
        for attempt in range(max_retries):
            try:
                # Apply rate limiting
                self.rate_limiter()
                # Add a small delay between requests
                time.sleep(self.request_delay)
                
                response = requests.get(cik_lookup_url, headers=cik_lookup_headers, timeout=30)
                
                # Handle rate limit errors with exponential backoff
                if response.status_code == 429:
                    wait_time = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limit hit on attempt {attempt + 1}/{max_retries}, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                
                try:
                    data = response.json()
                    logger.info(f"Found {len(data)} companies in SEC database")
                    
                    # The data structure is a dictionary where keys are indices and values are company info
                    for company_data in data.values():
                        if company_data.get("ticker") == ticker.upper():
                            cik = str(company_data["cik_str"]).zfill(10)
                            logger.info(f"Found CIK {cik} for ticker {ticker}")
                            return cik
                            
                    logger.error(f"No CIK found for ticker {ticker}")
                    return None
                    
                except requests.exceptions.JSONDecodeError as e:
                    logger.error(f"CIK lookup JSON Decode Error: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(base_delay * (2 ** attempt))
                        continue
                    return None
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch company tickers (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                return None
                
        logger.error(f"Failed to fetch CIK for {ticker} after {max_retries} attempts")
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
            
            # Skip dates before 2023
            if year < 2023:
                return None
            # Skip dates more than 30 days in the future
            elif filing_date > (current_date + timedelta(days=30)):
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
    def get_submissions(self, cik: str, max_items: int = 50, quarters: int = None, max_search: int = None, cutoff_date: Optional[datetime] = None) -> List[Dict]:
        """Fetch submission metadata for a given CIK using the submissions endpoint.
        
        Args:
            cik: The CIK number for the company
            max_items: Maximum number of items to process (default: 50)
            quarters: Number of quarterly/annual reports to find (10-Q + 10-K count)
            max_search: Maximum number of iterations before terminating early
            cutoff_date: Optional datetime to use as cutoff for data fetching
        """
        # Ensure CIK is properly formatted (10 digits with leading zeros)
        cik = str(cik).zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        
        logger.info(f"Fetching submissions from: {url}")
        response = self._make_request(url)
        
        if not response or response.status_code != 200:
            return []
            
        try:
            data = response.json()
            if "filings" not in data or "recent" not in data["filings"]:
                return []
                
            recent_filings = data["filings"]["recent"]
            
            form_types = recent_filings.get("form", [])
            filing_dates = recent_filings.get("filingDate", [])
            accession_numbers = recent_filings.get("accessionNumber", [])
            
            if not form_types or len(form_types) != len(filing_dates) or len(form_types) != len(accession_numbers):
                return []
                
            logger.info(f"Found {len(form_types)} total recent filings listed for CIK {cik}")
            
            # Create lists of indices for each relevant form type
            indices_10k = [i for i, ft in enumerate(form_types) if ft == "10-K"]
            indices_10q = [i for i, ft in enumerate(form_types) if ft == "10-Q"]
            indices_8k = [i for i, ft in enumerate(form_types) if ft == "8-K"]
            
            logger.info(f"Found {len(indices_10k)} 10-K filings listed.")
            logger.info(f"Found {len(indices_10q)} 10-Q filings listed.")
            logger.info(f"Found {len(indices_8k)} 8-K filings listed.")
            
            submissions = []
            current_date = cutoff_date or datetime.now()
            quarterly_report_count = 0
            processed_indices = set()
            latest_quarterly_date = None

            # Process 10-K and 10-Q filings up to 'quarters' limit
            if quarters is not None and quarters > 0:
                # Combine and sort indices for 10-K and 10-Q by filing date (descending)
                quarterly_indices = sorted(
                    indices_10k + indices_10q,
                    key=lambda i: filing_dates[i],
                    reverse=True
                )
                
                logger.info(f"Processing top {quarters} quarterly/annual reports (10-K/10-Q)...")
                for i in quarterly_indices:
                    if quarterly_report_count >= quarters:
                        break
                    if i in processed_indices:
                        continue
                        
                    filing_date_str = filing_dates[i]
                    accession_number_str = accession_numbers[i].replace('-', '')
                    form_type_str = form_types[i]
                    
                    try:
                        filing_dt = datetime.strptime(filing_date_str, "%Y-%m-%d")
                        if filing_dt.year < 2000 or filing_dt > current_date:
                            continue
                    except ValueError:
                        continue
                    
                    submission = {
                        "form_type": form_type_str,
                        "filing_date": filing_date_str,
                        "accession_number": accession_number_str
                    }
                    
                    submissions.append(submission)
                    processed_indices.add(i)
                    quarterly_report_count += 1
                    
                    # Track the date of the most recent quarterly report
                    if latest_quarterly_date is None or filing_dt > latest_quarterly_date:
                        latest_quarterly_date = filing_dt
                
                logger.info(f"Added {quarterly_report_count} quarterly/annual reports based on limit.")
            
            # Process 8-K filings only within the same time range as quarterly reports
            if latest_quarterly_date:
                logger.info(f"Processing 8-K filings from {latest_quarterly_date.strftime('%Y-%m-%d')} onwards...")
                eight_k_count = 0
                sorted_8k_indices = sorted(indices_8k, key=lambda i: filing_dates[i], reverse=True)
                
                for i in sorted_8k_indices:
                    if i in processed_indices:
                        continue
                        
                    filing_date_str = filing_dates[i]
                    try:
                        filing_dt = datetime.strptime(filing_date_str, "%Y-%m-%d")
                        # Only include 8-K filings from the date of the most recent quarterly report
                        if filing_dt < latest_quarterly_date:
                            continue
                        if filing_dt.year < 2000 or filing_dt > current_date:
                            continue
                    except ValueError:
                        continue
                    
                    submission = {
                        "form_type": "8-K",
                        "filing_date": filing_date_str,
                        "accession_number": accession_numbers[i].replace('-', '')
                    }
                    submissions.append(submission)
                    processed_indices.add(i)
                    eight_k_count += 1
                
                logger.info(f"Added {eight_k_count} 8-K reports within the quarterly report time range.")
            
            # Final sort of all collected submissions by filing date (descending)
            submissions.sort(key=lambda x: x["filing_date"], reverse=True)
            
            logger.info(f"Returning {len(submissions)} valid submissions after filtering and sorting.")
            return submissions
            
        except Exception as e:
            logger.error(f"Error processing submissions: {e}")
            return []

    @rate_limited
    @retry_on_http_error
    def get_filing_document(self, cik: str, accession_number: str) -> str:
        """Fetch the primary filing document text for a given CIK and accession number.
        
        Identifies the primary document (usually ending in .htm or .txt, but not index files)
        from the index file.
        """
        # Accession number needs to be without dashes for index URL, but CIK needs leading zeros.
        cik_formatted = str(cik).zfill(10)
        accession_nodash = accession_number.replace('-', '')
        
        # URL to the index JSON file for the filing
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_formatted}/{accession_nodash}/index.json"
        logger.info(f"Fetching filing index: {index_url}")
        
        # Use headers with Host: www.sec.gov for Archives access
        archive_headers = self.headers.copy()
        archive_headers["Host"] = "www.sec.gov"
        
        try:
            # Make request using requests directly as _make_request uses data.sec.gov host by default
            response = requests.get(index_url, headers=archive_headers, timeout=30)
            response.raise_for_status() # Check for HTTP errors
            index_data = response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching filing index {index_url}: {e}")
            return ""
        except requests.exceptions.JSONDecodeError as e:
             logger.error(f"Error decoding filing index JSON from {index_url}: {e}")
             return ""

        try:
            directory_items = index_data.get("directory", {}).get("item", [])
            if not directory_items:
                logger.warning(f"No items found in directory structure for index {index_url}")
                return ""
            
            # Find the primary filing document (e.g., form10k.htm, d49....htm, or the .txt file)
            primary_doc_name = None
            potential_docs = []
            for item in directory_items:
                name = item.get("name", "").lower()
                # Common primary document patterns
                if name.endswith(('.htm', '.html')) and 'index' not in name and 'rendering' not in name and 'summary' not in name:
                    potential_docs.append(item.get("name"))
                elif name.endswith('.txt') and 'index' not in name:
                    potential_docs.append(item.get("name")) # .txt often contains the full submission

            # Prioritize .htm/.html over .txt if available, otherwise take the first potential doc
            # This logic might need refinement based on SEC filing structures
            for doc in potential_docs:
                if doc.lower().endswith(('.htm', '.html')):
                    primary_doc_name = doc
                    break
            if not primary_doc_name and potential_docs:
                 primary_doc_name = potential_docs[0] # Fallback to first found (.txt likely)

            if not primary_doc_name:
                logger.warning(f"Could not identify primary document in index {index_url}")
                logger.debug(f"Directory items: {directory_items}")
                return ""

            # Construct URL for the primary document
            document_url = f"https://www.sec.gov/Archives/edgar/data/{cik_formatted}/{accession_nodash}/{primary_doc_name}"
            logger.info(f"Fetching primary document: {document_url}")
            
            # Fetch the document content
            doc_response = requests.get(document_url, headers=archive_headers, timeout=60) # Longer timeout for docs
            doc_response.raise_for_status()
            
            # Return document text (consider encoding issues, try utf-8 first)
            try:
                return doc_response.content.decode('utf-8')
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decode failed for {document_url}, trying ISO-8859-1")
                return doc_response.content.decode('iso-8859-1') # Common fallback

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching document {document_url}: {e}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error getting document content for {accession_number}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    # fetch_filings method remains largely the same, but relies on the updated get_submissions and get_filing_document
    def fetch_filings(self, ticker: str, filing_type: str, date_range: Optional[tuple] = None, 
                     quarters: int = None, max_search: int = None,
                     cutoff_date: Optional[datetime] = None) -> List[Dict]:
        """Fetch filings for a given ticker, filing type, and optional date range.
        
        Args:
            ticker: The stock ticker symbol
            filing_type: Type of filing to fetch ("10-K", "10-Q", "8-K", or "ALL")
            date_range: Optional tuple of (start_date, end_date)
            quarters: Number of quarterly/annual reports to find (10-Q + 10-K count)
            max_search: Maximum number of iterations before terminating early
            cutoff_date: Optional datetime to use as cutoff for data fetching
        """
        cik = self.get_cik(ticker)
        if not cik:
            logger.error(f"Could not retrieve CIK for {ticker}. Aborting filing fetch.")
            return []

        # Get relevant submission metadata based on 'quarters' param for 10-K/Q
        # get_submissions handles the logic of combining 10-K/Q up to 'quarters' and all relevant 8-K
        submissions_metadata = self.get_submissions(cik, quarters=quarters, cutoff_date=cutoff_date)
        
        if not submissions_metadata:
            logger.warning(f"No submission metadata found for CIK {cik} with given criteria.")
            return []

        logger.info(f"Retrieved {len(submissions_metadata)} submission metadata entries for CIK {cik}. Now fetching documents...")
        
        # Filter submissions by the requested filing_type ('ALL' means no filtering here)
        if filing_type != "ALL":
            filtered_metadata = [s for s in submissions_metadata if s["form_type"] == filing_type]
            logger.info(f"Filtered down to {len(filtered_metadata)} submissions matching type {filing_type}.")
        else:
            filtered_metadata = submissions_metadata # Use all if 'ALL' is requested
            logger.info(f"Processing all {len(filtered_metadata)} submissions as filing_type is 'ALL'.")

        if not filtered_metadata:
             logger.warning(f"No submissions remaining after filtering for type {filing_type}.")
             return []
             
        # Process submissions to fetch documents (consider using ThreadPoolExecutor for speed)
        filings_with_docs = []
        # Using more workers can speed this up significantly, but respects rate limits set earlier
        # Adjust max_workers based on system resources and observed performance/rate limits
        max_workers = 5 
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create futures for each document fetch
            future_to_submission = {
                executor.submit(self.get_filing_document, cik, submission["accession_number"]):
                submission for submission in filtered_metadata
            }
            
            # Process completed futures with progress tracking
            completed = 0
            total = len(future_to_submission)
            logger.info(f"Starting document fetch for {total} filings with {max_workers} workers...")
            for future in concurrent.futures.as_completed(future_to_submission):
                submission_meta = future_to_submission[future]
                try:
                    document_content = future.result()
                    completed += 1
                    if document_content: # Only add if document was fetched successfully
                        filings_with_docs.append({
                            "accession_number": submission_meta["accession_number"],
                            "filing_date": submission_meta["filing_date"],
                            "form_type": submission_meta["form_type"],
                            "document": document_content
                        })
                        # Log progress periodically
                        if completed % 5 == 0 or completed == total:
                           logger.info(f"✓ Fetched document {completed}/{total} ({((completed/total)*100):.1f}% complete)")
                    else:
                        logger.warning(f"Document fetch failed or returned empty for {submission_meta['accession_number']}")
                        # Log progress even on failure
                        if completed % 5 == 0 or completed == total:
                           logger.info(f"(Failed) Processed {completed}/{total} ({((completed/total)*100):.1f}% complete)")
                            
                except Exception as exc:
                    completed += 1
                    logger.error(f"Exception fetching document for {submission_meta['accession_number']}: {exc}")
                    # Log progress even on exception
                    if completed % 5 == 0 or completed == total:
                       logger.info(f"(Exception) Processed {completed}/{total} ({((completed/total)*100):.1f}% complete)")
        
        logger.info(f"✓ Completed document fetching. Successfully retrieved {len(filings_with_docs)} documents out of {len(filtered_metadata)} requests.")
        
        # Sort final list by date before returning (optional, get_submissions already sorted metadata)
        filings_with_docs.sort(key=lambda x: x["filing_date"], reverse=True)
        
        return filings_with_docs 