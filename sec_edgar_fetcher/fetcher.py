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

# logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
email = os.getenv("email")
first_name = os.getenv("first_name")
last_name = os.getenv("last_name")

class SECFetcher:
    def __init__(self):
        self.base_url = "https://data.sec.gov"
        self.headers = {
            "User-Agent": f"{first_name} {last_name} {email}",
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov"
        }
        # prevent rate limiting
        self.rate_limiter = limits(calls=5, period=1)(sleep_and_retry(lambda: None))
        self.request_delay = 0.5  
        self.max_retries = 5 
        self.retry_delay = 10 
        self.consecutive_rate_limits = 0  
        self.last_request_time = 0
        self.min_request_interval = 0.5 

    #rate limiting stuff for debugging
    def _adaptive_delay(self):
        if self.consecutive_rate_limits > 0:
            delay = self.request_delay * (2 ** (self.consecutive_rate_limits - 1))
            return min(delay, 10.0)
        return self.request_delay

    #make request with retry logic
    def _make_request(self, url: str, timeout: int = 30) -> Optional[requests.Response]:
        for attempt in range(self.max_retries):
            try:
                self.rate_limiter()
                
                delay = self._adaptive_delay()
                
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < self.min_request_interval:
                    time.sleep(self.min_request_interval - time_since_last)
                
                time.sleep(delay)
                
                self.last_request_time = time.time()
                
                if 'Archives/edgar/data' in url and not url.endswith('.json'):
                    timeout = 60 
                
                logger.debug(f"Making request to: {url} with delay: {delay:.2f}s")
                response = requests.get(url, headers=self.headers, timeout=timeout)
                
                #rate limit handling
                if response.status_code == 429:  
                    self.consecutive_rate_limits += 1
                    wait_time = self.retry_delay * (2 ** attempt) 
                    logger.warning(f"Rate limit hit on attempt {attempt + 1}/{self.max_retries}, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    self.consecutive_rate_limits = 0
                
                # 404 handling
                if response.status_code == 404:
                    logger.warning(f"404 error on attempt {attempt + 1}/{self.max_retries} for URL: {url}")
                    logger.warning(f"Response content: {response.text[:500]}")  # Log first 500 chars of response
                    return None
                
                # other error handling
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
        # this URL is for the CIK lookup- host should be www.sec.gov for this specific request
        cik_lookup_url = "https://www.sec.gov/files/company_tickers.json"
        cik_lookup_headers = self.headers.copy()
        cik_lookup_headers["Host"] = "www.sec.gov"
        
        # rate limiting stuff for debugging
        max_retries = 5
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.rate_limiter()
                time.sleep(self.request_delay)
                
                response = requests.get(cik_lookup_url, headers=cik_lookup_headers, timeout=30)
                
                #rate limit handling
                if response.status_code == 429:
                    wait_time = base_delay * (2 ** attempt) 
                    logger.warning(f"Rate limit hit on attempt {attempt + 1}/{max_retries}, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                
                try:
                    data = response.json()
                    logger.info(f"Found {len(data)} companies in SEC database")
                    
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

    #get form type from document
    def _get_form_type_from_document(self, doc_content: str) -> Optional[str]:
        if "CONFORMED SUBMISSION TYPE: 8-K" in doc_content:
            return "8-K"
        elif "CONFORMED SUBMISSION TYPE: 10-K" in doc_content:
            return "10-K"
        elif "CONFORMED SUBMISSION TYPE: 10-Q" in doc_content:
            return "10-Q"
        return None

    #get form type from filename
    def _get_form_type_from_filename(self, filename: str) -> Optional[str]:
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
        try:
            if item.get("type") != "folder.gif":
                return None

            accession_number = item.get("name")
            last_modified = item.get("last-modified", "")
            
            if not accession_number or not last_modified:
                return None

            date_parts = last_modified.split()[0].split('-')
            if len(date_parts) != 3:
                return None

            year = int(date_parts[0])
            month = int(date_parts[1])
            day = int(date_parts[2])
            
            filing_date = datetime(year, month, day)
            
            if year < 2023:
                return None
            elif filing_date > (current_date + timedelta(days=30)):
                return None
            
            form_type = None
            form_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/index.json"
            
            form_response = self._make_request(form_url)
            if not form_response or form_response.status_code != 200:
                return None
            
            form_data = form_response.json()
            form_items = form_data.get("directory", {}).get("item", [])
            
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
            
            if not form_type:
                for form_item in form_items:
                    name = form_item.get("name", "")
                    form_type = self._get_form_type_from_filename(name)
                    if form_type:
                        break
            
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

            if quarters is not None and quarters > 0:
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
                    
                    if latest_quarterly_date is None or filing_dt > latest_quarterly_date:
                        latest_quarterly_date = filing_dt
                
                logger.info(f"Added {quarterly_report_count} quarterly/annual reports based on limit.")
            
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
            
            submissions.sort(key=lambda x: x["filing_date"], reverse=True)
            
            logger.info(f"Returning {len(submissions)} valid submissions after filtering and sorting.")
            return submissions
            
        except Exception as e:
            logger.error(f"Error processing submissions: {e}")
            return []

    @rate_limited
    @retry_on_http_error
    def get_filing_document(self, cik: str, accession_number: str) -> str:
        cik_formatted = str(cik).zfill(10)
        accession_nodash = accession_number.replace('-', '')
        
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_formatted}/{accession_nodash}/index.json"
        logger.info(f"Fetching filing index: {index_url}")
        
        archive_headers = self.headers.copy()
        archive_headers["Host"] = "www.sec.gov"
        
        try:
            response = requests.get(index_url, headers=archive_headers, timeout=30)
            response.raise_for_status() 
            index_data = response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching filing index {index_url}: {e}")
            return ""

        try:
            directory_items = index_data.get("directory", {}).get("item", [])
            if not directory_items:
                logger.warning(f"No items found in directory structure for index {index_url}")
                return ""
            
            # Ffind main document
            primary_doc_name = None
            potential_docs = []
            for item in directory_items:
                name = item.get("name", "").lower()
                if name.endswith(('.htm', '.html')) and 'index' not in name and 'rendering' not in name and 'summary' not in name:
                    potential_docs.append(item.get("name"))
                elif name.endswith('.txt') and 'index' not in name:
                    potential_docs.append(item.get("name"))

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

            document_url = f"https://www.sec.gov/Archives/edgar/data/{cik_formatted}/{accession_nodash}/{primary_doc_name}"
            logger.info(f"Fetching primary document: {document_url}")
            
            doc_response = requests.get(document_url, headers=archive_headers, timeout=60)
            doc_response.raise_for_status()
            
            try:
                return doc_response.content.decode('utf-8')
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decode failed for {document_url}, trying ISO-8859-1")
                return doc_response.content.decode('iso-8859-1') 

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching document {document_url}: {e}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error getting document content for {accession_number}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    def fetch_filings(self, ticker: str, filing_type: str, date_range: Optional[tuple] = None, 
                     quarters: int = None, max_search: int = None,
                     cutoff_date: Optional[datetime] = None) -> List[Dict]:
        cik = self.get_cik(ticker)
        if not cik:
            logger.error(f"Could not retrieve CIK for {ticker}. Aborting filing fetch.")
            return []

        submissions_metadata = self.get_submissions(cik, quarters=quarters, cutoff_date=cutoff_date)
        
        if not submissions_metadata:
            logger.warning(f"No submission metadata found for CIK {cik} with given criteria.")
            return []

        logger.info(f"Retrieved {len(submissions_metadata)} submission metadata entries for CIK {cik}. Now fetching documents...")
        
        if filing_type != "ALL":
            filtered_metadata = [s for s in submissions_metadata if s["form_type"] == filing_type]
            logger.info(f"Filtered down to {len(filtered_metadata)} submissions matching type {filing_type}.")
        else:
            filtered_metadata = submissions_metadata # Use all if 'ALL' is requested
            logger.info(f"Processing all {len(filtered_metadata)} submissions as filing_type is 'ALL'.")

        if not filtered_metadata:
             logger.warning(f"No submissions remaining after filtering for type {filing_type}.")
             return []
        
        filings_with_docs = []
        max_workers = 5 
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_submission = {
                executor.submit(self.get_filing_document, cik, submission["accession_number"]):
                submission for submission in filtered_metadata
            }
            
            completed = 0
            total = len(future_to_submission)
            logger.info(f"Starting document fetch for {total} filings with {max_workers} workers...")
            for future in concurrent.futures.as_completed(future_to_submission):
                submission_meta = future_to_submission[future]
                try:
                    document_content = future.result()
                    completed += 1
                    if document_content: # add if document was fetched successfully
                        filings_with_docs.append({
                            "accession_number": submission_meta["accession_number"],
                            "filing_date": submission_meta["filing_date"],
                            "form_type": submission_meta["form_type"],
                            "document": document_content
                        })
                        # log progress periodically
                        if completed % 5 == 0 or completed == total:
                           logger.info(f"✓ Fetched document {completed}/{total} ({((completed/total)*100):.1f}% complete)")
                    else:
                        logger.warning(f"Document fetch failed or returned empty for {submission_meta['accession_number']}")
                        # log progress even on failure
                        if completed % 5 == 0 or completed == total:
                           logger.info(f"(Failed) Processed {completed}/{total} ({((completed/total)*100):.1f}% complete)")
                            
                except Exception as exc:
                    completed += 1
                    logger.error(f"Exception fetching document for {submission_meta['accession_number']}: {exc}")
                    # log progress on exception
                    if completed % 5 == 0 or completed == total:
                       logger.info(f"(Exception) Processed {completed}/{total} ({((completed/total)*100):.1f}% complete)")
        
        logger.info(f"completed document fetching. Successfully retrieved {len(filings_with_docs)} documents out of {len(filtered_metadata)} requests.")
        
        # sort final list by date
        filings_with_docs.sort(key=lambda x: x["filing_date"], reverse=True)
        
        return filings_with_docs 