# SEC API interaction logic 

import requests
from typing import Dict, List, Optional
from .utils import rate_limited, retry_on_http_error
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

class SECFetcher:
    def __init__(self):
        self.base_url = "https://data.sec.gov"
        self.headers = {
            "User-Agent": "SECFetcher/1.0 ({email}) Python/3.11.5".format(email=os.getenv("email")),
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov"
        }

    @rate_limited
    @retry_on_http_error
    def get_cik(self, ticker: str) -> Optional[str]:
        """Fetch CIK for a given ticker symbol."""
        url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(url, headers=self.headers)
        # print(f"Status Code: {response.status_code}")
        # print(f"Response Content: {response.text[:200]}")  # Print first 200 chars
        if response.status_code != 200:
            return None
        try:
            data = response.json()
            for company in data.values():
                if company["ticker"] == ticker:
                    return str(company["cik_str"]).zfill(10)
        except requests.exceptions.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            return None
        return None

    @rate_limited
    @retry_on_http_error
    def get_submissions(self, cik: str, max_items: int = 50) -> List[Dict]:
        """Fetch submission metadata for a given CIK."""
        # First get the company's filing history
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/index.json"
        response = requests.get(url, headers=self.headers, timeout=10)
        if response.status_code != 200:
            return []
        try:
            data = response.json()
            items = data.get("directory", {}).get("item", [])
            print(f"Found {len(items)} items in directory")
            
            # Sort items by last-modified date in reverse order
            items.sort(key=lambda x: x.get("last-modified", ""), reverse=True)
            
            # Process more items to find more filings
            submissions = []
            current_date = datetime.now()
            
            for i, item in enumerate(items[:max_items]):
                if item.get("type") == "folder.gif":
                    accession_number = item.get("name")
                    last_modified = item.get("last-modified", "")
                    
                    if accession_number and last_modified:
                        try:
                            # Parse the last-modified date
                            date_parts = last_modified.split()[0].split('-')
                            if len(date_parts) == 3:
                                year = int(date_parts[0])
                                month = int(date_parts[1])
                                day = int(date_parts[2])
                                
                                # Skip future dates and invalid dates
                                filing_date = datetime(year, month, day)
                                
                                # For 2025 filings, only skip if the date is in the future
                                if year == 2025:
                                    if filing_date > current_date:
                                        print(f"Skipping future 2025 date: {year}-{month:02d}-{day:02d}")
                                        continue
                                # For other years, skip if before 2023
                                elif year < 2023:
                                    print(f"Skipping old date: {year}-{month:02d}-{day:02d}")
                                    continue
                                
                                # Get the form type from the filing metadata
                                form_type = None
                                form_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/index.json"
                                
                                form_response = requests.get(form_url, headers=self.headers, timeout=10)
                                
                                if form_response.status_code == 200:
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
                                        doc_response = requests.get(doc_url, headers=self.headers, timeout=10)
                                        if doc_response.status_code == 200:
                                            doc_content = doc_response.text
                                            if "CONFORMED SUBMISSION TYPE: 8-K" in doc_content:
                                                form_type = "8-K"
                                            elif "CONFORMED SUBMISSION TYPE: 10-K" in doc_content:
                                                form_type = "10-K"
                                            elif "CONFORMED SUBMISSION TYPE: 10-Q" in doc_content:
                                                form_type = "10-Q"
                                    
                                    # If no form type found in main document, try file patterns
                                    if not form_type:
                                        for form_item in form_items:
                                            name = form_item.get("name", "").lower()
                                            if any(pattern in name for pattern in ['_8k.htm', '_8k_htm.xml', '8-k', '8k']):
                                                form_type = "8-K"
                                                break
                                            elif any(pattern in name for pattern in ['_10k.htm', '_10k_htm.xml', '10-k', '10k']):
                                                form_type = "10-K"
                                                break
                                            elif any(pattern in name for pattern in ['_10q.htm', '_10q_htm.xml', '10-q', '10q']):
                                                form_type = "10-Q"
                                                break
                                    
                                    # If still no form type, try index file
                                    if not form_type:
                                        for form_item in form_items:
                                            name = form_item.get("name", "").lower()
                                            if name.endswith('-index.txt'):
                                                index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{name}"
                                                index_response = requests.get(index_url, headers=self.headers, timeout=10)
                                                if index_response.status_code == 200:
                                                    index_content = index_response.text
                                                    if "CONFORMED SUBMISSION TYPE: 8-K" in index_content:
                                                        form_type = "8-K"
                                                        break
                                                    elif "CONFORMED SUBMISSION TYPE: 10-K" in index_content:
                                                        form_type = "10-K"
                                                        break
                                                    elif "CONFORMED SUBMISSION TYPE: 10-Q" in index_content:
                                                        form_type = "10-Q"
                                                        break
                                
                                if form_type in ["10-K", "10-Q", "8-K"]:  # Only include valid form types
                                    submissions.append({
                                        "form_type": form_type,
                                        "filing_date": f"{year}-{month:02d}-{day:02d}",
                                        "accession_number": accession_number
                                    })
                        except (ValueError, IndexError) as e:
                            print(f"Error parsing data for accession number {accession_number}: {e}")
                            continue
            print(f"\nFound {len(submissions)} valid submissions")
            return submissions
        except requests.exceptions.JSONDecodeError as e:
            print(f"Submissions JSON Decode Error: {e}")
            return []

    @rate_limited
    @retry_on_http_error
    def get_filing_document(self, cik: str, accession_number: str) -> str:
        """Fetch the filing document for a given CIK and accession number."""
        print(f"Fetching document for accession number: {accession_number}")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number.replace('-', '')}/index.json"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            # print(f"Document index Status Code: {response.status_code}")
            data = response.json()
            items = data.get("directory", {}).get("item", [])
            # print(f"Found {len(items)} items in filing directory")
            
            # Print all available files for debugging
            # print("Available files in directory:")
            # for item in items:
            #     print(f"- {item.get('name', '')} (type: {item.get('type', '')})")
            
            # Look for the main filing document (the .txt file)
            document_url = None
            for item in items:
                name = item.get("name", "").lower()
                if name.endswith('.txt') and not name.endswith('-index.txt'):
                    # Construct the full URL for the document
                    document_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number.replace('-', '')}/{name}"
                    # print(f"Found matching document: {name}")
                    break
            
            if document_url:
                # print(f"Found document URL: {document_url}")
                response = requests.get(document_url, headers=self.headers, timeout=10)
                # print(f"Document content Status Code: {response.status_code}")
                if response.status_code == 200:
                    # print(f"Document content preview: {response.text[:200]}")
                    return response.text
            else:
                print("No document URL found")
            return ""
        except requests.exceptions.Timeout:
            print(f"Timeout while fetching document for {accession_number}")
            return ""
        except Exception as e:
            print(f"Error fetching document: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return ""

    def fetch_filings(self, ticker: str, filing_type: str, date_range: Optional[tuple] = None) -> List[Dict]:
        """Fetch filings for a given ticker, filing type, and optional date range."""
        cik = self.get_cik(ticker)
        if not cik:
            return []

        # Get all submissions once
        submissions = self.get_submissions(cik)
        print(f"\nFound {len(submissions)} total submissions")
        
        # Process all submissions in a single pass
        filings = []
        for submission in submissions:
            # If filing_type is "ALL" or matches the requested type, process it
            if filing_type == "ALL" or submission["form_type"] == filing_type:
                accession_number = submission["accession_number"]
                if accession_number:
                    document = self.get_filing_document(cik, accession_number)
                    filings.append({
                        "accession_number": accession_number,
                        "filing_date": submission["filing_date"],
                        "form_type": submission["form_type"],
                        "document": document
                    })
        
        print(f"\nProcessed {len(filings)} {filing_type if filing_type != 'ALL' else 'total'} filings")
        return filings 