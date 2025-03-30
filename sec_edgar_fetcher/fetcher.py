# SEC API interaction logic 

import requests
from typing import Dict, List, Optional
from .utils import rate_limited, retry_on_http_error
from dotenv import load_dotenv
import os

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
        print(f"Status Code: {response.status_code}")
        print(f"Response Content: {response.text[:200]}")  # Print first 200 chars
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
    def get_submissions(self, cik: str) -> List[Dict]:
        """Fetch submission metadata for a given CIK."""
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/index.json"
        print(f"Requesting URL: {url}")
        response = requests.get(url, headers=self.headers, timeout=10)
        print(f"Submissions Status Code: {response.status_code}")
        if response.status_code != 200:
            return []
        try:
            data = response.json()
            items = data.get("directory", {}).get("item", [])
            print(f"Found {len(items)} items in directory")
            
            # Sort items by name (which contains the date) in reverse order
            items.sort(key=lambda x: x.get("name", ""), reverse=True)
            
            # Only process the first 10 items (most recent filings)
            submissions = []
            for i, item in enumerate(items[:10]):
                print(f"Processing item {i+1}/10")
                if item.get("type") == "folder.gif":
                    accession_number = item.get("name")
                    if accession_number:
                        try:
                            # Accession number format: YYYYMMDD-XXXXXX-XXXXXX
                            # Extract date from the first part (YYYYMMDD)
                            date_str = accession_number.split('-')[0]
                            year = int(date_str[:4])
                            month = int(date_str[4:6])
                            day = int(date_str[6:8])
                            
                            print(f"Found filing from {year}-{month}-{day}")
                            if year >= 2023:  # Only include recent filings
                                submissions.append({
                                    "form": "10-Q",
                                    "filingDate": f"{year}-{month:02d}-{day:02d}",
                                    "accessionNumber": accession_number
                                })
                        except (ValueError, IndexError) as e:
                            print(f"Error parsing accession number {accession_number}: {e}")
                            continue
            print(f"Found {len(submissions)} valid submissions")
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
            print(f"Document index Status Code: {response.status_code}")
            data = response.json()
            items = data.get("directory", {}).get("item", [])
            print(f"Found {len(items)} items in filing directory")
            
            # Print all available files for debugging
            print("Available files in directory:")
            for item in items:
                print(f"- {item.get('name', '')} (type: {item.get('type', '')})")
            
            # Look for the main filing document (the .txt file)
            document_url = None
            for item in items:
                name = item.get("name", "").lower()
                if name.endswith('.txt') and not name.endswith('-index.txt'):
                    # Construct the full URL for the document
                    document_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number.replace('-', '')}/{name}"
                    print(f"Found matching document: {name}")
                    break
            
            if document_url:
                print(f"Found document URL: {document_url}")
                response = requests.get(document_url, headers=self.headers, timeout=10)
                print(f"Document content Status Code: {response.status_code}")
                if response.status_code == 200:
                    print(f"Document content preview: {response.text[:200]}")
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

        submissions = self.get_submissions(cik)
        filings = []
        for submission in submissions:
            if submission["form"] == filing_type:
                accession_number = submission["accessionNumber"]
                if accession_number:
                    document = self.get_filing_document(cik, accession_number)
                    filings.append({
                        "accession_number": accession_number,
                        "filing_date": submission["filingDate"],
                        "document": document
                    })
        return filings 