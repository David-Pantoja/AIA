from sec_edgar_fetcher import fetch_filings
from sec_edgar_fetcher.fetcher import SECFetcher
from sec_edgar_fetcher.parser import parse_filings

def print_filing_results(filing):
    """Print filing results in a formatted way."""
    print(f"\nFiling Type: {filing['form_type']}")
    print(f"Accession Number: {filing['accession_number']}")
    print(f"Filing Date: {filing['filing_date']}")
    print("-" * 40)
    
    financials = filing['financials']
    
    if filing['form_type'] in ["10-K", "10-Q"]:
        print(f"Revenue: {financials.get('Revenue', 'N/A')}")
        print(f"Net Income: {financials.get('Net Income', 'N/A')}")
        print(f"EPS: {financials.get('EPS', 'N/A')}")
        print(f"Cash: {financials.get('Cash', 'N/A')}")
        print(f"Assets: {financials.get('Assets', 'N/A')}")
        print(f"Liabilities: {financials.get('Liabilities', 'N/A')}")
        print("\nManagement Discussion & Analysis:")
        print(f"{financials.get('Management Discussion & Analysis summary', 'N/A')}")
        print("\nRisk Factors:")
        print(f"{financials.get('Risk Factors summary', 'N/A')}")
    elif filing['form_type'] == "8-K":
        print(f"Event Type: {financials.get('Event Type', 'N/A')}")
        print(f"Event Date: {financials.get('Event Date', 'N/A')}")
        print("\nEvent Description:")
        print(f"{financials.get('Event Description', 'N/A')}")
        print("\nFinancial Impact:")
        print(f"{financials.get('Financial Impact', 'N/A')}")
        print("\nKey Takeaways:")
        print(f"{financials.get('Key Takeaways', 'N/A')}")
    else:
        print("\nKey Information:")
        print(f"{financials.get('Key Information', 'N/A')}")
        print("\nFinancial Impact:")
        print(f"{financials.get('Financial Impact', 'N/A')}")
        print("\nImportant Details:")
        print(f"{financials.get('Important Details', 'N/A')}")
    print("=" * 80)

# Example usage
ticker = "AAPL"  # Replace with your desired ticker
filing_types = ["10-K", "10-Q", "8-K"]  # List of filing types to fetch
date_range = ("2023-01-01", "2023-12-31")  # Optional date range
max_files_to_check = 50  # Configure how many files to check

# Create a single fetcher instance
fetcher = SECFetcher()
cik = fetcher.get_cik(ticker)
if not cik:
    print(f"Could not find CIK for ticker {ticker}")
    exit(1)

# Get all submissions once
submissions = fetcher.get_submissions(cik, max_items=max_files_to_check)
print(f"\nFound {len(submissions)} total submissions")
print(f"Checked up to {max_files_to_check} most recent files")

# Process all filings at once
all_filings = []
for submission in submissions:
    accession_number = submission["accession_number"]
    if accession_number:
        document = fetcher.get_filing_document(cik, accession_number)
        all_filings.append({
            "accession_number": accession_number,
            "filing_date": submission["filing_date"],
            "form_type": submission["form_type"],
            "document": document
        })

# Parse all filings
parsed_filings = parse_filings(all_filings)

# Print results by filing type
for filing_type in filing_types:
    print(f"\n{filing_type} Filing Analysis Results:")
    print("=" * 80)
    type_filings = [f for f in parsed_filings if f["form_type"] == filing_type]
    for filing in type_filings:
        print_filing_results(filing)