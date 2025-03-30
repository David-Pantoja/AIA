from sec_edgar_fetcher import fetch_filings

# Example usage
ticker = "AAPL"  # Replace with your desired ticker
filing_type = "10-Q"  # Replace with your desired filing type
date_range = ("2023-01-01", "2023-12-31")  # Optional date range

# Fetch and parse filings
filings = fetch_filings(ticker, filing_type, date_range)

# Print the results in a cleaner format
print("\nFiling Analysis Results:")
print("=" * 80)
for filing in filings:
    print(f"\nAccession Number: {filing['accession_number']}")
    print("-" * 40)
    financials = filing['financials']
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
    print("=" * 80)