
import yfinance as yf

#could fail for some tickers (mostly ETFs)
msft = yf.Ticker("TLT")

company_name = msft.info['longName']
company_summary = msft.info['longBusinessSummary']
company_legal_type = msft.info['legalType']

#Output = 'Microsoft Corporation'

print(msft.info)