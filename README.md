# Portfolio Analysis Tool

This tool analyzes investment portfolios to provide insights, detect conflicts, and generate rebalancing recommendations. It uses data from SEC filings, stock market data, and AI-powered analysis.

## Features

- **Portfolio Insights**: Generate detailed insights for each position in your portfolio
- **Conflict Detection**: Identify conflicting positions that may work against each other
- **AI-Powered Recommendations**: Get rebalancing recommendations based on latest financial data
- **Risk Assessment**: Evaluate your portfolio's risk level and characteristics
- **Diversification Analysis**: Receive suggestions to improve portfolio diversification

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/portfolio-analysis-tool.git
   cd portfolio-analysis-tool
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   ```
   export OPENAI_API_KEY="your-openai-api-key"
   ```
   Or create a `.env` file with:
   ```
   OPENAI_API_KEY=your-openai-api-key
   ```

## Usage

### Command Line Interface

The simplest way to use this tool is through the command-line interface:

```
python analyze_portfolio.py sample_portfolio.json
```

#### Options:

- `--cutoff-date`, `-d`: Specify a cutoff date for historical data (format: YYYY-MM-DD)
- `--output`, `-o`: Save the analysis to a JSON file
- `--summary-only`, `-s`: Show only the summary and recommendations
- `--verbose`, `-v`: Show detailed information during analysis

Example:

```
python analyze_portfolio.py sample_portfolio.json --cutoff-date 2023-07-01 --output analysis.json
```

### Portfolio JSON Format

Create a JSON file with your portfolio information:

```json
{
  "name": "My Investment Portfolio",
  "positions": [
    {
      "ticker": "AAPL",
      "shares": 100,
      "cost_basis": 150.75
    },
    {
      "ticker": "MSFT",
      "shares": 75,
      "cost_basis": 280.5
    }
  ]
}
```

Required fields for each position:

- `ticker`: Stock symbol
- `shares`: Number of shares

Optional fields:

- `cost_basis`: Purchase price per share
- `purchase_date`: Date of purchase (YYYY-MM-DD)

### Python API

You can also use the tool programmatically:

```python
from portfolio_analyzer import analyze_portfolio
from datetime import datetime

# Analyze a portfolio with today's date as cutoff
results = analyze_portfolio('sample_portfolio.json')

# Analyze with a specific cutoff date
cutoff_date = datetime(2023, 7, 1)
results = analyze_portfolio('sample_portfolio.json', cutoff_date=cutoff_date)

# Access the analysis results
summary = results.get('summary')
recommendations = results.get('rebalancing_recommendations')
conflicts = results.get('conflicts')
```

## Testing

Run the unit tests to verify the tool is working correctly:

```
python -m unittest test_portfolio_analyzer.py
```

## Dependencies

- Python 3.7+
- OpenAI API
- yfinance
- pandas
- requests
- dotenv

## License

This project is licensed under the MIT License - see the LICENSE file for details.
