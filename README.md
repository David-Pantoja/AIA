# Portfolio Analysis Tool

A comprehensive portfolio analysis system that provides investment insights, conflict detection, and rebalancing recommendations using financial data from multiple sources and AI-powered analysis.

## Architecture Overview

The system follows an agentic architecture where specialized components work together:

```
Portfolio Analyzer
├── Insight Generator
│   ├── SEC Data
│   └── yfinance Data
└── Conflict Generator
    └── yfinance Company Info
```

- **Portfolio Analyzer**: Core component that orchestrates analysis, aggregating insights and identifying conflicts
- **Insight Generator**: Analyzes individual securities using data from SEC filings and Yahoo Finance
- **Conflict Generator**: Detects conflicting positions within a portfolio

## Features

- **Portfolio Insights**: Generate detailed insights for each position in your portfolio
- **Conflict Detection**: Identify conflicting positions that may work against each other
- **AI-Powered Recommendations**: Get rebalancing recommendations based on latest financial data
- **Risk Assessment**: Evaluate your portfolio's risk level and characteristics
- **Diversification Analysis**: Receive suggestions to improve portfolio diversification

## Components

### API Server

The Flask API server (`portfolio_api.py`) handles requests to analyze portfolios, providing a RESTful interface to the analysis engine.

### Web Application

A React-based frontend for submitting portfolios for analysis and viewing results with detailed visualizations.

### Securities Insight

The securities insight module evaluates individual stocks with:

- Financial data retrieval from SEC filings and market sources
- AI-powered analysis of company performance and outlook
- Price target generation and recommendation scoring

### Conflict Evaluator

Identifies potential conflicts in portfolio positions:

- Sector contradictions
- Directional conflicts
- Factor exposure conflicts
- Other investment strategy contradictions

## Installation

### Prerequisites

- Python 3.7+
- Node.js 14+
- OpenAI API key

### Backend Setup

1. Clone this repository:

   ```
   git clone https://github.com/David-Pantoja/AIA.git
   cd portfolio-analysis-tool
   ```

2. Create and activate a virtual environment:

   ```
   # Create a virtual environment
   python -m venv venv

   # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate

   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with:

   ```
   OPENAI_API_KEY=your-openai-api-key
   email=your-email@example.com
   first_name=YourFirstName
   last_name=YourLastName
   ```

### Frontend Setup

1. Navigate to the webapp directory:

   ```
   cd portfolio-webapp
   ```

2. Install dependencies:
   ```
   npm install
   ```

## Running the System

### Backend API

Start the API server:

```
python portfolio_api.py
```

The server will run on `http://localhost:8000`.

### Frontend Application

From the `portfolio-webapp` directory:

```
npm start
```

The application will be available at `http://localhost:3000`.

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

### Web Application

1. Open the web application at `http://localhost:3000`
2. Enter portfolio details and positions
3. Configure analysis parameters
4. Click "Analyze Portfolio"
5. View results including insights, conflicts, and recommendations

### Portfolio JSON Format

Create a JSON file with your portfolio information:

```json
{
  "name": "My Portfolio",
  "owner": "User",
  "date": "current",
  "config": {
    "quarters": 4,
    "max_search": 200,
    "use_SEC": true,
    "use_yfinance": true
  },
  "positions": [
    {
      "ticker": "AAPL",
      "shares": 100,
      "cost_basis": 150.75
    },
    {
      "ticker": "CRWD",
      "shares": 50,
      "cost_basis": 225.4
    },
    {
      "ticker": "F",
      "shares": 200,
      "cost_basis": 8
    },
    {
      "ticker": "TSLA",
      "shares": 200,
      "cost_basis": 225.4
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

## API Documentation

The API serves several endpoints:

- `POST /api/analyze`: Submit a portfolio for analysis
- `GET /api/health`: Health check endpoint
- `GET /api/documentation`: Get API documentation

Example request to `/api/analyze`:

```json
{
  "name": "My Portfolio",
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
  ],
  "config": {
    "use_yfinance": true,
    "use_SEC": true,
    "quarters": 4,
    "max_search": 200
  }
}
```

## Evaluation Tools

### Securities Insight Evaluation

Run the securities insight evaluator to assess model performance:

```
python insight_eval.py
```

This tool compares the predictions of different models against actual market performance and generates reports on accuracy and reliability.

### Conflict Detection Evaluation

Run the conflict evaluator:

```
python conflict_evaluator.py
```

This tool tests the conflict detection system against known test cases to measure its accuracy in identifying portfolio contradictions.

## Example Evaluation Results

Example evaluation results can be found in the following files:

### Securities Insight Evaluation

File: `insight_eval_dated_info.txt`

This file contains results from evaluating the accuracy of 1-month price target predictions. The evaluation compares:

- Regular model (using SEC filings and Yahoo Finance data)
- Blind model (without external data sources)

Summary metrics include:

- Target range accuracy (25.2% for regular model vs 5.6% for blind model)
- Price target precision (89 wins for regular model vs 16 for blind model)
- Average error from target midpoint (10.05% for regular model vs 47.71% for blind model)

### Conflict Detection Evaluation

File: `conflict_eval_output_20250404_001205.json`

This JSON file contains evaluation results of the conflict detection system against 14 test cases. The evaluation compares:

- Full analyzer (using company info from Yahoo Finance)
- Ticker-only analyzer (using only symbol names)

Results show:

- Full analyzer accuracy: 100%
- Ticker-only analyzer accuracy: 85.7%
- Detailed breakdown of detected conflicts by portfolio

## Configuration Options

The analysis can be customized with various options:

- `quarters`: Number of quarters of financial data to analyze (default: 4)
- `max_search`: Maximum number of search results to consider (default: 200)
- `use_SEC`: Whether to include SEC filing data in analysis (default: true)
- `use_yfinance`: Whether to include Yahoo Finance data in analysis (default: true)

## Dependencies

- Python 3.7+
- OpenAI API
- yfinance
- pandas
- requests
- dotenv
- Flask
- React

## License

This project is licensed under the MIT License - see the LICENSE file for details.
