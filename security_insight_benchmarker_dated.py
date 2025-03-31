import os
import json
from datetime import datetime, timedelta
from insight_generator import InsightGenerator
from typing import Dict, List, Tuple
import pandas as pd
import yfinance as yf
import numpy as np
import warnings
import scipy  # needed

# Configuration
USE_SEC = True           # Get and analyze SEC filings (looks backward from cutoff date for quarterly reports)
USE_YFINANCE = True      # Get and analyze yfinance data
USE_TOTAL_RETURN = True  # Include dividends in performance calculation
CUTOFF_DATE = "2025-03-01"  # Anchor date: SEC data looks backward from here
QUARTERS = 4            # Target number of quarterly reports to fetch BEFORE the cutoff date
MAX_SEARCH = 100         # Max filings to search for quarterly reports

# List of tickers to benchmark
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",  # Tech - Large Cap
    "NVDA", "AMD", "INTC", "TSM", "MU",  # Semiconductors
    "CRM", "ADBE", "ORCL", "IBM", "SAP",  # Enterprise Software
    "CSCO", "AVGO", "QCOM", "TXN", "NXPI",  # Networking/Hardware
    "PYPL", "SQ", "ADYEY", "V", "MA",  # Fintech
    
    "JPM", "BAC", "GS", "MS", "WFC",  # Banking
    "BLK", "BX", "KKR", "C", "AXP",  # Financial Services
    "PGR", "ALL", "TRV", "CB", "MET",  # Insurance
    
    "JNJ", "PFE", "UNH", "MRK", "ABBV",  # Healthcare/Pharma
    "MDT", "TMO", "DHR", "ABT", "ISRG",  # Medical Devices/Equipment
    
    "XOM", "CVX", "COP", "SLB", "EOG",  # Oil & Gas
    "NEE", "DUK", "SO", "AEP", "PCG",  # Utilities
    
    "PG", "KO", "WMT", "MCD", "DIS",  # Consumer Staples/Entertainment
    "NKE", "SBUX", "LULU", "TGT", "HD",  # Retail
    "PEP", "KHC", "GIS", "K", "CAG",  # Food & Beverage
    
    "TSLA", "GM", "F", "TM", "VWAGY",  # Auto
    "BA", "LMT", "RTX", "GE", "HON",  # Aerospace/Industrial
    
    "NFLX", "CMCSA", "CHTR", "ROKU", "SPOT",  # Media/Streaming
    "T", "VZ", "TMUS", "LBRDK", "DISH",  # Telecom
    
    "AMGN", "GILD", "BIIB", "MRNA", "REGN",  # Biotech
    "ABNB", "UBER", "LYFT", "DASH", "BKNG",  # Travel/Mobility
    
    "TWLO", "NET", "ZS", "OKTA", "CRWD",  # Cloud Security
    "NOW", "TEAM", "ZM", "DOCU", "WDAY"   # SaaS
]

# Quiet config output
# Print only essential info
print(f"==== BENCHMARKING CONFIG ====")
print(f"Cutoff date: {CUTOFF_DATE}")
print(f"SEC reports: {QUARTERS} quarters before cutoff")
print(f"Tickers: {', '.join(TICKERS)}")
print(f"==============================")

def get_stock_performance(ticker: str, start_date: str, end_date: str) -> Dict:
    """
    Get stock performance data using yfinance with enhanced features.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        dict: Price data and performance metrics
    """
    try:
        # Create Ticker object to access more methods
        ticker_obj = yf.Ticker(ticker)
        
        # Get historical data with more control over parameters
        stock_data = ticker_obj.history(
            start=start_date,
            end=end_date,
            interval="1d",        # Daily data
            auto_adjust=True,     # Auto-adjust OHLC
            actions=True,         # Include dividends and splits
            back_adjust=True,     # Back-adjust data to mimic true historical prices
            repair=True,          # Detect and repair currency unit mixups
            rounding=False,       # Keep precision as suggested by Yahoo
            timeout=20            # Longer timeout for reliability
        )
        
        if stock_data.empty:
            return {
                'success': False,
                'message': f"No data available for {ticker} in the specified date range"
            }
        
        # Get dividends in the period (if any)
        dividends = ticker_obj.dividends
        # Filter dividends to our date range
        if not dividends.empty:
            dividends = dividends[
                (dividends.index >= pd.Timestamp(start_date)) & 
                (dividends.index <= pd.Timestamp(end_date))
            ]
            total_dividends = dividends.sum() if not dividends.empty else 0
        else:
            total_dividends = 0
            
        # Get splits in the period (if any)
        splits = ticker_obj.splits
        has_splits = False
        if not splits.empty:
            splits_in_range = splits[
                (splits.index >= pd.Timestamp(start_date)) & 
                (splits.index <= pd.Timestamp(end_date))
            ]
            has_splits = not splits_in_range.empty
        
        # Calculate performance metrics
        start_price = stock_data['Close'].iloc[0]
        end_price = stock_data['Close'].iloc[-1]
        price_change = end_price - start_price
        
        # Include dividends in total return calculation
        total_return = price_change + total_dividends
        percent_change = (price_change / start_price) * 100
        total_return_percent = (total_return / start_price) * 100
        
        # Create a performance score from -1 to 1 based on percent change
        performance_score = np.clip(percent_change / 50, -1, 1)
        total_return_score = np.clip(total_return_percent / 50, -1, 1)
        
        # Get volatility (standard deviation of daily returns)
        daily_returns = stock_data['Close'].pct_change().dropna()
        volatility = daily_returns.std() * 100  # as percentage
        
        # Get max drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns/running_max - 1) * 100
        max_drawdown = drawdown.min()
        
        # Calculate simple moving averages without scipy
        sma_20 = stock_data['Close'].rolling(window=20).mean().iloc[-1]
        sma_50 = stock_data['Close'].rolling(window=50).mean().iloc[-1]
        
        # Calculate Bollinger Bands without scipy
        std_20 = stock_data['Close'].rolling(window=20).std().iloc[-1]
        bb_upper = sma_20 + (2 * std_20)
        bb_lower = sma_20 - (2 * std_20)
        
        # Calculate RSI without scipy
        delta = stock_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Calculate MACD without scipy
        exp1 = stock_data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = stock_data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_value = macd.iloc[-1]
        signal_value = signal.iloc[-1]
        
        return {
            'success': True,
            'start_date': stock_data.index[0].strftime('%Y-%m-%d'),
            'end_date': stock_data.index[-1].strftime('%Y-%m-%d'),
            'start_price': start_price,
            'end_price': end_price,
            'price_change': price_change,
            'percent_change': percent_change,
            'total_dividends': total_dividends,
            'total_return': total_return,
            'total_return_percent': total_return_percent,
            'has_splits': has_splits,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'performance_score': performance_score,
            'total_return_score': total_return_score,
            'data_points': len(stock_data),
            'technical_indicators': {
                'sma_20': sma_20,
                'sma_50': sma_50,
                'bollinger_bands': {
                    'upper': bb_upper,
                    'lower': bb_lower
                },
                'rsi': rsi,
                'macd': {
                    'macd': macd_value,
                    'signal': signal_value
                }
            }
        }
    except Exception as e:
        return {
            'success': False,
            'message': f"Error fetching data for {ticker}: {str(e)}"
        }

def validate_probability_distribution(probs: Dict[str, float]) -> bool:
    """Validate that probabilities sum to 1.0 within a small tolerance."""
    total = sum(probs.values())
    return abs(total - 1.0) < 0.0001

def calculate_mse(yf_probs: Dict[str, float], model_probs: Dict[str, float]) -> float:
    """Calculate Mean Square Error between two probability distributions."""
    if not validate_probability_distribution(yf_probs) or not validate_probability_distribution(model_probs):
        return float('inf')
    
    mse = 0.0
    for action in ["strongBuy", "buy", "hold", "sell", "strongSell"]:
        yf_prob = yf_probs.get(action, 0)
        model_prob = model_probs.get(action, 0)
        mse += (yf_prob - model_prob) ** 2
    return mse / 5  # Divide by number of actions

def get_stock_price_on_date(ticker: str, date_str: str) -> Dict:
    """
    Get stock price on a specific date using yfinance.
    If the date is in the future, try to get the most recent available price.
    
    Args:
        ticker (str): Stock ticker symbol
        date_str (str): Date in 'YYYY-MM-DD' format
        
    Returns:
        dict: Price data for the specified date
    """
    try:
        # Check if date is in the future
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        today = datetime.now().replace(tzinfo=None)  # Ensure today has no timezone info
        
        if target_date > today:
            # For future dates, use the most recent data available
            # Get historical data for the last 7 days
            start_date = (today - timedelta(days=7)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
        else:
            # For past dates, get a window around the target date
            start_date = (target_date - timedelta(days=5)).strftime('%Y-%m-%d')
            end_date = (target_date + timedelta(days=5)).strftime('%Y-%m-%d')
        
        # Create Ticker object
        ticker_obj = yf.Ticker(ticker)
        
        # Get historical data
        stock_data = ticker_obj.history(
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True
        )
        
        if stock_data.empty:
            return {
                'success': False,
                'message': f"No data available for {ticker} around {date_str}"
            }
        
        # If date is in the future, use the most recent price
        if target_date > today:
            # Get the most recent date in the data
            closest_date = stock_data.index[-1]  # Last entry
            # Convert Timestamp to naive datetime if it has timezone info
            if closest_date.tzinfo is not None:
                closest_date = closest_date.replace(tzinfo=None)
            price = stock_data.loc[closest_date, 'Close']
            closest_date_str = closest_date.strftime('%Y-%m-%d')
            
            return {
                'success': True,
                'ticker': ticker,
                'date': closest_date_str,
                'price': price,
                'is_exact_date': False,
                'note': f"Future date. Using most recent price as of {closest_date_str}"
            }
        else:
            # Find the closest available date (using pandas date logic)
            target_date_pd = pd.Timestamp(date_str)
            # Make target_date_pd timezone naive if stock_data.index has timezone info
            if not stock_data.empty and stock_data.index[0].tzinfo is not None:
                target_date_pd = target_date_pd.tz_localize(stock_data.index[0].tzinfo)
            
            closest_date = min(stock_data.index, key=lambda x: abs(x - target_date_pd))
            closest_date_str = closest_date.strftime('%Y-%m-%d')
            
            # Get price on the closest date
            price = stock_data.loc[closest_date, 'Close']
            
            return {
                'success': True,
                'ticker': ticker,
                'date': closest_date_str,
                'price': price,
                'is_exact_date': closest_date_str == date_str
            }
    except Exception as e:
        return {
            'success': False,
            'message': f"Error fetching price for {ticker} on {date_str}: {str(e)}"
        }

def process_ticker(ticker: str) -> Dict:
    """Process a single ticker and return results including standard and blind analysis."""
    cutoff_date_str = CUTOFF_DATE
    cutoff_datetime = datetime.strptime(cutoff_date_str, '%Y-%m-%d')
    
    print(f"\n==== INSIGHT ENGINE INPUT FOR {ticker} ====")
    
    # Fetch stock price on cutoff date
    price_data = get_stock_price_on_date(ticker, cutoff_date_str)
    if price_data.get('success'):
        price = price_data.get('price')
        actual_date = price_data.get('date')
        note = ""
        if not price_data.get('is_exact_date'):
            note = price_data.get('note', f"(closest trading day to {cutoff_date_str})")
        
        print(f"PRICE: ${price:.2f} on {actual_date} {note}")
    else:
        print(f"PRICE ERROR: {price_data.get('message', 'Unknown error fetching price')}")
    
    # Initialize the insight generator with datetime
    generator = InsightGenerator(
        os.getenv("OPENAI_API_KEY"),
        use_yfinance=USE_YFINANCE,
        use_SEC=USE_SEC,
        cutoff_date=cutoff_datetime  # Pass datetime object
    )
    
    # --- Standard Insight Generation ---
    print(f"CUTOFF DATE: {cutoff_date_str}")
    insight = generator.generate_insight(
        ticker,
        quarters=QUARTERS,
        max_search=MAX_SEARCH,
        cutoff_date=cutoff_datetime  # Pass datetime object
    )
    
    if not insight:
        print(f"ERROR: Failed to generate standard insights")
        return None # Return None if standard insight fails
    
    # Count quarterly reports if SEC data was used in standard insight
    if USE_SEC and "filings" in insight and insight["filings"]:
        quarterly_count = sum(
            1 for f in insight.get("filings", [])
            if f.get("form_type") in ["10-Q", "10-K"]
        )
        print(f"SEC DATA: {quarterly_count} quarterly reports")
        
        # Print filing dates for verification
        filing_dates = [
            f"{f.get('form_type')} {f.get('filing_date')}" 
            for f in insight.get("filings", [])
            if f.get("form_type") in ["10-Q", "10-K"]
        ]
        print(f"SEC FILINGS: {', '.join(filing_dates)}")
    
    # Extract blind probabilities
    blind_model_probs = None
    if insight and "recommendation" in insight:
        recommendation = insight["recommendation"]
        model_probs = {
            "strongBuy": recommendation.get("buy_probability", 0) * 0.6,
            "buy": recommendation.get("buy_probability", 0) * 0.4,
            "hold": recommendation.get("hold_probability", 0),
            "sell": recommendation.get("sell_probability", 0) * 0.4,
            "strongSell": recommendation.get("sell_probability", 0) * 0.6
        }
        print(f"MODEL PROBABILITIES: {json.dumps(model_probs, indent=2)}")
        
        # Print price target information
        price_target = recommendation.get("price_target", {})
        price_target_1m = recommendation.get("price_target_1m", {})
        
        if price_target:
            price_target_timeframe = recommendation.get("price_target_timeframe", "12 months")
            print(f"PRICE TARGET ({price_target_timeframe}):")
            print(f"  Low: ${price_target.get('low', 0.0):.2f}")
            print(f"  Mid: ${price_target.get('mid', 0.0):.2f}")
            print(f"  High: ${price_target.get('high', 0.0):.2f}")
            
        if price_target_1m:
            print(f"PRICE TARGET (1 month):")
            print(f"  Low: ${price_target_1m.get('low', 0.0):.2f}")
            print(f"  Mid: ${price_target_1m.get('mid', 0.0):.2f}")
            print(f"  High: ${price_target_1m.get('high', 0.0):.2f}")
    
    print(f"==== END OF INPUT FOR {ticker} ====")
    
    # --- Run Blind Analysis in Background ---
    # We don't print blind analysis inputs
    blind_insight = generator.generate_insight(
        ticker,
        quarters=0,  # No quarters needed for blind
        max_search=0, # No search needed for blind
        use_SEC_override=False,
        use_yfinance_override=False
    )

    if not blind_insight:
        blind_model_probs = None
        blind_mse = None
    else:
        # Extract blind probabilities
        blind_recommendation = blind_insight.get("recommendation", {})
        blind_model_probs = {
            "strongBuy": blind_recommendation.get("buy_probability", 0) * 0.6,
            "buy": blind_recommendation.get("buy_probability", 0) * 0.4,
            "hold": blind_recommendation.get("hold_probability", 0),
            "sell": blind_recommendation.get("sell_probability", 0) * 0.4,
            "strongSell": blind_recommendation.get("sell_probability", 0) * 0.6
        }
        # Normalize blind probabilities after apportionment
        total_blind_prob = sum(blind_model_probs.values())
        if total_blind_prob > 0:
             blind_model_probs = {k: v / total_blind_prob for k, v in blind_model_probs.items()}
        else:
             blind_model_probs = {k: 0.0 for k in blind_model_probs}

    # Initialize result dict
    result = {
        "ticker": ticker,
        "model_probabilities": None,
        "mse": None,
        "yfinance_probabilities": None,
        "quarterly_reports_found": 0,
        "blind_model_probabilities": blind_model_probs,
        "blind_mse": None,
        "cutoff_date": cutoff_date_str,
        "price_at_cutoff": price_data.get('price') if price_data.get('success', False) else None,
        "price_date": price_data.get('date') if price_data.get('success', False) else None,
        "price_note": price_data.get('note', None),
        "price_target": None,
        "price_target_1m": None,
        "blind_price_target": None,
        "blind_price_target_1m": None
    }
    
    # --- Process Standard Insight Results ---
    # Count quarterly reports if SEC data was used in standard insight
    if USE_SEC and "filings" in insight and insight["filings"]:
        result["quarterly_reports_found"] = sum(
            1 for f in insight.get("filings", [])
            if f.get("form_type") in ["10-Q", "10-K"]
        )
    
    # Save standard insight probabilities
    if "recommendation" in insight:
        recommendation = insight["recommendation"]
        result["model_probabilities"] = {
            "strongBuy": recommendation.get("buy_probability", 0) * 0.6,
            "buy": recommendation.get("buy_probability", 0) * 0.4,
            "hold": recommendation.get("hold_probability", 0),
            "sell": recommendation.get("sell_probability", 0) * 0.4,
            "strongSell": recommendation.get("sell_probability", 0) * 0.6
        }
        
        # Save price targets
        result["price_target"] = recommendation.get("price_target")
        result["price_target_1m"] = recommendation.get("price_target_1m")
    
    # Get yfinance comparison if available from standard insight
    if USE_YFINANCE and "yfinance_comparison" in insight:
        yf_comparison = insight["yfinance_comparison"]
        yf_probs = yf_comparison.get("yfinance_probabilities")
        model_probs = yf_comparison.get("model_probabilities")

        if yf_probs:
             result["yfinance_probabilities"] = yf_probs
             # Calculate standard MSE
             if model_probs:
                 result["model_probabilities"] = model_probs
                 result["mse"] = calculate_mse(yf_probs, model_probs)
             
             # Calculate blind MSE if blind probs exist
             if blind_model_probs:
                 result["blind_mse"] = calculate_mse(yf_probs, blind_model_probs)
    
    # Save blind insight price targets if available
    if blind_insight and "recommendation" in blind_insight:
        blind_recommendation = blind_insight.get("recommendation", {})
        result["blind_price_target"] = blind_recommendation.get("price_target")
        result["blind_price_target_1m"] = blind_recommendation.get("price_target_1m")
    
    return result

def save_results(results: List[Dict], filename: str) -> None:
    """Save benchmark results to a CSV file."""
    import pandas as pd
    
    # Convert results to a pandas DataFrame for easier CSV handling
    df = pd.DataFrame(results)
    
    # Convert JSON columns to strings for CSV compatibility
    for col in ['model_probabilities', 'blind_model_probabilities', 'yfinance_probabilities', 
                'price_data', 'price_target', 'price_target_1m', 'blind_price_target', 
                'blind_price_target_1m']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(x) if x is not None else None)
    
    # Add additional columns for easier analysis of price targets
    def extract_target_value(json_str, key):
        if not json_str or json_str == 'null':
            return None
        try:
            data = json.loads(json_str)
            return data.get(key) if isinstance(data, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None
            
    # Extract 12-month price target values
    df['price_target_low'] = df['price_target'].apply(lambda x: extract_target_value(x, 'low'))
    df['price_target_mid'] = df['price_target'].apply(lambda x: extract_target_value(x, 'mid'))
    df['price_target_high'] = df['price_target'].apply(lambda x: extract_target_value(x, 'high'))
    
    # Extract 1-month price target values
    df['price_target_1m_low'] = df['price_target_1m'].apply(lambda x: extract_target_value(x, 'low'))
    df['price_target_1m_mid'] = df['price_target_1m'].apply(lambda x: extract_target_value(x, 'mid'))
    df['price_target_1m_high'] = df['price_target_1m'].apply(lambda x: extract_target_value(x, 'high'))
    
    # Extract blind price target values
    df['blind_price_target_low'] = df['blind_price_target'].apply(lambda x: extract_target_value(x, 'low'))
    df['blind_price_target_mid'] = df['blind_price_target'].apply(lambda x: extract_target_value(x, 'mid'))
    df['blind_price_target_high'] = df['blind_price_target'].apply(lambda x: extract_target_value(x, 'high'))
    
    # Extract blind 1-month price target values
    df['blind_price_target_1m_low'] = df['blind_price_target_1m'].apply(lambda x: extract_target_value(x, 'low'))
    df['blind_price_target_1m_mid'] = df['blind_price_target_1m'].apply(lambda x: extract_target_value(x, 'mid'))
    df['blind_price_target_1m_high'] = df['blind_price_target_1m'].apply(lambda x: extract_target_value(x, 'high'))
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    
def print_input_summary(results: List[Dict]) -> None:
    """Print a focused summary of what was fed to the insight engine."""
    print("\n==== INSIGHT ENGINE INPUTS SUMMARY ====")
    print(f"{'Ticker':<6} | {'Price':<10} | {'Date':<12} | {'SEC Filings':<15} | {'12m Price Target':<25} | {'1m Price Target'}")
    print("-" * 105)
    
    for result in results:
        ticker = result['ticker']
        price = result.get('price_at_cutoff', None)
        date = result.get('price_date', None)
        reports = result.get('quarterly_reports_found', 0)
        price_target = result.get('price_target', None)
        price_target_1m = result.get('price_target_1m', None)
        
        price_str = f"${price:.2f}" if isinstance(price, (int, float)) else "N/A"
        date_str = date if date else "N/A"
        reports_str = f"{reports} quarterly reports"
        
        # Format 12-month price target if available
        if price_target and isinstance(price_target, dict):
            mid_target = price_target.get('mid')
            if mid_target:
                target_str = f"${mid_target:.2f} (${price_target.get('low', 0):.2f}-${price_target.get('high', 0):.2f})"
            else:
                target_str = "N/A"
        else:
            target_str = "N/A"
            
        # Format 1-month price target if available
        if price_target_1m and isinstance(price_target_1m, dict):
            mid_target_1m = price_target_1m.get('mid')
            if mid_target_1m:
                target_1m_str = f"${mid_target_1m:.2f} (${price_target_1m.get('low', 0):.2f}-${price_target_1m.get('high', 0):.2f})"
            else:
                target_1m_str = "N/A"
        else:
            target_1m_str = "N/A"
            
        print(f"{ticker:<6} | {price_str:<10} | {date_str:<12} | {reports_str:<15} | {target_str:<25} | {target_1m_str}")
    
    print("=" * 105)

def main():
    """Run the benchmarking process."""
    print(f"\n==== BENCHMARKING ====")
    print(f"Cutoff date: {CUTOFF_DATE}")
    print(f"SEC reports: {QUARTERS} quarters before cutoff")
    print(f"Tickers: {', '.join(TICKERS)}")
    results = []
    
    # Process each ticker
    for ticker in TICKERS:
        ticker_result = process_ticker(ticker)
        if ticker_result:
            results.append(ticker_result)
    
    # Save results to CSV file
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cutoff_str = CUTOFF_DATE.replace('-', '')
        file_name = f"benchmark_results_{cutoff_str}_{timestamp}.csv"
        save_results(results, file_name)
    
    print(f"\n==== BENCHMARKING COMPLETED ====")
    
if __name__ == "__main__":
    main() 