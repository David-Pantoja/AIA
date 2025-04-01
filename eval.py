import pandas as pd
import json
import numpy as np
from datetime import datetime
import yfinance as yf

CSVy = "benchmark_results.csv"

def calculate_score(probabilities):
    """
    Calculate prediction score using the formula:
    2*strongBuy + buy + 0*hold - sell - 2*strongSell
    
    Args:
        probabilities (dict or str): Dictionary or JSON string of probability ratings
        
    Returns:
        float: Calculated score
    """
    # Handle None, NaN, or invalid input types
    if probabilities is None or pd.isna(probabilities) or not (isinstance(probabilities, (dict, str))):
        return None
    
    # Parse JSON string if needed
    if isinstance(probabilities, str):
        try:
            probabilities = json.loads(probabilities)
        except (json.JSONDecodeError, TypeError):
            return None
    
    # Ensure we have a dictionary at this point
    if not isinstance(probabilities, dict):
        return None
    
    score = (
        2 * probabilities.get('strongBuy', 0) +
        1 * probabilities.get('buy', 0) +
        0 * probabilities.get('hold', 0) -
        1 * probabilities.get('sell', 0) -
        2 * probabilities.get('strongSell', 0)
    )
    
    return score

def get_stock_performance(ticker, start_date, end_date):
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
        # This normalizes performance to be comparable to our prediction scores
        # Map a range of -50% to +50% to a score of -1 to 1
        # Values outside this range are capped
        performance_score = np.clip(percent_change / 50, -1, 1)
        
        # Total return score (including dividends)
        total_return_score = np.clip(total_return_percent / 50, -1, 1)
        
        # Get volatility (standard deviation of daily returns)
        daily_returns = stock_data['Close'].pct_change().dropna()
        volatility = daily_returns.std() * 100  # as percentage
        
        # Get max drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns/running_max - 1) * 100
        max_drawdown = drawdown.min()
        
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
            'data_points': len(stock_data)
        }
    except Exception as e:
        return {
            'success': False,
            'message': f"Error fetching data for {ticker}: {str(e)}"
        }

def evaluate_predictions_with_prices(csv_path, start_date, end_date, use_total_return=True):
    """
    Evaluate which prediction method had better price movement against actual performance.
    
    Args:
        csv_path (str): Path to the CSV file with prediction data
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        use_total_return (bool): If True, use total return (with dividends) for comparison.
                                Otherwise use just price movement.
        
    Returns:
        dict: Results of the evaluation
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Handle missing or invalid data in probability columns
    for col in ['model_probabilities', 'blind_model_probabilities', 'yfinance_probabilities']:
        # Ensure all NaN values are properly identified
        df[col] = df[col].apply(lambda x: None if pd.isna(x) else x)
    
    # Calculate scores for each prediction method
    df['model_score'] = df['model_probabilities'].apply(calculate_score)
    df['blind_model_score'] = df['blind_model_probabilities'].apply(calculate_score)
    df['yfinance_score'] = df['yfinance_probabilities'].apply(calculate_score)
    
    # Create a results dictionary
    results = {
        'evaluation_date': datetime.now().strftime('%Y-%m-%d'),
        'start_date': start_date,
        'end_date': end_date,
        'ticker_results': [],
        'summary': {}
    }
    
    # Fetch actual price data and analyze each ticker
    for _, row in df.iterrows():
        ticker = row['ticker']
        
        # Get actual price performance
        performance = get_stock_performance(ticker, start_date, end_date)
        
        ticker_result = {
            'ticker': ticker,
            'model_score': round(row['model_score'], 4) if pd.notna(row['model_score']) else None,
            'blind_model_score': round(row['blind_model_score'], 4) if pd.notna(row['blind_model_score']) else None,
            'yfinance_score': round(row['yfinance_score'], 4) if pd.notna(row['yfinance_score']) else None,
        }
        
        # Add performance data if available
        if performance['success']:
            # Copy all performance data directly
            ticker_result['actual_performance'] = performance
            
            # Determine which performance score to use (regular or total return)
            if use_total_return and 'total_return_score' in performance:
                perf_score = performance['total_return_score']
            else:
                perf_score = performance['performance_score']
                
            # Calculate which model was closer to actual performance
            model_error = abs(row['model_score'] - perf_score) if pd.notna(row['model_score']) else float('inf')
            blind_error = abs(row['blind_model_score'] - perf_score) if pd.notna(row['blind_model_score']) else float('inf')
            
            if model_error < blind_error:
                ticker_result['closer_to_actual'] = 'Regular Model'
                ticker_result['model_error'] = round(model_error, 4)
                ticker_result['blind_error'] = round(blind_error, 4) if blind_error != float('inf') else None
            elif blind_error < model_error:
                ticker_result['closer_to_actual'] = 'Blind Model'
                ticker_result['model_error'] = round(model_error, 4) if model_error != float('inf') else None
                ticker_result['blind_error'] = round(blind_error, 4)
            elif model_error == blind_error and model_error != float('inf'):
                ticker_result['closer_to_actual'] = 'Tie'
                ticker_result['model_error'] = round(model_error, 4)
                ticker_result['blind_error'] = round(blind_error, 4)
            else:
                ticker_result['closer_to_actual'] = 'N/A'
        else:
            ticker_result['actual_performance'] = {'error': performance['message']}
            ticker_result['closer_to_actual'] = 'N/A'
        
        # Determine which model performed better (based on prediction scores only)
        if pd.notna(row['model_score']) and pd.notna(row['blind_model_score']):
            if row['model_score'] > row['blind_model_score']:
                ticker_result['better_score'] = 'Regular Model'
            elif row['blind_model_score'] > row['model_score']:
                ticker_result['better_score'] = 'Blind Model'
            else:
                ticker_result['better_score'] = 'Tie'
        else:
            ticker_result['better_score'] = 'N/A'
        
        results['ticker_results'].append(ticker_result)
    
    # Calculate summary statistics
    valid_model_scores = [r['model_score'] for r in results['ticker_results'] if r['model_score'] is not None]
    valid_blind_scores = [r['blind_model_score'] for r in results['ticker_results'] if r['blind_model_score'] is not None]
    valid_yfinance_scores = [r['yfinance_score'] for r in results['ticker_results'] if r['yfinance_score'] is not None]
    
    # Collect actual performance metrics for comparison
    performance_scores = []
    model_errors = []
    blind_errors = []
    
    # For correlation calculation
    model_perf_pairs = []
    blind_model_perf_pairs = []
    
    for r in results['ticker_results']:
        if 'actual_performance' in r and r['actual_performance'].get('success', False):
            # Get the appropriate performance score based on use_total_return
            if use_total_return and 'total_return_score' in r['actual_performance']:
                perf_score = r['actual_performance']['total_return_score']
            else:
                perf_score = r['actual_performance']['performance_score']
                
            performance_scores.append(perf_score)
            
            if 'model_error' in r and r['model_error'] is not None:
                model_errors.append(r['model_error'])
                
            if 'blind_error' in r and r['blind_error'] is not None:
                blind_errors.append(r['blind_error'])
                
            # Collect pairs for correlation calculation
            if r['model_score'] is not None:
                model_perf_pairs.append((r['model_score'], perf_score))
                
            if r['blind_model_score'] is not None:
                blind_model_perf_pairs.append((r['blind_model_score'], perf_score))
    
    # Calculate correlations if we have enough data points
    model_actual_corr = None
    blind_model_actual_corr = None
    
    if len(model_perf_pairs) >= 2:
        model_scores, perf_scores = zip(*model_perf_pairs)
        model_actual_corr = np.corrcoef(model_scores, perf_scores)[0, 1]
        
    if len(blind_model_perf_pairs) >= 2:
        blind_scores, perf_scores = zip(*blind_model_perf_pairs)
        blind_model_actual_corr = np.corrcoef(blind_scores, perf_scores)[0, 1]
    
    results['summary'] = {
        'average_scores': {
            'model': round(np.mean(valid_model_scores), 4) if valid_model_scores else None,
            'blind_model': round(np.mean(valid_blind_scores), 4) if valid_blind_scores else None,
            'yfinance': round(np.mean(valid_yfinance_scores), 4) if valid_yfinance_scores else None,
            'actual_performance': round(np.mean(performance_scores), 4) if performance_scores else None
        },
        'average_errors': {
            'model': round(np.mean(model_errors), 4) if model_errors else None,
            'blind_model': round(np.mean(blind_errors), 4) if blind_errors else None
        },
        'model_actual_correlation': round(model_actual_corr, 4) if model_actual_corr is not None else None,
        'blind_model_actual_correlation': round(blind_model_actual_corr, 4) if blind_model_actual_corr is not None else None,
        'wins': {
            'model_score': len([r for r in results['ticker_results'] if r.get('better_score') == 'Regular Model']),
            'blind_model_score': len([r for r in results['ticker_results'] if r.get('better_score') == 'Blind Model']),
            'score_ties': len([r for r in results['ticker_results'] if r.get('better_score') == 'Tie']),
            'model_accuracy': len([r for r in results['ticker_results'] if r.get('closer_to_actual') == 'Regular Model']),
            'blind_model_accuracy': len([r for r in results['ticker_results'] if r.get('closer_to_actual') == 'Blind Model']),
            'accuracy_ties': len([r for r in results['ticker_results'] if r.get('closer_to_actual') == 'Tie'])
        }
    }
    
    # Determine the overall better model based on score
    if results['summary']['average_scores']['model'] is not None and results['summary']['average_scores']['blind_model'] is not None:
        if results['summary']['average_scores']['model'] > results['summary']['average_scores']['blind_model']:
            results['summary']['overall_better_score'] = 'Regular Model'
        elif results['summary']['average_scores']['blind_model'] > results['summary']['average_scores']['model']:
            results['summary']['overall_better_score'] = 'Blind Model'
        else:
            results['summary']['overall_better_score'] = 'Tie'
    else:
        results['summary']['overall_better_score'] = 'N/A'
    
    # Determine the overall more accurate model based on errors
    if results['summary']['average_errors']['model'] is not None and results['summary']['average_errors']['blind_model'] is not None:
        if results['summary']['average_errors']['model'] < results['summary']['average_errors']['blind_model']:
            results['summary']['overall_more_accurate'] = 'Regular Model'
        elif results['summary']['average_errors']['blind_model'] < results['summary']['average_errors']['model']:
            results['summary']['overall_more_accurate'] = 'Blind Model'
        else:
            results['summary']['overall_more_accurate'] = 'Tie'
    else:
        results['summary']['overall_more_accurate'] = 'N/A'
    
    return results

def print_results_with_prices(results):
    """
    Print the evaluation results in a readable format with enhanced performance metrics.
    
    Args:
        results (dict): Results from evaluate_predictions_with_prices function
    """
    print(f"Price Movement Prediction Evaluation with Actual Price Data")
    print(f"Evaluation Date: {results['evaluation_date']}")
    print(f"Price Period: {results['start_date']} to {results['end_date']}")
    
    # Print the basic comparison table first
    print("\nBasic Comparison by Ticker:")
    print("-" * 110)
    print(f"{'Ticker':<6} | {'Regular Model':<12} | {'Blind Model':<12} | {'Actual Perf':<12} | {'Closer Model':<12} | {'R-Error':<8} | {'B-Error':<8}")
    print("-" * 110)
    
    for result in results['ticker_results']:
        model_score = f"{result['model_score']:.4f}" if result['model_score'] is not None else "N/A"
        blind_score = f"{result['blind_model_score']:.4f}" if result['blind_model_score'] is not None else "N/A"
        
        if 'actual_performance' in result and 'performance_score' in result['actual_performance']:
            perf_score = f"{result['actual_performance']['performance_score']:.4f}"
            pct_change = f"({result['actual_performance']['percent_change']:.2f}%)"
        else:
            perf_score = "N/A"
            pct_change = ""
            
        closer = result.get('closer_to_actual', 'N/A')
        model_err = f"{result.get('model_error', 'N/A')}" if result.get('model_error') is not None else "N/A"
        blind_err = f"{result.get('blind_error', 'N/A')}" if result.get('blind_error') is not None else "N/A"
        
        print(f"{result['ticker']:<6} | {model_score:<12} | {blind_score:<12} | {perf_score:<10} {pct_change:<8} | {closer:<12} | {model_err:<8} | {blind_err:<8}")
    
    # Now print detailed performance metrics for each ticker
    print("\nDetailed Price Performance by Ticker:")
    print("-" * 120)
    print(f"{'Ticker':<6} | {'Price Change':<11} | {'% Change':<9} | {'Total Return':<11} | {'TR %':<7} | {'Volatility':<10} | {'Max Drawdown':<12} | {'Divs':<7} | {'Splits':<6}")
    print("-" * 120)
    
    for result in results['ticker_results']:
        if 'actual_performance' in result and result['actual_performance'].get('success', False):
            perf = result['actual_performance']
            
            price_change = f"${perf['price_change']:.2f}" if 'price_change' in perf else "N/A"
            pct_change = f"{perf['percent_change']:.2f}%" if 'percent_change' in perf else "N/A"
            
            total_return = f"${perf['total_return']:.2f}" if 'total_return' in perf else "N/A"
            tr_pct = f"{perf['total_return_percent']:.2f}%" if 'total_return_percent' in perf else "N/A"
            
            volatility = f"{perf['volatility']:.2f}%" if 'volatility' in perf else "N/A"
            max_dd = f"{perf['max_drawdown']:.2f}%" if 'max_drawdown' in perf else "N/A"
            
            divs = f"${perf['total_dividends']:.2f}" if 'total_dividends' in perf else "N/A"
            splits = "Yes" if perf.get('has_splits', False) else "No"
            
            print(f"{result['ticker']:<6} | {price_change:<11} | {pct_change:<9} | {total_return:<11} | {tr_pct:<7} | {volatility:<10} | {max_dd:<12} | {divs:<7} | {splits:<6}")
        else:
            print(f"{result['ticker']:<6} | {'N/A':<11} | {'N/A':<9} | {'N/A':<11} | {'N/A':<7} | {'N/A':<10} | {'N/A':<12} | {'N/A':<7} | {'N/A':<6}")
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Average Scores:")
    if results['summary']['average_scores']['model'] is not None:
        print(f"  Regular Model: {results['summary']['average_scores']['model']:.4f}")
    else:
        print(f"  Regular Model: N/A")
        
    if results['summary']['average_scores']['blind_model'] is not None:
        print(f"  Blind Model: {results['summary']['average_scores']['blind_model']:.4f}")
    else:
        print(f"  Blind Model: N/A")
    
    if results['summary']['average_scores']['actual_performance'] is not None:
        print(f"  Actual Performance: {results['summary']['average_scores']['actual_performance']:.4f}")
    else:
        print(f"  Actual Performance: N/A")
    
    print(f"\nAverage Prediction Errors:")
    if results['summary']['average_errors']['model'] is not None:
        print(f"  Regular Model: {results['summary']['average_errors']['model']:.4f}")
    else:
        print(f"  Regular Model: N/A")
        
    if results['summary']['average_errors']['blind_model'] is not None:
        print(f"  Blind Model: {results['summary']['average_errors']['blind_model']:.4f}")
    else:
        print(f"  Blind Model: N/A")
    
    print(f"\nWin Count (Better Score):")
    print(f"  Regular Model wins: {results['summary']['wins']['model_score']}")
    print(f"  Blind Model wins: {results['summary']['wins']['blind_model_score']}")
    print(f"  Ties: {results['summary']['wins']['score_ties']}")
    
    print(f"\nWin Count (Closer to Actual Performance):")
    print(f"  Regular Model wins: {results['summary']['wins']['model_accuracy']}")
    print(f"  Blind Model wins: {results['summary']['wins']['blind_model_accuracy']}")
    print(f"  Ties: {results['summary']['wins']['accuracy_ties']}")
    
    print(f"\nOverall Better Score: {results['summary']['overall_better_score']}")
    print(f"Overall More Accurate Model: {results['summary']['overall_more_accurate']}")
    
    # Print correlation between model predictions and actual performance
    if ('model_actual_correlation' in results['summary'] and 
        'blind_model_actual_correlation' in results['summary']):
        print("\nCorrelation with Actual Performance:")
        model_corr = results['summary'].get('model_actual_correlation')
        blind_corr = results['summary'].get('blind_model_actual_correlation')
        print(f"  Regular Model: {model_corr:.4f}" if model_corr is not None else "  Regular Model: N/A")
        print(f"  Blind Model: {blind_corr:.4f}" if blind_corr is not None else "  Blind Model: N/A")

def evaluate_price_targets(csv_path, target_timeframe="1m"):
    """
    Evaluate model price targets against current stock prices.
    
    Args:
        csv_path (str): Path to the CSV file with prediction data
        target_timeframe (str): Timeframe for evaluation, either "12m" or "1m"
        
    Returns:
        dict: Results of the price target evaluation
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create a results dictionary
    results = {
        'evaluation_date': datetime.now().strftime('%Y-%m-%d'),
        'target_timeframe': target_timeframe,
        'ticker_results': [],
        'summary': {}
    }
    
    # Check if the required columns exist
    column_exists = {}
    for column in df.columns:
        column_exists[column] = True
    
    # Prepare counters for summary
    model_within_range = 0
    blind_within_range = 0
    model_more_accurate = 0
    blind_more_accurate = 0
    accuracy_ties = 0
    
    # Select the right columns based on timeframe
    if target_timeframe == "1m":
        price_target_field = 'price_target_1m'
        blind_price_target_field = 'blind_price_target_1m'
        price_target_low_field = 'price_target_1m_low'
        price_target_mid_field = 'price_target_1m_mid'
        price_target_high_field = 'price_target_1m_high'
        blind_price_target_low_field = 'blind_price_target_1m_low'
        blind_price_target_mid_field = 'blind_price_target_1m_mid'
        blind_price_target_high_field = 'blind_price_target_1m_high'
    else:  # 12m is default
        price_target_field = 'price_target'
        blind_price_target_field = 'blind_price_target'
        price_target_low_field = 'price_target_low'
        price_target_mid_field = 'price_target_mid'
        price_target_high_field = 'price_target_high'
        blind_price_target_low_field = 'blind_price_target_low'
        blind_price_target_mid_field = 'blind_price_target_mid'
        blind_price_target_high_field = 'blind_price_target_high'
    
    # Check if all required columns are present
    required_columns = [
        price_target_field, blind_price_target_field,
        price_target_low_field, price_target_mid_field, price_target_high_field,
        blind_price_target_low_field, blind_price_target_mid_field, blind_price_target_high_field
    ]
    
    missing_columns = [col for col in required_columns if col not in column_exists]
    if missing_columns:
        print(f"Warning: Missing columns for {target_timeframe} evaluation: {', '.join(missing_columns)}")
        print(f"Available columns: {', '.join(df.columns)}")
    
    # Process price targets for each ticker
    for _, row in df.iterrows():
        ticker = row['ticker']
        
        # Get current price data
        try:
            ticker_obj = yf.Ticker(ticker)
            current_data = ticker_obj.history(period="1d")
            
            if current_data.empty:
                current_price = None
                price_date = None
            else:
                current_price = current_data['Close'].iloc[-1]
                price_date = current_data.index[-1].strftime('%Y-%m-%d')
                
        except Exception as e:
            current_price = None
            price_date = None
        
        # Parse price targets - try multiple approaches
        model_price_target = None
        blind_price_target = None
        
        # First try to use the direct JSON fields
        try:
            if price_target_field in row and pd.notna(row[price_target_field]):
                model_price_target = json.loads(row[price_target_field])
            if blind_price_target_field in row and pd.notna(row[blind_price_target_field]):
                blind_price_target = json.loads(row[blind_price_target_field])
        except (json.JSONDecodeError, TypeError):
            model_price_target = None
            blind_price_target = None
        
        # If direct JSON fields didn't work, try using individual columns
        if model_price_target is None:
            has_model_columns = (
                price_target_low_field in row and 
                price_target_mid_field in row and 
                price_target_high_field in row and
                pd.notna(row.get(price_target_low_field)) and 
                pd.notna(row.get(price_target_mid_field)) and 
                pd.notna(row.get(price_target_high_field))
            )
            
            if has_model_columns:
                model_price_target = {
                    'low': row[price_target_low_field],
                    'mid': row[price_target_mid_field],
                    'high': row[price_target_high_field]
                }
                
        if blind_price_target is None:
            has_blind_columns = (
                blind_price_target_low_field in row and 
                blind_price_target_mid_field in row and 
                blind_price_target_high_field in row and
                pd.notna(row.get(blind_price_target_low_field)) and 
                pd.notna(row.get(blind_price_target_mid_field)) and 
                pd.notna(row.get(blind_price_target_high_field))
            )
            
            if has_blind_columns:
                blind_price_target = {
                    'low': row[blind_price_target_low_field],
                    'mid': row[blind_price_target_mid_field],
                    'high': row[blind_price_target_high_field]
                }
        
        # Initialize ticker result
        ticker_result = {
            'ticker': ticker,
            'cutoff_date': row['cutoff_date'] if pd.notna(row['cutoff_date']) else None,
            'price_at_cutoff': row['price_at_cutoff'] if pd.notna(row['price_at_cutoff']) else None,
            'current_price': current_price,
            'current_price_date': price_date,
            'model_price_target': model_price_target,
            'blind_price_target': blind_price_target,
            'timeframe': target_timeframe
        }
        
        # If we have current price and price targets, evaluate accuracy
        if current_price is not None:
            # Calculate percent change from cutoff date
            if pd.notna(row['price_at_cutoff']):
                percent_change = ((current_price - row['price_at_cutoff']) / row['price_at_cutoff']) * 100
                ticker_result['percent_change'] = percent_change
            
            # Check if current price is within target ranges
            if model_price_target:
                model_within = (model_price_target['low'] <= current_price <= model_price_target['high'])
                ticker_result['model_target_accurate'] = model_within
                
                # Calculate accuracy metrics for the model
                model_mid = model_price_target['mid']
                model_error = abs(current_price - model_mid)
                model_error_percent = (model_error / model_mid) * 100
                ticker_result['model_error'] = model_error
                ticker_result['model_error_percent'] = model_error_percent
                
                # Add to counter if within range
                if model_within:
                    model_within_range += 1
            
            if blind_price_target:
                blind_within = (blind_price_target['low'] <= current_price <= blind_price_target['high'])
                ticker_result['blind_target_accurate'] = blind_within
                
                # Calculate accuracy metrics for the blind model
                blind_mid = blind_price_target['mid']
                blind_error = abs(current_price - blind_mid)
                blind_error_percent = (blind_error / blind_mid) * 100
                ticker_result['blind_error'] = blind_error
                ticker_result['blind_error_percent'] = blind_error_percent
                
                # Add to counter if within range
                if blind_within:
                    blind_within_range += 1
            
            # Compare which model was more accurate (only if both have targets)
            if model_price_target and blind_price_target:
                if model_error < blind_error:
                    ticker_result['more_accurate_model'] = 'Regular Model'
                    model_more_accurate += 1
                elif blind_error < model_error:
                    ticker_result['more_accurate_model'] = 'Blind Model'
                    blind_more_accurate += 1
                else:
                    ticker_result['more_accurate_model'] = 'Tie'
                    accuracy_ties += 1
        
        results['ticker_results'].append(ticker_result)
    
    # Calculate summary statistics
    valid_results = [r for r in results['ticker_results'] if r.get('current_price') is not None]
    total_evaluated = len(valid_results)
    
    if total_evaluated > 0:
        # Calculate average errors
        model_errors = [r['model_error'] for r in valid_results if 'model_error' in r]
        blind_errors = [r['blind_error'] for r in valid_results if 'blind_error' in r]
        
        model_errors_pct = [r['model_error_percent'] for r in valid_results if 'model_error_percent' in r]
        blind_errors_pct = [r['blind_error_percent'] for r in valid_results if 'blind_error_percent' in r]
        
        # Set up summary
        results['summary'] = {
            'total_evaluated': total_evaluated,
            'model_within_range': model_within_range,
            'blind_within_range': blind_within_range,
            'model_more_accurate': model_more_accurate,
            'blind_more_accurate': blind_more_accurate,
            'accuracy_ties': accuracy_ties,
            'average_errors': {
                'model': round(np.mean(model_errors), 2) if model_errors else None,
                'blind_model': round(np.mean(blind_errors), 2) if blind_errors else None,
                'model_percent': round(np.mean(model_errors_pct), 2) if model_errors_pct else None,
                'blind_model_percent': round(np.mean(blind_errors_pct), 2) if blind_errors_pct else None
            }
        }
        
        # Determine which model has better overall accuracy
        if model_more_accurate > blind_more_accurate:
            results['summary']['overall_more_accurate'] = 'Regular Model'
        elif blind_more_accurate > model_more_accurate:
            results['summary']['overall_more_accurate'] = 'Blind Model'
        else:
            results['summary']['overall_more_accurate'] = 'Tie'
    
    return results

def print_price_target_results(results):
    """
    Print the price target evaluation results in a readable format.
    
    Args:
        results (dict): Results from evaluate_price_targets function
    """
    timeframe = results.get('target_timeframe', '12m')
    timeframe_label = "1-Month" if timeframe == "1m" else "12-Month"
    
    print(f"Price Target Evaluation ({timeframe_label})")
    print(f"Evaluation Date: {results['evaluation_date']}")
    
    # Print the table header
    print(f"\nPrice Target Comparison by Ticker ({timeframe_label}):")
    print("-" * 140)
    header = (f"{'Ticker':<6} | {'Cutoff Price':<12} | {'Current Price':<12} | {'% Change':<9} | " +
              f"{'Model Target':<20} | {'Blind Target':<20} | {'Model Acc':<9} | {'Blind Acc':<9} | {'Better Model':<12}")
    print(header)
    print("-" * 140)
    
    for result in results['ticker_results']:
        ticker = result['ticker']
        cutoff_price = f"${result['price_at_cutoff']:.2f}" if result['price_at_cutoff'] is not None else "N/A"
        current_price = f"${result['current_price']:.2f}" if result['current_price'] is not None else "N/A"
        
        pct_change = f"{result.get('percent_change', 0):.2f}%" if 'percent_change' in result else "N/A"
        
        # Format price targets
        if result.get('model_price_target'):
            model_target = f"${result['model_price_target']['low']:.2f}-${result['model_price_target']['high']:.2f}"
        else:
            model_target = "N/A"
            
        if result.get('blind_price_target'):
            blind_target = f"${result['blind_price_target']['low']:.2f}-${result['blind_price_target']['high']:.2f}"
        else:
            blind_target = "N/A"
        
        # Format accuracy indicators
        model_acc = "✓" if result.get('model_target_accurate', False) else "✗"
        model_acc = model_acc if result.get('model_price_target') else "N/A"
        
        blind_acc = "✓" if result.get('blind_target_accurate', False) else "✗"
        blind_acc = blind_acc if result.get('blind_price_target') else "N/A"
        
        better_model = result.get('more_accurate_model', "N/A")
        
        row = (f"{ticker:<6} | {cutoff_price:<12} | {current_price:<12} | {pct_change:<9} | " +
               f"{model_target:<20} | {blind_target:<20} | {model_acc:<9} | {blind_acc:<9} | {better_model:<12}")
        print(row)
    
    # Print summary statistics
    print(f"\nSummary ({timeframe_label} Targets):")
    if 'summary' in results and results['summary']:
        summary = results['summary']
        print(f"Total tickers evaluated: {summary['total_evaluated']}")
        
        if summary['total_evaluated'] > 0:
            print("\nTarget Range Accuracy:")
            if 'model_within_range' in summary:
                model_acc = (summary['model_within_range'] / summary['total_evaluated']) * 100
                print(f"  Regular Model within range: {summary['model_within_range']} of {summary['total_evaluated']} ({model_acc:.1f}%)")
            
            if 'blind_within_range' in summary:
                blind_acc = (summary['blind_within_range'] / summary['total_evaluated']) * 100
                print(f"  Blind Model within range: {summary['blind_within_range']} of {summary['total_evaluated']} ({blind_acc:.1f}%)")
            
            print("\nPrice Target Precision:")
            print(f"  Regular Model more accurate: {summary['model_more_accurate']}")
            print(f"  Blind Model more accurate: {summary['blind_more_accurate']}")
            print(f"  Ties: {summary['accuracy_ties']}")
            
            print("\nAverage Error from Target Midpoint:")
            avg_errors = summary.get('average_errors', {})
            
            model_err = avg_errors.get('model')
            model_err_pct = avg_errors.get('model_percent')
            if model_err is not None and model_err_pct is not None:
                print(f"  Regular Model: ${model_err:.2f} ({model_err_pct:.2f}%)")
            else:
                print(f"  Regular Model: N/A")
                
            blind_err = avg_errors.get('blind_model')
            blind_err_pct = avg_errors.get('blind_model_percent')
            if blind_err is not None and blind_err_pct is not None:
                print(f"  Blind Model: ${blind_err:.2f} ({blind_err_pct:.2f}%)")
            else:
                print(f"  Blind Model: N/A")
            
            print(f"\nOverall More Accurate Model: {summary.get('overall_more_accurate', 'N/A')}")
    else:
        print("No summary statistics available.")

def compare_timeframe_results(results_12m, results_1m):
    """
    Compare the results between 1-month and 12-month predictions.
    
    Args:
        results_12m (dict): Results from 12-month price target evaluation
        results_1m (dict): Results from 1-month price target evaluation
    """
    print("\n====== COMPARING 1-MONTH VS 12-MONTH PREDICTIONS ======\n")
    
    # Get statistics from both timeframes
    stats_12m = results_12m.get('summary', {})
    stats_1m = results_1m.get('summary', {})
    
    if not stats_12m or not stats_1m:
        print("Insufficient data for comparison.")
        return
    
    # Create combined ticker results table
    tickers_12m = {r['ticker']: r for r in results_12m['ticker_results']}
    tickers_1m = {r['ticker']: r for r in results_1m['ticker_results']}
    
    # Get all tickers from both sets
    all_tickers = sorted(set(list(tickers_12m.keys()) + list(tickers_1m.keys())))
    
    print("Side-by-Side Comparison by Ticker:")
    print("-" * 140)
    header = (f"{'Ticker':<6} | {'1M Model':<10} | {'1M Blind':<10} | {'1M Error %':<10} | {'1M Better':<10} | " +
              f"{'12M Model':<10} | {'12M Blind':<10} | {'12M Error %':<10} | {'12M Better':<10} | {'Better Timeframe':<15}")
    print(header)
    print("-" * 140)
    
    ticker_timeframe_wins = {'1m': 0, '12m': 0, 'tie': 0}
    
    for ticker in all_tickers:
        r1m = tickers_1m.get(ticker, {})
        r12m = tickers_12m.get(ticker, {})
        
        # Get 1-month accuracy
        model_1m_acc = "✓" if r1m.get('model_target_accurate', False) else "✗"
        model_1m_acc = model_1m_acc if r1m.get('model_price_target') else "N/A"
        
        blind_1m_acc = "✓" if r1m.get('blind_target_accurate', False) else "✗"
        blind_1m_acc = blind_1m_acc if r1m.get('blind_price_target') else "N/A"
        
        model_1m_err_pct = f"{r1m.get('model_error_percent', 0):.2f}%" if 'model_error_percent' in r1m else "N/A"
        better_1m = r1m.get('more_accurate_model', "N/A")
        
        # Get 12-month accuracy
        model_12m_acc = "✓" if r12m.get('model_target_accurate', False) else "✗"
        model_12m_acc = model_12m_acc if r12m.get('model_price_target') else "N/A"
        
        blind_12m_acc = "✓" if r12m.get('blind_target_accurate', False) else "✗"
        blind_12m_acc = blind_12m_acc if r12m.get('blind_price_target') else "N/A"
        
        model_12m_err_pct = f"{r12m.get('model_error_percent', 0):.2f}%" if 'model_error_percent' in r12m else "N/A"
        better_12m = r12m.get('more_accurate_model', "N/A")
        
        # Determine which timeframe was more accurate for regular model
        better_timeframe = "N/A"
        if ('model_error_percent' in r1m and 'model_error_percent' in r12m and
            r1m.get('model_price_target') and r12m.get('model_price_target')):
            
            err_1m = r1m['model_error_percent']
            err_12m = r12m['model_error_percent']
            
            if err_1m < err_12m:
                better_timeframe = "1-Month"
                ticker_timeframe_wins['1m'] += 1
            elif err_12m < err_1m:
                better_timeframe = "12-Month"
                ticker_timeframe_wins['12m'] += 1
            else:
                better_timeframe = "Tie"
                ticker_timeframe_wins['tie'] += 1
                
        row = (f"{ticker:<6} | {model_1m_acc:<10} | {blind_1m_acc:<10} | {model_1m_err_pct:<10} | {better_1m:<10} | " +
               f"{model_12m_acc:<10} | {blind_12m_acc:<10} | {model_12m_err_pct:<10} | {better_12m:<10} | {better_timeframe:<15}")
        print(row)
    
    # Compare average errors
    avg_model_err_1m = stats_1m.get('average_errors', {}).get('model_percent')
    avg_model_err_12m = stats_12m.get('average_errors', {}).get('model_percent')
    
    print("\nSummary Statistics Comparison:")
    print("-" * 60)
    print(f"{'Metric':<30} | {'1-Month':<12} | {'12-Month':<12}")
    print("-" * 60)
    
    # Target range accuracy
    model_acc_1m = (stats_1m.get('model_within_range', 0) / stats_1m.get('total_evaluated', 1)) * 100
    model_acc_12m = (stats_12m.get('model_within_range', 0) / stats_12m.get('total_evaluated', 1)) * 100
    
    print(f"{'Model within range %':<30} | {model_acc_1m:.1f}% | {model_acc_12m:.1f}%")
    
    # Average errors
    if avg_model_err_1m is not None and avg_model_err_12m is not None:
        print(f"{'Model avg error %':<30} | {avg_model_err_1m:.2f}% | {avg_model_err_12m:.2f}%")
    
    # Overall better timeframe
    overall_better = "1-Month" if avg_model_err_1m is not None and avg_model_err_12m is not None and avg_model_err_1m < avg_model_err_12m else "12-Month"
    print(f"{'Better timeframe by ticker count':<30} | {ticker_timeframe_wins['1m']} | {ticker_timeframe_wins['12m']}")
    print(f"{'Overall more accurate timeframe':<30} | {overall_better if avg_model_err_1m != avg_model_err_12m else 'Tie'}")

if __name__ == "__main__":
    # File path for the benchmark results
    csv_path = CSVy
    
    # Configure settings to only run 1-month evaluation
    run_1m_evaluation = True
    save_to_csv = True
    
    # Store results
    results_1m = None
    
    # ===== Price Target Evaluations =====
    if run_1m_evaluation:
        print("\n====== EVALUATING 1-MONTH PRICE TARGETS ======\n")
        results_1m = evaluate_price_targets(csv_path, target_timeframe="1m")
        print_price_target_results(results_1m)
        
        # Save final results
        if save_to_csv and results_1m:
            # Convert results to DataFrame for easier analysis
            def results_to_df(results):
                return pd.DataFrame([
                    {
                        'ticker': r['ticker'],
                        'cutoff_price': r['price_at_cutoff'],
                        'current_price': r['current_price'],
                        'percent_change': r.get('percent_change'),
                        'timeframe': r.get('timeframe', '1m'),
                        'model_target_low': r['model_price_target']['low'] if r.get('model_price_target') else None,
                        'model_target_mid': r['model_price_target']['mid'] if r.get('model_price_target') else None,
                        'model_target_high': r['model_price_target']['high'] if r.get('model_price_target') else None,
                        'blind_target_low': r['blind_price_target']['low'] if r.get('blind_price_target') else None,
                        'blind_target_mid': r['blind_price_target']['mid'] if r.get('blind_price_target') else None,
                        'blind_target_high': r['blind_price_target']['high'] if r.get('blind_price_target') else None,
                        'model_target_accurate': r.get('model_target_accurate'),
                        'blind_target_accurate': r.get('blind_target_accurate'),
                        'model_error': r.get('model_error'),
                        'blind_error': r.get('blind_error'),
                        'model_error_percent': r.get('model_error_percent'),
                        'blind_error_percent': r.get('blind_error_percent'),
                        'more_accurate_model': r.get('more_accurate_model')
                    }
                    for r in results['ticker_results']
                ])
            
            results_df = results_to_df(results_1m)
            output_filename = f'price_target_evaluation_1m_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            results_df.to_csv(output_filename, index=False)
            print(f"\nResults exported to '{output_filename}'")
    
    print("\nEvaluation complete!")