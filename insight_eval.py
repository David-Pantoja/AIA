import pandas as pd
import json
import numpy as np
from datetime import datetime
import yfinance as yf

CSVy = "securities_insight_output.csv"

# uses the output from securities insight to calculate a score against the blind model
def calculate_score(probabilities):
    if probabilities is None or pd.isna(probabilities) or not (isinstance(probabilities, (dict, str))):
        return None
    
    if isinstance(probabilities, str):
        try:
            probabilities = json.loads(probabilities)
        except (json.JSONDecodeError, TypeError):
            return None
    
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
    # get the stock data and analyze it
    try:
        ticker_obj = yf.Ticker(ticker)
        
        stock_data = ticker_obj.history(
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True,
            actions=True,
            back_adjust=True,
            repair=True,
            rounding=False,
            timeout=20
        )
        
        if stock_data.empty:
            return {
                'success': False,
                'message': f"No data available for {ticker} in the specified date range"
            }
        
        dividends = ticker_obj.dividends
        if not dividends.empty:
            dividends = dividends[
                (dividends.index >= pd.Timestamp(start_date)) & 
                (dividends.index <= pd.Timestamp(end_date))
            ]
            total_dividends = dividends.sum() if not dividends.empty else 0
        else:
            total_dividends = 0
            
        splits = ticker_obj.splits
        has_splits = False
        if not splits.empty:
            splits_in_range = splits[
                (splits.index >= pd.Timestamp(start_date)) & 
                (splits.index <= pd.Timestamp(end_date))
            ]
            has_splits = not splits_in_range.empty
        
        start_price = stock_data['Close'].iloc[0]
        end_price = stock_data['Close'].iloc[-1]
        price_change = end_price - start_price
        
        total_return = price_change + total_dividends
        percent_change = (price_change / start_price) * 100
        total_return_percent = (total_return / start_price) * 100
        
        performance_score = np.clip(percent_change / 50, -1, 1)
        
        total_return_score = np.clip(total_return_percent / 50, -1, 1)
        
        daily_returns = stock_data['Close'].pct_change().dropna()
        volatility = daily_returns.std() * 100
        
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
    # figure out which model did better with actual price data
    df = pd.read_csv(csv_path)
    
    for col in ['model_probabilities', 'blind_model_probabilities', 'yfinance_probabilities']:
        df[col] = df[col].apply(lambda x: None if pd.isna(x) else x)
    
    df['model_score'] = df['model_probabilities'].apply(calculate_score)
    df['blind_model_score'] = df['blind_model_probabilities'].apply(calculate_score)
    df['yfinance_score'] = df['yfinance_probabilities'].apply(calculate_score)
    
    results = {
        'evaluation_date': datetime.now().strftime('%Y-%m-%d'),
        'start_date': start_date,
        'end_date': end_date,
        'ticker_results': [],
        'summary': {}
    }
    
    for _, row in df.iterrows():
        ticker = row['ticker']
        
        performance = get_stock_performance(ticker, start_date, end_date)
        
        ticker_result = {
            'ticker': ticker,
            'model_score': round(row['model_score'], 4) if pd.notna(row['model_score']) else None,
            'blind_model_score': round(row['blind_model_score'], 4) if pd.notna(row['blind_model_score']) else None,
            'yfinance_score': round(row['yfinance_score'], 4) if pd.notna(row['yfinance_score']) else None,
        }
        
        if performance['success']:
            ticker_result['actual_performance'] = performance
            
            if use_total_return and 'total_return_score' in performance:
                perf_score = performance['total_return_score']
            else:
                perf_score = performance['performance_score']
                
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
    
    valid_model_scores = [r['model_score'] for r in results['ticker_results'] if r['model_score'] is not None]
    valid_blind_scores = [r['blind_model_score'] for r in results['ticker_results'] if r['blind_model_score'] is not None]
    valid_yfinance_scores = [r['yfinance_score'] for r in results['ticker_results'] if r['yfinance_score'] is not None]
    
    performance_scores = []
    model_errors = []
    blind_errors = []
    
    model_perf_pairs = []
    blind_model_perf_pairs = []
    
    for r in results['ticker_results']:
        if 'actual_performance' in r and r['actual_performance'].get('success', False):
            if use_total_return and 'total_return_score' in r['actual_performance']:
                perf_score = r['actual_performance']['total_return_score']
            else:
                perf_score = r['actual_performance']['performance_score']
                
            performance_scores.append(perf_score)
            
            if 'model_error' in r and r['model_error'] is not None:
                model_errors.append(r['model_error'])
                
            if 'blind_error' in r and r['blind_error'] is not None:
                blind_errors.append(r['blind_error'])
                
            if r['model_score'] is not None:
                model_perf_pairs.append((r['model_score'], perf_score))
                
            if r['blind_model_score'] is not None:
                blind_model_perf_pairs.append((r['blind_model_score'], perf_score))
    
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
    
    if results['summary']['average_scores']['model'] is not None and results['summary']['average_scores']['blind_model'] is not None:
        if results['summary']['average_scores']['model'] > results['summary']['average_scores']['blind_model']:
            results['summary']['overall_better_score'] = 'Regular Model'
        elif results['summary']['average_scores']['blind_model'] > results['summary']['average_scores']['model']:
            results['summary']['overall_better_score'] = 'Blind Model'
        else:
            results['summary']['overall_better_score'] = 'Tie'
    else:
        results['summary']['overall_better_score'] = 'N/A'
    
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

def evaluate_price_targets(csv_path, target_timeframe="1m"):
    # check how accurate the price targets were
    df = pd.read_csv(csv_path)
    
    results = {
        'evaluation_date': datetime.now().strftime('%Y-%m-%d'),
        'target_timeframe': target_timeframe,
        'ticker_results': [],
        'summary': {}
    }
    
    column_exists = {}
    for column in df.columns:
        column_exists[column] = True
    
    model_within_range = 0
    blind_within_range = 0
    model_more_accurate = 0
    blind_more_accurate = 0
    accuracy_ties = 0
    
    if target_timeframe == "1m":
        price_target_field = 'price_target_1m'
        blind_price_target_field = 'blind_price_target_1m'
        price_target_low_field = 'price_target_1m_low'
        price_target_mid_field = 'price_target_1m_mid'
        price_target_high_field = 'price_target_1m_high'
        blind_price_target_low_field = 'blind_price_target_1m_low'
        blind_price_target_mid_field = 'blind_price_target_1m_mid'
        blind_price_target_high_field = 'blind_price_target_1m_high'
    else:
        price_target_field = 'price_target'
        blind_price_target_field = 'blind_price_target'
        price_target_low_field = 'price_target_low'
        price_target_mid_field = 'price_target_mid'
        price_target_high_field = 'price_target_high'
        blind_price_target_low_field = 'blind_price_target_low'
        blind_price_target_mid_field = 'blind_price_target_mid'
        blind_price_target_high_field = 'blind_price_target_high'
    
    required_columns = [
        price_target_field, blind_price_target_field,
        price_target_low_field, price_target_mid_field, price_target_high_field,
        blind_price_target_low_field, blind_price_target_mid_field, blind_price_target_high_field
    ]
    
    missing_columns = [col for col in required_columns if col not in column_exists]
    
    for _, row in df.iterrows():
        ticker = row['ticker']
        
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
        
        model_price_target = None
        blind_price_target = None
        
        try:
            if price_target_field in row and pd.notna(row[price_target_field]):
                model_price_target = json.loads(row[price_target_field])
            if blind_price_target_field in row and pd.notna(row[blind_price_target_field]):
                blind_price_target = json.loads(row[blind_price_target_field])
        except (json.JSONDecodeError, TypeError):
            model_price_target = None
            blind_price_target = None
        
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
        
        if current_price is not None:
            if pd.notna(row['price_at_cutoff']):
                percent_change = ((current_price - row['price_at_cutoff']) / row['price_at_cutoff']) * 100
                ticker_result['percent_change'] = percent_change
            
            if model_price_target:
                model_within = (model_price_target['low'] <= current_price <= model_price_target['high'])
                ticker_result['model_target_accurate'] = model_within
                
                model_mid = model_price_target['mid']
                model_error = abs(current_price - model_mid)
                model_error_percent = (model_error / model_mid) * 100
                ticker_result['model_error'] = model_error
                ticker_result['model_error_percent'] = model_error_percent
                
                if model_within:
                    model_within_range += 1
            
            if blind_price_target:
                blind_within = (blind_price_target['low'] <= current_price <= blind_price_target['high'])
                ticker_result['blind_target_accurate'] = blind_within
                
                blind_mid = blind_price_target['mid']
                blind_error = abs(current_price - blind_mid)
                blind_error_percent = (blind_error / blind_mid) * 100
                ticker_result['blind_error'] = blind_error
                ticker_result['blind_error_percent'] = blind_error_percent
                
                if blind_within:
                    blind_within_range += 1
            
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
    
    valid_results = [r for r in results['ticker_results'] if r.get('current_price') is not None]
    total_evaluated = len(valid_results)
    
    if total_evaluated > 0:
        model_errors = [r['model_error'] for r in valid_results if 'model_error' in r]
        blind_errors = [r['blind_error'] for r in valid_results if 'blind_error' in r]
        
        model_errors_pct = [r['model_error_percent'] for r in valid_results if 'model_error_percent' in r]
        blind_errors_pct = [r['blind_error_percent'] for r in valid_results if 'blind_error_percent' in r]
        
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
        
        if model_more_accurate > blind_more_accurate:
            results['summary']['overall_more_accurate'] = 'Regular Model'
        elif blind_more_accurate > model_more_accurate:
            results['summary']['overall_more_accurate'] = 'Blind Model'
        else:
            results['summary']['overall_more_accurate'] = 'Tie'
    
    return results

if __name__ == "__main__":
    csv_path = CSVy
    run_1m_evaluation = True
    save_to_csv = True
    results_1m = None
    
    if run_1m_evaluation:
        results_1m = evaluate_price_targets(csv_path, target_timeframe="1m")
        
        if save_to_csv and results_1m:
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
            output_filename = f'insight_eval_dated_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            results_df.to_csv(output_filename, index=False)