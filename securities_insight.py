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

# config stuff
USE_SEC = True
USE_YFINANCE = True
USE_TOTAL_RETURN = True
CUTOFF_DATE = "2025-03-01"
QUARTERS = 4
MAX_SEARCH = 100

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "AMD", "INTC", "TSM", "MU",
    "CRM", "ADBE", "ORCL", "IBM", "SAP",
    "CSCO", "AVGO", "QCOM", "TXN", "NXPI",
    "PYPL", "SQ", "ADYEY", "V", "MA",
    "ZM", "CRWD", "F",
    "JPM", "BAC", "GS", "MS", "WFC",
    "BLK", "BX", "KKR", "C", "AXP",
    "PGR", "ALL", "TRV", "CB", "MET",
    
    "JNJ", "PFE", "UNH", "MRK", "ABBV",
    "MDT", "TMO", "DHR", "ABT", "ISRG",

    "XOM", "CVX", "COP", "SLB", "EOG",
    "NEE", "DUK", "SO", "AEP", "PCG",
    
    "PG", "KO", "WMT", "MCD", "DIS",
    "NKE", "SBUX", "LULU", "TGT", "HD",
    "PEP", "KHC", "GIS", "K", "CAG",

    "TSLA", "GM", "TM", "VWAGY",
    "BA", "LMT", "RTX", "GE", "HON",
    
    "NFLX", "CMCSA", "CHTR", "ROKU", "SPOT",
    "T", "VZ", "TMUS", "LBRDK", "DISH",
    
    "AMGN", "GILD", "BIIB", "MRNA", "REGN",
    "ABNB", "UBER", "LYFT", "DASH", "BKNG",
    
    "TWLO", "NET", "ZS", "OKTA",
    "NOW", "TEAM", "DOCU", "WDAY"
]

def get_stock_performance(ticker: str, start_date: str, end_date: str) -> Dict:
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
        
        # stonks only go up
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
        
        sma_20 = stock_data['Close'].rolling(window=20).mean().iloc[-1]
        sma_50 = stock_data['Close'].rolling(window=50).mean().iloc[-1]
        
        std_20 = stock_data['Close'].rolling(window=20).std().iloc[-1]
        bb_upper = sma_20 + (2 * std_20)
        bb_lower = sma_20 - (2 * std_20)
        
        delta = stock_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
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
    total = sum(probs.values())
    return abs(total - 1.0) < 0.0001

def calculate_mse(yf_probs: Dict[str, float], model_probs: Dict[str, float]) -> float:
    if not validate_probability_distribution(yf_probs) or not validate_probability_distribution(model_probs):
        return float('inf')
    
    mse = 0.0
    for action in ["strongBuy", "buy", "hold", "sell", "strongSell"]:
        yf_prob = yf_probs.get(action, 0)
        model_prob = model_probs.get(action, 0)
        mse += (yf_prob - model_prob) ** 2
    return mse / 5

def get_stock_price_on_date(ticker: str, date_str: str) -> Dict:
    try:
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        today = datetime.now().replace(tzinfo=None)
        
        if target_date > today:
            start_date = (today - timedelta(days=7)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
        else:
            start_date = (target_date - timedelta(days=5)).strftime('%Y-%m-%d')
            end_date = (target_date + timedelta(days=5)).strftime('%Y-%m-%d')
        
        ticker_obj = yf.Ticker(ticker)
        
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
        
        if target_date > today:
            closest_date = stock_data.index[-1]
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
            target_date_pd = pd.Timestamp(date_str)
            if not stock_data.empty and stock_data.index[0].tzinfo is not None:
                target_date_pd = target_date_pd.tz_localize(stock_data.index[0].tzinfo)
            
            closest_date = min(stock_data.index, key=lambda x: abs(x - target_date_pd))
            closest_date_str = closest_date.strftime('%Y-%m-%d')
            
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
    cutoff_date_str = CUTOFF_DATE
    cutoff_datetime = datetime.strptime(cutoff_date_str, '%Y-%m-%d')
    
    price_data = get_stock_price_on_date(ticker, cutoff_date_str)
    
    generator = InsightGenerator(
        os.getenv("OPENAI_API_KEY"),
        use_yfinance=USE_YFINANCE,
        use_SEC=USE_SEC,
        cutoff_date=cutoff_datetime
    )
    
    insight = generator.generate_insight(
        ticker,
        quarters=QUARTERS,
        max_search=MAX_SEARCH,
        cutoff_date=cutoff_datetime
    )
    
    if not insight:
        return None
        
    blind_insight = generator.generate_insight(
        ticker,
        quarters=0,
        max_search=0,
        use_SEC_override=False,
        use_yfinance_override=False
    )

    if not blind_insight:
        blind_model_probs = None
        blind_mse = None
    else:
        blind_recommendation = blind_insight.get("recommendation", {})
        blind_model_probs = {
            "strongBuy": blind_recommendation.get("buy_probability", 0) * 0.6,
            "buy": blind_recommendation.get("buy_probability", 0) * 0.4,
            "hold": blind_recommendation.get("hold_probability", 0),
            "sell": blind_recommendation.get("sell_probability", 0) * 0.4,
            "strongSell": blind_recommendation.get("sell_probability", 0) * 0.6
        }
        total_blind_prob = sum(blind_model_probs.values())
        if total_blind_prob > 0:
             blind_model_probs = {k: v / total_blind_prob for k, v in blind_model_probs.items()}
        else:
             blind_model_probs = {k: 0.0 for k in blind_model_probs}

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
    
    if USE_SEC and "filings" in insight and insight["filings"]:
        result["quarterly_reports_found"] = sum(
            1 for f in insight.get("filings", [])
            if f.get("form_type") in ["10-Q", "10-K"]
        )
    
    if "recommendation" in insight:
        recommendation = insight["recommendation"]
        result["model_probabilities"] = {
            "strongBuy": recommendation.get("buy_probability", 0) * 0.6,
            "buy": recommendation.get("buy_probability", 0) * 0.4,
            "hold": recommendation.get("hold_probability", 0),
            "sell": recommendation.get("sell_probability", 0) * 0.4,
            "strongSell": recommendation.get("sell_probability", 0) * 0.6
        }
        
        result["price_target"] = recommendation.get("price_target")
        result["price_target_1m"] = recommendation.get("price_target_1m")
    
    if USE_YFINANCE and "yfinance_comparison" in insight:
        yf_comparison = insight["yfinance_comparison"]
        yf_probs = yf_comparison.get("yfinance_probabilities")
        model_probs = yf_comparison.get("model_probabilities")

        if yf_probs:
             result["yfinance_probabilities"] = yf_probs
             if model_probs:
                 result["model_probabilities"] = model_probs
                 result["mse"] = calculate_mse(yf_probs, model_probs)
             
             if blind_model_probs:
                 result["blind_mse"] = calculate_mse(yf_probs, blind_model_probs)
    
    if blind_insight and "recommendation" in blind_insight:
        blind_recommendation = blind_insight.get("recommendation", {})
        result["blind_price_target"] = blind_recommendation.get("price_target")
        result["blind_price_target_1m"] = blind_recommendation.get("price_target_1m")
    
    return result

def save_results(results: List[Dict], filename: str) -> None:
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    for col in ['model_probabilities', 'blind_model_probabilities', 'yfinance_probabilities', 
                'price_data', 'price_target', 'price_target_1m', 'blind_price_target', 
                'blind_price_target_1m']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(x) if x is not None else None)
    
    def extract_target_value(json_str, key):
        if not json_str or json_str == 'null':
            return None
        try:
            data = json.loads(json_str)
            return data.get(key) if isinstance(data, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None
            
    df['price_target_low'] = df['price_target'].apply(lambda x: extract_target_value(x, 'low'))
    df['price_target_mid'] = df['price_target'].apply(lambda x: extract_target_value(x, 'mid'))
    df['price_target_high'] = df['price_target'].apply(lambda x: extract_target_value(x, 'high'))
    
    df['price_target_1m_low'] = df['price_target_1m'].apply(lambda x: extract_target_value(x, 'low'))
    df['price_target_1m_mid'] = df['price_target_1m'].apply(lambda x: extract_target_value(x, 'mid'))
    df['price_target_1m_high'] = df['price_target_1m'].apply(lambda x: extract_target_value(x, 'high'))
    
    df['blind_price_target_low'] = df['blind_price_target'].apply(lambda x: extract_target_value(x, 'low'))
    df['blind_price_target_mid'] = df['blind_price_target'].apply(lambda x: extract_target_value(x, 'mid'))
    df['blind_price_target_high'] = df['blind_price_target'].apply(lambda x: extract_target_value(x, 'high'))
    
    df['blind_price_target_1m_low'] = df['blind_price_target_1m'].apply(lambda x: extract_target_value(x, 'low'))
    df['blind_price_target_1m_mid'] = df['blind_price_target_1m'].apply(lambda x: extract_target_value(x, 'mid'))
    df['blind_price_target_1m_high'] = df['blind_price_target_1m'].apply(lambda x: extract_target_value(x, 'high'))
    
    df.to_csv(filename, index=False)
    
def main():
    results = []
    
    for ticker in TICKERS:
        ticker_result = process_ticker(ticker)
        if ticker_result:
            results.append(ticker_result)
    
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cutoff_str = CUTOFF_DATE.replace('-', '')
        file_name = f"securities_insight_output_{cutoff_str}_{timestamp}.csv"
        save_results(results, file_name)
    
if __name__ == "__main__":
    main() 