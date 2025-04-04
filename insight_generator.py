import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import openai
from dotenv import load_dotenv
import os
from sec_edgar_fetcher import fetch_filings
from sec_edgar_fetcher.fetcher import SECFetcher
from stock_data_fetcher.fetcher import StockDataFetcher
from cachetools import TTLCache, cached

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

load_dotenv()

class InsightGenerator:
    def __init__(self, openai_api_key: str, use_yfinance: bool = True, use_SEC: bool = True, cutoff_date: Optional[datetime] = None):
        self.openai_api_key = openai_api_key
        self.use_yfinance = use_yfinance
        self.use_SEC = use_SEC
        self.cutoff_date = cutoff_date or datetime.now()
        self.sec_fetcher = SECFetcher()
        self.stock_fetcher = StockDataFetcher()
        openai.api_key = openai_api_key

    def _prepare_financial_data(self, ticker: str, quarters: int = 8, max_search: int = 100, 
                               use_SEC_override: Optional[bool] = None, 
                               use_yfinance_override: Optional[bool] = None,
                               cutoff_date: Optional[datetime] = None) -> Dict:
        try:
            effective_cutoff = cutoff_date or self.cutoff_date
            
            fetch_sec = self.use_SEC if use_SEC_override is None else use_SEC_override
            fetch_yfinance = self.use_yfinance if use_yfinance_override is None else use_yfinance_override
            
            all_filings = []
            price_reaction = None
            analyst_ratings = None
            price_data = None
            technical_indicators = None

            # get sec filings if enabled
            if fetch_sec:
                all_filings = fetch_filings(
                    ticker, 
                    "ALL",
                    quarters=quarters,
                    max_search=max_search,
                    cutoff_date=effective_cutoff
                )

            # get market data if enabled
            if fetch_yfinance:
                fetcher = StockDataFetcher()
                
                cutoff_date_str = effective_cutoff.strftime("%Y-%m-%d")
                start_date_str = (effective_cutoff - timedelta(days=365)).strftime("%Y-%m-%d")
                
                analyst_ratings = fetcher.get_analyst_ratings(ticker, cutoff_date=effective_cutoff)
                
                price_data = fetcher.get_historical_prices(
                    ticker,
                    start_date=start_date_str,
                    end_date=cutoff_date_str
                )
                
                if price_data:
                    technical_indicators = fetcher.calculate_technical_indicators(price_data)

            if fetch_sec and not all_filings and fetch_yfinance and not price_data:
                return {}

            return {
                "filings": all_filings,
                "price_reaction": price_reaction,
                "analyst_ratings": analyst_ratings,
                "price_data": price_data,
                "technical_indicators": technical_indicators
            }
            
        except Exception as e:
            return {}

    def _create_analysis_prompt(self, financial_data: Dict, ticker: str, is_blind: bool = False) -> str:
        if is_blind:
            prompt = f"""You are a financial analyst assistant. Provide a general investment recommendation for the ticker {ticker}. 
            Base your assessment ONLY on your general knowledge of this company, its sector, and overall market conditions. 
            Do NOT assume you have been given specific financial data, filings, or technical indicators for this ticker.

            Provide a probabilistic recommendation (buy/hold/sell) with confidence levels and price targets for both 1-month and 12-month timeframes.

            Format your response as JSON with the following structure:
            {{
              "summary": "General assessment based on common knowledge.",
              "financial_health": "General assessment based on common knowledge.",
              "recommendation": {{
                "action": "",
                "buy_probability": 0.0,
                "hold_probability": 0.0,
                "sell_probability": 0.0,
                "confidence_level": 0.0,
                "price_target_1m": {{"low": 0.0, "mid": 0.0, "high": 0.0}},
                "price_target": {{"low": 0.0, "mid": 0.0, "high": 0.0}},
                "price_target_timeframe": "12 months"
              }},
              "risk_factors": ["General risks associated with this type of company/sector."],
              "technical_analysis": {{}},
              "market_sentiment": {{}}
            }}
            Ensure the probabilities sum to 1.0."""
            return prompt

        prompt_sections = []
        
        use_sec_data = self.use_SEC and financial_data.get("filings")
        use_yfinance_data = self.use_yfinance and (financial_data.get("technical_indicators") or financial_data.get("price_data"))

        if use_sec_data:
            all_filings = financial_data["filings"]
            if all_filings:
                latest_filing = all_filings[0]
                financial_metrics = latest_filing.get("financials", {})
                mda_text = financial_metrics.get("Management Discussion & Analysis summary", "")
                
                prompt_sections.append(f"""Financial Metrics:
{json.dumps(financial_metrics, indent=2)}

Management Discussion:
{mda_text}

Recent Filings Summary:
{chr(10).join(f"- {f['form_type']} ({f['filing_date']}): {f.get('financials', {}).get('Revenue', 'N/A')} Revenue" for f in all_filings[:5])}""")
        
        if use_yfinance_data:
            tech_indicators = financial_data.get("technical_indicators", {})
            price_data = financial_data.get("price_data", {})
            analyst_ratings = financial_data.get("analyst_ratings", {})
            
            if price_data and "close" in price_data:
                latest_price = price_data["close"][-1] if price_data["close"] else "N/A"
                latest_volume = price_data["volume"][-1] if price_data["volume"] else "N/A"
            else:
                latest_price = "N/A"
                latest_volume = "N/A"
            
            price_target_info = ""
            if analyst_ratings:
                mean_target = analyst_ratings.get("mean_target")
                median_target = analyst_ratings.get("median_target")
                if mean_target is not None or median_target is not None:
                    price_target_info = f"""
Analyst Price Targets:
  Mean Target: ${mean_target if mean_target is not None else 'N/A'}
  Median Target: ${median_target if median_target is not None else 'N/A'}"""
            
            prompt_sections.append(f"""Market Data:
Price: ${latest_price}
Volume: {latest_volume}{price_target_info}
RSI: {tech_indicators.get('rsi', 'N/A')}
MACD: {tech_indicators.get('macd', 'N/A')}
SMA 20: ${tech_indicators.get('sma_20', 'N/A')}
SMA 50: ${tech_indicators.get('sma_50', 'N/A')}
Bollinger Bands:
  Upper: ${tech_indicators.get('bollinger_bands', {}).get('upper', 'N/A')}
  Lower: ${tech_indicators.get('bollinger_bands', {}).get('lower', 'N/A')}""")
        
        data_presence_message = "Based on the provided data" if prompt_sections else "Based on general knowledge (no specific data provided)"
        prompt = f"""You are a financial analyst assistant. Analyze the following financial data for {ticker}:
{chr(10).join(prompt_sections) if prompt_sections else 'No specific financial data provided.'}

{data_presence_message}, provide:
1. A concise summary of key performance indicators
2. Your assessment of the company's financial health
3. A probabilistic recommendation (buy/hold/sell) with confidence levels
4. Price targets for both 1-month and 12-month timeframes, each with low, mid, and high ranges
5. Key risk factors to consider

Format your response as JSON with the following structure:
{{
  "summary": "",
  "financial_health": "",
  "recommendation": {{
    "action": "",
    "buy_probability": 0.0,
    "hold_probability": 0.0,
    "sell_probability": 0.0,
    "confidence_level": 0.0,
    "price_target_1m": {{
      "low": 0.0,
      "mid": 0.0,
      "high": 0.0
    }},
    "price_target": {{
      "low": 0.0,
      "mid": 0.0,
      "high": 0.0
    }},
    "price_target_timeframe": "12 months"
  }},
  "risk_factors": [],
  "technical_analysis": {{
    "trend": "",
    "momentum": "",
    "support_resistance": ""
  }},
  "market_sentiment": {{
    "analyst_consensus": "",
    "price_momentum": "",
    "volume_trend": ""
  }}
}}"""

        return prompt

    def generate_insight(self, ticker: str, quarters: int = 8, max_search: int = 100,
                       use_SEC_override: Optional[bool] = None, 
                       use_yfinance_override: Optional[bool] = None,
                       cutoff_date: Optional[datetime] = None) -> Dict:
        try:
            # check if blind mode
            is_blind_run = use_SEC_override is False and use_yfinance_override is False
            
            financial_data = self._prepare_financial_data(
                ticker, quarters, max_search, 
                use_SEC_override=use_SEC_override, 
                use_yfinance_override=use_yfinance_override,
                cutoff_date=cutoff_date
            )
            
            prompt = self._create_analysis_prompt(financial_data, ticker, is_blind=is_blind_run)

            # let's ask gpt what it thinks
            response = openai.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": """You are an expert financial analyst with deep knowledge of market analysis, 
                    technical indicators, and fundamental analysis. Your task is to provide comprehensive investment insights 
                    based on the provided data. Always respond with valid JSON and ensure all probabilities sum to 1.0."""},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                presence_penalty=0.6,
                frequency_penalty=0.6
            )

            try:
                content = response.choices[0].message.content.strip()
                
                if content.startswith("```"):
                    content = content.replace("```json", "").replace("```", "").strip()
                
                try:
                    analysis = json.loads(content)
                except json.JSONDecodeError as je:
                    import re
                    
                    pattern = r'"([^"]+)"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]|"[^"]*"|[^,}\]]*)'
                    matches = re.findall(pattern, content)
                    
                    fixed_content = "{"
                    for i, (key, value) in enumerate(matches):
                        fixed_content += f'"{key}": {value}'
                        if i < len(matches) - 1:
                            fixed_content += ","
                    fixed_content += "}"
                    
                    try:
                        analysis = json.loads(fixed_content)
                    except json.JSONDecodeError:
                        import re
                        
                        fixed_content = re.sub(r':\s+\{', ': {', content)
                        fixed_content = re.sub(r',\s+', ', ', fixed_content)
                        fixed_content = re.sub(r':\s+', ': ', fixed_content)
                        fixed_content = re.sub(r'\s+}', ' }', fixed_content)
                        
                        fixed_content = re.sub(r'"([^"]*)\n([^"]*)"', r'"\1\\n\2"', fixed_content)
                        
                        try:
                            analysis = json.loads(fixed_content)
                        except json.JSONDecodeError:
                            try:
                                try:
                                    import json5
                                    analysis = json5.loads(content)
                                except ImportError:
                                    fixed_content = re.sub(r'\s+', ' ', content)
                                    fixed_content = fixed_content.replace("'", '"').replace(': "', ':"').replace('", ', '",')
                                    fixed_content = re.sub(r',\s*}', '}', fixed_content)
                                    fixed_content = re.sub(r',\s*]', ']', fixed_content)
                                    
                                    analysis = json.loads(fixed_content)
                            except Exception as e:
                                try:
                                    sections = {}
                                    for section in ["summary", "financial_health", "recommendation", "risk_factors", "technical_analysis", "market_sentiment"]:
                                        pattern = rf'"{section}"\s*:\s*(.*?)(?=,"[^"]+":|\}}$)'
                                        match = re.search(pattern, content, re.DOTALL)
                                        if match:
                                            sections[section] = match.group(1).strip()
                                    
                                    manual_json = {
                                        "summary": sections.get("summary", "").replace('"', '').strip('"{}[], '),
                                        "financial_health": sections.get("financial_health", "").replace('"', '').strip('"{}[], '),
                                        "recommendation": {},
                                        "risk_factors": [],
                                        "technical_analysis": {},
                                        "market_sentiment": {}
                                    }
                                    
                                    if "recommendation" in sections:
                                        rec_str = sections["recommendation"]
                                        action_match = re.search(r'"action"\s*:\s*"([^"]*)"', rec_str)
                                        buy_prob_match = re.search(r'"buy_probability"\s*:\s*([\d.]+)', rec_str)
                                        hold_prob_match = re.search(r'"hold_probability"\s*:\s*([\d.]+)', rec_str)
                                        sell_prob_match = re.search(r'"sell_probability"\s*:\s*([\d.]+)', rec_str)
                                        conf_match = re.search(r'"confidence_level"\s*:\s*([\d.]+)', rec_str)
                                        
                                        pt_low_match = re.search(r'"price_target"\s*:\s*\{\s*"low"\s*:\s*([\d.]+)', rec_str)
                                        pt_mid_match = re.search(r'"mid"\s*:\s*([\d.]+)', rec_str)
                                        pt_high_match = re.search(r'"high"\s*:\s*([\d.]+)', rec_str)
                                        
                                        pt_1m_low_match = re.search(r'"price_target_1m"\s*:\s*\{\s*"low"\s*:\s*([\d.]+)', rec_str)
                                        pt_1m_mid_match = re.search(r'"mid"\s*:\s*([\d.]+)', rec_str, re.DOTALL)
                                        pt_1m_high_match = re.search(r'"high"\s*:\s*([\d.]+)', rec_str, re.DOTALL)
                                        
                                        manual_json["recommendation"] = {
                                            "action": action_match.group(1) if action_match else "hold",
                                            "buy_probability": float(buy_prob_match.group(1)) if buy_prob_match else 0.33,
                                            "hold_probability": float(hold_prob_match.group(1)) if hold_prob_match else 0.34,
                                            "sell_probability": float(sell_prob_match.group(1)) if sell_prob_match else 0.33,
                                            "confidence_level": float(conf_match.group(1)) if conf_match else 0.5,
                                            "price_target": {
                                                "low": float(pt_low_match.group(1)) if pt_low_match else 0.0,
                                                "mid": float(pt_mid_match.group(1)) if pt_mid_match else 0.0,
                                                "high": float(pt_high_match.group(1)) if pt_high_match else 0.0
                                            },
                                            "price_target_1m": {
                                                "low": float(pt_1m_low_match.group(1)) if pt_1m_low_match else 0.0,
                                                "mid": float(pt_1m_mid_match.group(1)) if pt_1m_mid_match else 0.0,
                                                "high": float(pt_1m_high_match.group(1)) if pt_1m_high_match else 0.0
                                            },
                                            "price_target_timeframe": "12 months"
                                        }
                                    
                                    if "risk_factors" in sections:
                                        risk_str = sections["risk_factors"]
                                        if risk_str.startswith("[") and risk_str.endswith("]"):
                                            try:
                                                manual_json["risk_factors"] = json.loads(risk_str)
                                            except:
                                                items = re.findall(r'"([^"]+)"', risk_str)
                                                manual_json["risk_factors"] = items
                                    
                                    analysis = manual_json
                                except Exception as final_e:
                                    raise je
                
                required_fields = ["summary", "financial_health", "recommendation", "risk_factors", "technical_analysis", "market_sentiment"]
                missing_fields = [field for field in required_fields if field not in analysis]
                if missing_fields:
                    for field in missing_fields:
                        if field == "recommendation":
                            analysis["recommendation"] = {
                                "action": "hold",
                                "buy_probability": 0.33,
                                "hold_probability": 0.34,
                                "sell_probability": 0.33,
                                "confidence_level": 0.5,
                                "price_target_1m": {"low": 0.0, "mid": 0.0, "high": 0.0},
                                "price_target": {"low": 0.0, "mid": 0.0, "high": 0.0},
                                "price_target_timeframe": "12 months"
                            }
                        elif field == "risk_factors":
                            analysis["risk_factors"] = ["No risk factors provided"]
                        elif field in ["technical_analysis", "market_sentiment"]:
                            analysis[field] = {}
                        else:
                            analysis[field] = "No information provided"
                
                recommendation = analysis.get("recommendation", {})
                required_rec_fields = ["action", "buy_probability", "hold_probability", "sell_probability", "confidence_level", "price_target", "price_target_1m", "price_target_timeframe"]
                missing_rec_fields = [field for field in required_rec_fields if field not in recommendation]
                if missing_rec_fields:
                    for field in missing_rec_fields:
                        if field == "action":
                            recommendation["action"] = "hold"
                        elif field in ["buy_probability", "hold_probability", "sell_probability"]:
                            if "buy_probability" not in recommendation:
                                recommendation["buy_probability"] = 0.33
                            if "hold_probability" not in recommendation:
                                recommendation["hold_probability"] = 0.34
                            if "sell_probability" not in recommendation:
                                recommendation["sell_probability"] = 0.33
                        elif field == "confidence_level":
                            recommendation["confidence_level"] = 0.5
                        elif field == "price_target_timeframe":
                            recommendation["price_target_timeframe"] = "12 months"
                        elif field == "price_target_1m":
                            latest_price = None
                            if financial_data.get("price_data", {}).get("close"):
                                latest_price = financial_data["price_data"]["close"][-1]
                            
                            if latest_price:
                                recommendation["price_target_1m"] = {
                                    "low": round(latest_price * 0.95, 2),
                                    "mid": round(latest_price * 1.0, 2),
                                    "high": round(latest_price * 1.05, 2)
                                }
                            else:
                                recommendation["price_target_1m"] = {"low": 0.0, "mid": 0.0, "high": 0.0}
                        elif field == "price_target":
                            latest_price = None
                            if financial_data.get("price_data", {}).get("close"):
                                latest_price = financial_data["price_data"]["close"][-1]
                            
                            if latest_price:
                                recommendation["price_target"] = {
                                    "low": round(latest_price * 0.9, 2),
                                    "mid": round(latest_price * 1.0, 2),
                                    "high": round(latest_price * 1.1, 2)
                                }
                            else:
                                recommendation["price_target"] = {"low": 0.0, "mid": 0.0, "high": 0.0}
                
                price_target = recommendation.get("price_target", {})
                if not isinstance(price_target, dict) or not all(k in price_target for k in ["low", "mid", "high"]):
                    latest_price = None
                    if financial_data.get("price_data", {}).get("close"):
                        latest_price = financial_data["price_data"]["close"][-1]
                    
                    if latest_price:
                        recommendation["price_target"] = {
                            "low": round(latest_price * 0.9, 2),
                            "mid": round(latest_price * 1.0, 2),
                            "high": round(latest_price * 1.1, 2)
                        }
                    else:
                        recommendation["price_target"] = {"low": 0.0, "mid": 0.0, "high": 0.0}
                
                price_target_1m = recommendation.get("price_target_1m", {})
                if not isinstance(price_target_1m, dict) or not all(k in price_target_1m for k in ["low", "mid", "high"]):
                    latest_price = None
                    if financial_data.get("price_data", {}).get("close"):
                        latest_price = financial_data["price_data"]["close"][-1]
                    
                    if latest_price:
                        recommendation["price_target_1m"] = {
                            "low": round(latest_price * 0.95, 2),
                            "mid": round(latest_price * 1.0, 2),
                            "high": round(latest_price * 1.05, 2)
                        }
                    else:
                        recommendation["price_target_1m"] = {"low": 0.0, "mid": 0.0, "high": 0.0}
                
                probs = {
                    "buy": recommendation.get("buy_probability", 0),
                    "hold": recommendation.get("hold_probability", 0),
                    "sell": recommendation.get("sell_probability", 0)
                }
                total = sum(probs.values())
                if abs(total - 1.0) > 0.0001:
                    if total > 0:
                        for key in probs:
                            probs[key] /= total
                    else:
                        probs = {"buy": 0.33, "hold": 0.34, "sell": 0.33}
                    
                    recommendation["buy_probability"] = probs["buy"]
                    recommendation["hold_probability"] = probs["hold"]
                    recommendation["sell_probability"] = probs["sell"]
                
                if self.use_yfinance and financial_data.get("analyst_ratings"):
                    yf_ratings = financial_data["analyst_ratings"]
                    raw_recommendations = yf_ratings.get("raw_recommendations", {})
                    total_ratings = sum(raw_recommendations.values())
                    
                    if total_ratings > 0:
                        yf_probs = {
                            "strongBuy": raw_recommendations.get("strongBuy", 0) / total_ratings,
                            "buy": raw_recommendations.get("buy", 0) / total_ratings,
                            "hold": raw_recommendations.get("hold", 0) / total_ratings,
                            "sell": raw_recommendations.get("sell", 0) / total_ratings,
                            "strongSell": raw_recommendations.get("strongSell", 0) / total_ratings
                        }
                        
                        analysis["yfinance_comparison"] = {
                            "yfinance_probabilities": yf_probs,
                            "model_probabilities": {
                                "strongBuy": recommendation["buy_probability"] * 0.6,
                                "buy": recommendation["buy_probability"] * 0.4,
                                "hold": recommendation["hold_probability"],
                                "sell": recommendation["sell_probability"] * 0.4,
                                "strongSell": recommendation["sell_probability"] * 0.6
                            }
                        }
                
                # add some metadata
                analysis["ticker"] = ticker
                analysis["generated_at"] = datetime.now().isoformat()
                analysis["is_blind"] = is_blind_run

                if not is_blind_run and self.use_SEC and financial_data.get("filings"):
                     analysis["filing_date"] = financial_data["filings"][0]["filing_date"]
                     analysis["filings"] = financial_data["filings"]

                return analysis
                
            except json.JSONDecodeError as e:
                return {}
            except Exception as e:
                return {}

        except Exception as e:
            return {}

    def get_latest_insights(self, ticker: str) -> Dict:
        try:
            # just give the regular stuff
            return self.generate_insight(ticker)
            
        except Exception as e:
            return {} 