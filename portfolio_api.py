#!/usr/bin/env python3

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import logging
import json
import tempfile
import os
from portfolio_analyzer import analyze_portfolio

app = Flask(__name__)
CORS(app)
# just some basic logging setup
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        portfolio_data = request.json
        if not portfolio_data:
            return jsonify({"error": "No portfolio data provided"}), 400
        
        if "positions" not in portfolio_data or not isinstance(portfolio_data["positions"], list):
            return jsonify({"error": "Invalid portfolio format: 'positions' array is required"}), 400
        
        use_yfinance = portfolio_data.get("config", {}).get("use_yfinance", True)
        use_SEC = portfolio_data.get("config", {}).get("use_SEC", True)
        quarters = portfolio_data.get("config", {}).get("quarters", 4)
        max_search = portfolio_data.get("config", {}).get("max_search", 200)
        cutoff_date_str = portfolio_data.get("date", "current")

        cutoff_date = None
        if cutoff_date_str == "current":
            cutoff_date = datetime.now()
        else:
            cutoff_date = datetime.strptime(cutoff_date_str, "%Y-%m-%d")
        
        logger.info(f"Analyzing portfolio with {len(portfolio_data['positions'])} positions")
        
        # gotta make a temp file real quick
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(portfolio_data, temp_file)
            temp_file_path = temp_file.name
        
        try:
            results = analyze_portfolio(
                temp_file_path, 
                cutoff_date=cutoff_date,
                use_yfinance=use_yfinance, 
                use_SEC=use_SEC,
                quarters=quarters, 
                max_search=max_search
            )
        finally:
            # cleanup 
            os.unlink(temp_file_path)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error analyzing portfolio: {str(e)}")
        return jsonify({"error": f"Failed to analyze portfolio: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

@app.route('/api/documentation', methods=['GET'])
def get_documentation():
    return jsonify({
        "name": "Portfolio Analyzer API",
        "version": "1.0.0",
        "description": "API for analyzing investment portfolios",
        "endpoints": [
            {
                "path": "/api/analyze",
                "method": "POST",
                "description": "Analyze an investment portfolio",
                "request_body": "JSON portfolio data",
                "responses": {
                    "200": "Analysis results",
                    "400": "Invalid request",
                    "500": "Server error"
                }
            },
            {
                "path": "/api/health",
                "method": "GET",
                "description": "Health check endpoint",
                "responses": {
                    "200": "API status information"
                }
            }
        ]
    })

if __name__ == "__main__":
    #weird port requirement on my machine
    app.run(host='0.0.0.0', port=8000)