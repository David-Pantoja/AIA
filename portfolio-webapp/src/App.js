import React, { useState } from "react";
import "./App.css";
// Import the new components
import PortfolioSummary from "./components/PortfolioSummary";
import PortfolioDetails from "./components/PortfolioDetails";
import Conflicts from "./components/Conflicts";
import RebalancingRecommendations from "./components/RebalancingRecommendations";
import Insights from "./components/Insights";

function App() {
  // State for top-level portfolio info
  const [portfolioName, setPortfolioName] = useState("My Portfolio");
  const [portfolioOwner, setPortfolioOwner] = useState("User");
  const [portfolioDateType, setPortfolioDateType] = useState("current"); // 'current' or 'specific'
  const [specificPortfolioDate, setSpecificPortfolioDate] =
    useState("2024-08-01");

  // State for config
  const [configQuarters, setConfigQuarters] = useState(4);
  const [configMaxSearch, setConfigMaxSearch] = useState(200);
  const [configUseSEC, setConfigUseSEC] = useState(true);
  const [configUseYFinance, setConfigUseYFinance] = useState(true);

  // State for positions (array of objects)
  const [positions, setPositions] = useState([
    { id: 1, ticker: "AAPL", shares: 100, cost_basis: 150.75 },
    { id: 2, ticker: "CRWD", shares: 50, cost_basis: 225.4 },
  ]);
  const [nextPositionId, setNextPositionId] = useState(3);

  // State for API interaction
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Handlers for position changes
  const handlePositionChange = (id, field, value) => {
    setPositions((prevPositions) =>
      prevPositions.map((pos) =>
        pos.id === id ? { ...pos, [field]: value } : pos
      )
    );
  };

  const handleAddPosition = () => {
    setPositions((prevPositions) => [
      ...prevPositions,
      { id: nextPositionId, ticker: "", shares: 0, cost_basis: 0 },
    ]);
    setNextPositionId((prevId) => prevId + 1);
  };

  const handleRemovePosition = (id) => {
    setPositions((prevPositions) =>
      prevPositions.filter((pos) => pos.id !== id)
    );
  };

  const analyzePortfolio = async () => {
    setLoading(true);
    setError(null);
    setAnalysisResult(null);

    const finalPortfolioDate =
      portfolioDateType === "current" ? "current" : specificPortfolioDate;

    const portfolioData = {
      name: portfolioName,
      owner: portfolioOwner,
      date: finalPortfolioDate,
      config: {
        quarters: parseInt(configQuarters, 10) || 4,
        max_search: parseInt(configMaxSearch, 10) || 200,
        use_SEC: configUseSEC,
        use_yfinance: configUseYFinance,
      },
      positions: positions.map(({ id, ...rest }) => ({
        ...rest,
        shares: parseFloat(rest.shares) || 0,
        cost_basis: parseFloat(rest.cost_basis) || 0,
      })),
    };

    try {
      const apiUrl = "http://localhost:8000/api/analyze";
      console.log(JSON.stringify(portfolioData, null, 2));
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(portfolioData),
      });

      if (!response.ok) {
        throw new Error("Analysis request failed");
      }

      const data = await response.json();
      console.log(JSON.stringify(data, null, 2));
      setAnalysisResult(data);
    } catch (err) {
      setError(err.message);
      console.error("Error analyzing portfolio:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Portfolio Analyzer</h1>
      </header>
      <main className="App-main">
        <form
          className="portfolio-form"
          onSubmit={(e) => {
            e.preventDefault();
            analyzePortfolio();
          }}
        >
          <h2>Portfolio Details</h2>
          <div className="form-section">
            <div className="form-group">
              <label htmlFor="portfolio-name">Name:</label>
              <input
                type="text"
                id="portfolio-name"
                value={portfolioName}
                onChange={(e) => setPortfolioName(e.target.value)}
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="portfolio-owner">Owner:</label>
              <input
                type="text"
                id="portfolio-owner"
                value={portfolioOwner}
                onChange={(e) => setPortfolioOwner(e.target.value)}
                required
              />
            </div>
            <div className="form-group date-selection-group">
              <label>Portfolio Date:</label>
              <div className="radio-group">
                <label>
                  <input
                    type="radio"
                    name="portfolioDateType"
                    value="current"
                    checked={portfolioDateType === "current"}
                    onChange={(e) => setPortfolioDateType(e.target.value)}
                  />
                  Current
                </label>
                <label>
                  <input
                    type="radio"
                    name="portfolioDateType"
                    value="specific"
                    checked={portfolioDateType === "specific"}
                    onChange={(e) => setPortfolioDateType(e.target.value)}
                  />
                  Specific Date:
                </label>
              </div>
              {portfolioDateType === "specific" && (
                <input
                  type="date"
                  id="portfolio-date"
                  value={specificPortfolioDate}
                  onChange={(e) => setSpecificPortfolioDate(e.target.value)}
                  required
                  className="specific-date-input"
                />
              )}
              {portfolioDateType === "specific" && (
                <div className="date-note">
                  It is recommended you use a date after 07/18/2024 as the
                  reasoning model may have price knowledge baked in for days
                  before that.
                </div>
              )}
            </div>
          </div>

          <h2>Configuration</h2>
          <div className="form-section">
            <div className="form-group">
              <label htmlFor="config-quarters">Quarters:</label>
              <input
                type="number"
                id="config-quarters"
                value={configQuarters}
                onChange={(e) => setConfigQuarters(e.target.value)}
                min="1"
              />
            </div>
            {/*
            <div className="form-group">
              <label htmlFor="config-max-search">Max Search:</label>
              <input
                type="number"
                id="config-max-search"
                value={configMaxSearch}
                onChange={(e) => setConfigMaxSearch(e.target.value)}
                min="1"
              />
            </div>
            */}
            <div className="form-group checkbox-group">
              <label htmlFor="config-use-sec">
                <input
                  type="checkbox"
                  id="config-use-sec"
                  checked={configUseSEC}
                  onChange={(e) => setConfigUseSEC(e.target.checked)}
                />
                Use SEC
              </label>
            </div>
            <div className="form-group checkbox-group">
              <label htmlFor="config-use-yfinance">
                <input
                  type="checkbox"
                  id="config-use-yfinance"
                  checked={configUseYFinance}
                  onChange={(e) => setConfigUseYFinance(e.target.checked)}
                />
                Use yfinance
              </label>
            </div>
          </div>

          <h2>Positions</h2>
          <div className="form-section positions-list">
            {positions.map((pos, index) => (
              <div key={pos.id} className="position-item">
                <span>#{index + 1}</span>
                <input
                  type="text"
                  placeholder="Ticker"
                  value={pos.ticker}
                  onChange={(e) =>
                    handlePositionChange(pos.id, "ticker", e.target.value)
                  }
                  required
                />
                <input
                  type="number"
                  placeholder="Shares"
                  value={pos.shares}
                  onChange={(e) =>
                    handlePositionChange(pos.id, "shares", e.target.value)
                  }
                  min="0"
                  step="any"
                  required
                />
                <input
                  type="number"
                  placeholder="Cost Basis"
                  value={pos.cost_basis}
                  onChange={(e) =>
                    handlePositionChange(pos.id, "cost_basis", e.target.value)
                  }
                  min="0"
                  step="any"
                  required
                />
                <button
                  type="button"
                  onClick={() => handleRemovePosition(pos.id)}
                  className="remove-btn"
                >
                  Remove
                </button>
              </div>
            ))}
            <button
              type="button"
              onClick={handleAddPosition}
              className="add-btn"
            >
              + Add Position
            </button>
          </div>

          <div className="controls">
            <button type="submit" disabled={loading}>
              {loading ? "Analyzing..." : "Analyze Portfolio"}
            </button>
          </div>
        </form>

        {error && <div className="error">Error: {error}</div>}

        {analysisResult && (
          <div className="results-display">
            <h2>Analysis Results for {portfolioName}</h2>

            {/* Use the new components */}
            <PortfolioSummary
              summary={analysisResult.summary}
              riskAssessment={analysisResult.risk_assessment}
              diversification={analysisResult.diversification_suggestions}
            />
            <Conflicts conflictsData={analysisResult.conflicts} />
            <RebalancingRecommendations
              recommendations={analysisResult.rebalancing_recommendations}
            />
            <Insights insightsData={analysisResult.insights} />
            <PortfolioDetails portfolio={analysisResult.portfolio} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
