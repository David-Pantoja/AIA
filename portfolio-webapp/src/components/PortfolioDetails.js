import React from "react";

const PortfolioDetails = ({ portfolio }) => {
  if (!portfolio) return null;

  const formatDate = (dateString) => {
    try {
      return new Date(dateString).toLocaleDateString();
    } catch (e) {
      return dateString;
    }
  };

  return (
    <div className="result-section portfolio-details">
      <h3>Original Portfolio Input</h3>
      <p>
        <strong>Cutoff Date Used:</strong> {formatDate(portfolio.cutoff_date)}
      </p>
      <h4>Configuration Used:</h4>
      <ul>
        <li>Quarters: {portfolio.config.quarters}</li>
        <li>Use SEC Filings: {portfolio.config.use_SEC ? "Yes" : "No"}</li>
        <li>Use yfinance: {portfolio.config.use_yfinance ? "Yes" : "No"}</li>
      </ul>
      <h4>Positions Analyzed:</h4>
      {portfolio.positions && portfolio.positions.length > 0 ? (
        <ul>
          {portfolio.positions.map((pos, index) => (
            <li key={index}>
              {pos.ticker} ({pos.shares} shares @ ${pos.cost_basis?.toFixed(2)})
            </li>
          ))}
        </ul>
      ) : (
        <p>No positions data available.</p>
      )}
    </div>
  );
};

export default PortfolioDetails;
