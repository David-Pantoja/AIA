import React from "react";

const RebalancingRecommendations = ({ recommendations }) => {
  if (!recommendations || recommendations.length === 0) return null;

  return (
    <div className="result-section recommendations">
      <h3>Rebalancing Recommendations</h3>
      <ul>
        {recommendations.map((rec, index) => (
          <li key={index} className={`action-${rec.action?.toLowerCase()}`}>
            <strong>{rec.ticker}:</strong>
            <span className="action">{rec.action?.toUpperCase()}</span>
            <p className="reason">{rec.reason}</p>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default RebalancingRecommendations;
