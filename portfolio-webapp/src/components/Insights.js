import React from "react";
import InsightCard from "./InsightCard";

const Insights = ({ insightsData }) => {
  if (!insightsData || Object.keys(insightsData).length === 0) return null;

  const tickers = Object.keys(insightsData);

  return (
    <div className="result-section insights-container">
      <h3>Insights per Ticker</h3>
      <div className="insights-grid">
        {tickers.map((ticker) => (
          <InsightCard key={ticker} insightData={insightsData[ticker]} />
        ))}
      </div>
    </div>
  );
};

export default Insights;
