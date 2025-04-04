import React from "react";
import Recommendation from "./Recommendation";
import TechnicalAnalysis from "./TechnicalAnalysis";
import MarketSentiment from "./MarketSentiment";

const InsightCard = ({ insightData }) => {
  if (!insightData) return null;

  const { ticker, filing_date, generated_at, insights } = insightData;

  // format optional date
  const formatDate = (dateString) => {
    if (!dateString) return "N/A";
    try {
      return new Date(dateString).toLocaleDateString();
    } catch (e) {
      return dateString;
    }
  };

  // render list items
  const renderList = (items) => {
    if (!Array.isArray(items) || items.length === 0) return <p>N/A</p>;
    return (
      <ul>
        {items.map((item, index) => (
          <li key={index}>
            {typeof item === "object" ? JSON.stringify(item) : item}
          </li>
        ))}
      </ul>
    );
  };

  return (
    <div className="insight-card">
      <h4>{ticker}</h4>
      <div className="insight-meta">
        <span>Generated: {formatDate(generated_at)}</span>
        {filing_date && <span> | Last Filing: {formatDate(filing_date)}</span>}
      </div>

      {insights ? (
        <div className="insight-details">
          {insights.summary && (
            <details open>
              {" "}
              <summary>Summary</summary>
              <p>{insights.summary}</p>
            </details>
          )}
          {insights.financial_health && (
            <details>
              <summary>Financial Health</summary>
              <p>{insights.financial_health}</p>
            </details>
          )}

          {insights.recommendation && (
            <Recommendation data={insights.recommendation} />
          )}
          {insights.technical_analysis && (
            <TechnicalAnalysis data={insights.technical_analysis} />
          )}
          {insights.market_sentiment && (
            <MarketSentiment data={insights.market_sentiment} />
          )}
          {insights.risk_factors && (
            <details>
              <summary>Risk Factors</summary>
              {insights.risk_factors.map((factor, index) => (
                <div key={index}>{factor}</div>
              ))}
            </details>
          )}
        </div>
      ) : (
        <p>No detailed insights available.</p>
      )}
    </div>
  );
};

export default InsightCard;
