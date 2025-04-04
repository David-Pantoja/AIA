import React from "react";

const PortfolioSummary = ({ summary, riskAssessment, diversification }) => {
  if (!summary && !riskAssessment && !diversification) return null;

  return (
    <div className="result-section portfolio-summary">
      <h3>Overall Summary & Assessment</h3>
      {summary && (
        <>
          <h4>Summary:</h4>
          <p>{summary}</p>
        </>
      )}
      {riskAssessment && (
        <>
          <h4>Risk Assessment:</h4>
          <p>{riskAssessment}</p>
        </>
      )}
      {diversification && (
        <>
          <h4>Diversification Suggestions:</h4>
          <p>{diversification}</p>
        </>
      )}
    </div>
  );
};

export default PortfolioSummary;
