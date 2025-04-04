import React from "react";

const TechnicalAnalysis = ({ data }) => {
  if (!data) return null;

  const { trend, momentum, support_resistance } = data;

  // safely render potentially missing values or objects
  const renderValue = (value, label) => {
    if (value === null || value === undefined || value === "") return null;
    if (typeof value === "object" && Object.keys(value).length === 0)
      return null;

    let displayValue;
    if (typeof value === "object") {
      if (
        label === "Support/Resistance" &&
        value._support_levels_ &&
        value._resistance_levels_
      ) {
        displayValue = (
          <>
            Support: [{value._support_levels_.join(", ")}] | Resistance: [
            {value._resistance_levels_.join(", ")}]
          </>
        );
      } else {
        displayValue = JSON.stringify(value);
      }
    } else {
      displayValue = value;
    }

    return (
      <p>
        <strong>{label}:</strong> {displayValue}
      </p>
    );
  };

  const hasData =
    trend ||
    momentum ||
    (support_resistance &&
      typeof support_resistance === "object" &&
      Object.keys(support_resistance).length > 0) ||
    (typeof support_resistance === "string" && support_resistance !== "");

  if (!hasData) {
    return (
      <details className="insight-subsection technical-analysis-details">
        <summary>Technical Analysis</summary>
        <p>N/A</p>
      </details>
    );
  }

  return (
    <details className="insight-subsection technical-analysis-details">
      <summary>Technical Analysis</summary>
      {renderValue(trend, "Trend")}
      {renderValue(momentum, "Momentum")}
      {renderValue(support_resistance, "Support/Resistance")}
    </details>
  );
};

export default TechnicalAnalysis;
