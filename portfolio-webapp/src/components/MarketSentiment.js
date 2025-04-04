import React from "react";

const MarketSentiment = ({ data }) => {
  if (!data) return null;

  // Function to safely render potentially missing values
  const renderValue = (value, label) => {
    // Clean up label keys like _analyst_consensus_
    const cleanLabel = label.replace(/_/g, " ").trim();
    const formattedLabel =
      cleanLabel.charAt(0).toUpperCase() + cleanLabel.slice(1);

    if (value === null || value === undefined || value === "") return null;
    if (typeof value === "object" && Object.keys(value).length === 0)
      return null; // Ignore empty objects

    return (
      <p>
        <strong>{formattedLabel}:</strong>{" "}
        {typeof value === "object" ? JSON.stringify(value) : value}
      </p>
    );
  };

  const keys = Object.keys(data);

  // Check if there is any data to display (excluding empty objects)
  const hasData = keys.some((key) => {
    const value = data[key];
    return !(
      value === null ||
      value === undefined ||
      value === "" ||
      (typeof value === "object" && Object.keys(value).length === 0)
    );
  });

  if (!hasData) {
    return (
      <details className="insight-subsection market-sentiment-details">
        <summary>Market Sentiment</summary>
        <p>N/A</p>
      </details>
    );
  }

  return (
    <details className="insight-subsection market-sentiment-details">
      <summary>Market Sentiment</summary>
      {keys.map((key) => (
        <React.Fragment key={key}>{renderValue(data[key], key)}</React.Fragment>
      ))}
    </details>
  );
};

export default MarketSentiment;
