import React from "react";

const Recommendation = ({ data }) => {
  if (!data) return null;

  const {
    action,
    buy_probability,
    hold_probability,
    sell_probability,
    confidence_level,
    price_target,
    price_target_1m,
    price_target_timeframe,
  } = data;

  const formatProbability = (prob) => (prob * 100).toFixed(0) + "%";
  const formatPrice = (price) => price?.toFixed(2);

  return (
    <details className="insight-subsection recommendation-details">
      <summary>Recommendation</summary>
      <p className={`action action-${action?.toLowerCase()}`}>
        <strong>Action: {action?.toUpperCase() || "N/A"}</strong> (Confidence:{" "}
        {confidence_level ? formatProbability(confidence_level) : "N/A"})
      </p>
      <div className="probabilities">
        <span>
          Buy: {buy_probability ? formatProbability(buy_probability) : "N/A"}
        </span>
        <span>
          Hold: {hold_probability ? formatProbability(hold_probability) : "N/A"}
        </span>
        <span>
          Sell: {sell_probability ? formatProbability(sell_probability) : "N/A"}
        </span>
      </div>
      {price_target && (
        <div className="price-target">
          <strong>{price_target_timeframe || "Price Target"}:</strong>
          <span> Low: ${formatPrice(price_target.low) || "N/A"}</span>
          <span> Mid: ${formatPrice(price_target.mid) || "N/A"}</span>
          <span> High: ${formatPrice(price_target.high) || "N/A"}</span>
        </div>
      )}
      {price_target_1m && (
        <div className="price-target">
          <strong>1 Month Target:</strong>
          <span> Low: ${formatPrice(price_target_1m.low) || "N/A"}</span>
          <span> Mid: ${formatPrice(price_target_1m.mid) || "N/A"}</span>
          <span> High: ${formatPrice(price_target_1m.high) || "N/A"}</span>
        </div>
      )}
    </details>
  );
};

export default Recommendation;
