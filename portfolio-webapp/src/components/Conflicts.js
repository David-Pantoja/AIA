import React from "react";

const Conflicts = ({ conflictsData }) => {
  if (!conflictsData || !conflictsData.has_conflicts) return null;

  return (
    <div className="result-section conflicts">
      <h3 className="warning">Potential Conflicts Detected!</h3>
      {conflictsData.conflicts && conflictsData.conflicts.length > 0 ? (
        <ul>
          {conflictsData.conflicts.map((conflict, index) => (
            <li key={index}>
              <strong>Positions:</strong> {conflict.positions.join(", ")}
              <br />
              <strong>Reason:</strong> {conflict.reason}
              <br />
              <strong>Explanation:</strong> {conflict.explanation}
            </li>
          ))}
        </ul>
      ) : (
        <p>Conflict data is present but no specific conflicts listed.</p>
      )}
      {/* <p><em>Analyzed At: {new Date(conflictsData.analyzed_at).toLocaleString()}</em></p> */}
    </div>
  );
};

export default Conflicts;
