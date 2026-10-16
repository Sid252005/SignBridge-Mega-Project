import React, { useState, useEffect, useRef } from 'react';

/**
 * PredictionPanel adhering strictly to Section 4.3 and 5.1 of the design blueprint.
 * Displays detected letter or word hero, animated transition on change, and smooth confidence bar.
 */
export const PredictionPanel = React.memo(function PredictionPanel({
  prediction,
  confidence,
  handDetected,
  currentMode
}) {
  const [isPopping, setIsPopping] = useState(false);
  const prevPredRef = useRef(prediction);

  // Trigger brief scale pop animation when prediction changes
  useEffect(() => {
    if (prediction !== prevPredRef.current && prediction !== '—' && prediction !== '') {
      setIsPopping(true);
      const timer = setTimeout(() => setIsPopping(false), 150);
      prevPredRef.current = prediction;
      return () => clearTimeout(timer);
    }
    prevPredRef.current = prediction;
  }, [prediction]);

  const hasDetection = handDetected && prediction && prediction !== '—';
  const displayVal = hasDetection ? prediction : '—';
  const displayConfidence = hasDetection ? `${Math.round(confidence)}%` : '—';
  const barWidth = hasDetection ? Math.min(Math.max(confidence, 0), 100) : 0;
  const isLettersMode = currentMode === 'letters';
  const confTierClass = confidence >= 70 ? 'high' : (confidence >= 40 ? 'mid' : 'low');

  return (
    <section className="sb-prediction-panel" aria-labelledby="prediction-heading">
      <div className="sb-section-label" id="prediction-heading">
        {isLettersMode ? 'Detected Letter' : 'Detected Word'}
      </div>

      {/* Hero Prediction Display */}
      <div
        className={`sb-prediction-value ${isLettersMode ? 'letters' : 'words'} ${hasDetection ? 'active' : 'empty'} ${isPopping ? 'prediction-pop' : ''}`}
        aria-live="polite"
        aria-atomic="true"
      >
        {displayVal}
      </div>

      {/* Confidence Bar & Percentage */}
      <div className="sb-confidence-row" aria-label={`Confidence: ${displayConfidence}`}>
        <div className="sb-confidence-track">
          <div
            className={`sb-confidence-fill ${hasDetection ? confTierClass : ''}`}
            style={{ width: `${barWidth}%` }}
          />
        </div>
        <span className="sb-confidence-text">
          {displayConfidence}
        </span>
      </div>
    </section>
  );
});
