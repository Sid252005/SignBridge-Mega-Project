import React from 'react';

/**
 * Header component adhering strictly to Section 4.1 of the design blueprint.
 * Clean, composed workspace header with brand mark, active mode indicator, and connection status.
 */
export const Header = React.memo(function Header({
  connected,
  currentLanguage,
  currentMode
}) {
  // Compute mode pill text
  let modeLabel = 'ASL · Letters';
  if (currentMode === 'words') {
    modeLabel = 'Common Words';
  } else if (currentLanguage === 'ISL') {
    modeLabel = 'ISL · Letters';
  }

  return (
    <header className="sb-header" role="banner">
      {/* Left: Brand Identity */}
      <div className="sb-header-brand">
        <span className="sb-header-mark" aria-hidden="true">✦</span>
        <span className="sb-header-title">
          SignBridge <span className="sb-header-ai">AI</span>
        </span>
      </div>

      {/* Center: Current Workspace Mode */}
      <div className="sb-header-center">
        <span className="sb-pill" id="header-mode-indicator">
          {modeLabel}
        </span>
      </div>

      {/* Right: Live Connection Telemetry */}
      <div className="sb-header-status">
        <span
          className={`sb-status-dot ${connected ? 'connected' : 'disconnected'}`}
          aria-hidden="true"
        />
        <span className="sb-header-status-text">
          {connected ? 'Connected' : 'Disconnected'}
        </span>
      </div>
    </header>
  );
});
