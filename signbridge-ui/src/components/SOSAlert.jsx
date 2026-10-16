import React, { useEffect, useRef } from 'react';

/**
 * SOSAlert adhering strictly to Section 4.8 and Section 6 of the design blueprint.
 * Full-viewport emergency overlay with accessible alertdialog role, auto-focus,
 * and immediate dismissal.
 */
export const SOSAlert = React.memo(function SOSAlert({
  active,
  onDismiss
}) {
  const dismissBtnRef = useRef(null);

  useEffect(() => {
    if (active && dismissBtnRef.current) {
      dismissBtnRef.current.focus();
    }
  }, [active]);

  if (!active) {
    return null;
  }

  return (
    <div
      className="sb-sos-overlay fade-in"
      role="alertdialog"
      aria-modal="true"
      aria-labelledby="sos-title"
      aria-describedby="sos-desc"
    >
      <div className="sb-sos-card">
        <h1 className="sb-sos-title" id="sos-title">
          EMERGENCY
        </h1>
        <p className="sb-sos-desc" id="sos-desc">
          HELP NEEDED
        </p>
        <button
          type="button"
          ref={dismissBtnRef}
          className="sb-sos-dismiss-btn"
          onClick={onDismiss}
          id="btn-dismiss-sos"
        >
          Dismiss Alert
        </button>
      </div>
    </div>
  );
});
