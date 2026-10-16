import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';

/**
 * CameraPanel adhering strictly to Section 4.2 and Section 9 of the design blueprint.
 * Clean, distraction-free video feed with live frame capture loop, status overlay bar,
 * and integrated countdown/recording states.
 */
export const CameraPanel = React.memo(function CameraPanel({
  connected,
  handDetected,
  currentLanguage,
  currentMode,
  sendFrame,
  motionState,
  motionCountdown,
  motionFrameCount,
  motionResult = '',
  sosProgress = 0
}) {
  const webcamRef = useRef(null);
  const [cameraError, setCameraError] = useState(null);

  // Dynamic frame capture loop: 40ms (~25 FPS) during motion recording, 100ms (10 FPS) normally
  useEffect(() => {
    if (!connected) return;

    const intervalMs = motionState === 'recording' ? 40 : 100;
    const intervalId = setInterval(() => {
      if (document.hidden) return;
      // Do not send frames during motion countdown
      if (motionState === 'countdown') return;

      if (webcamRef.current) {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
          sendFrame(imageSrc);
        }
      }
    }, intervalMs);

    return () => clearInterval(intervalId);
  }, [connected, sendFrame, motionState]);

  // Compute bottom overlay text
  let modeOverlayText = 'ASL Letters';
  if (currentMode === 'words') {
    modeOverlayText = 'Words';
  } else if (currentLanguage === 'ISL') {
    modeOverlayText = 'ISL Letters';
  }

  const isRecording = motionState === 'recording';
  const isCountdown = motionState === 'countdown';

  return (
    <div
      className={`sb-camera-container ${isRecording ? 'recording' : ''}`}
      aria-label="Live camera workspace"
    >
      {cameraError ? (
        <div className="sb-camera-error">
          <p className="sb-camera-error-title">Camera Unavailable</p>
          <p className="sb-camera-error-subtitle">
            {cameraError.name === 'NotAllowedError'
              ? 'Camera access required. Please grant permission in your browser.'
              : 'No camera detected or device is in use.'}
          </p>
        </div>
      ) : (
        <Webcam
          ref={webcamRef}
          audio={false}
          mirrored={true}
          screenshotFormat="image/jpeg"
          screenshotQuality={0.6}
          className="sb-camera-video"
          videoConstraints={{
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user'
          }}
          onUserMediaError={(err) => {
            console.error('[Webcam Error]:', err);
            setCameraError(err);
          }}
        />
      )}

      {/* SOS Progress HUD Badge */}
      {sosProgress > 0 && (
        <div className="sb-sos-progress-badge" role="status" aria-live="assertive">
          <span className="sb-sos-badge-icon" aria-hidden="true">⚠️</span>
          <span className="sb-sos-badge-text">
            SOS Trigger: <strong>{sosProgress}/3</strong> — Sign &apos;S&apos; {3 - sosProgress} more time{3 - sosProgress > 1 ? 's' : ''}
          </span>
        </div>
      )}

      {/* Countdown Overlay (3 -> 2 -> 1 -> GO) */}
      {isCountdown && (
        <div className="sb-countdown-overlay" aria-live="assertive">
          <span className="sb-countdown-number" key={motionCountdown}>
            {motionCountdown > 0 ? motionCountdown : 'GO'}
          </span>
        </div>
      )}

      {/* Bottom Telemetry Overlay Bar */}
      <div className="sb-camera-overlay-bar">
        <div className="sb-camera-status-left">
          <span
            className={`sb-status-dot ${handDetected ? 'connected' : 'disconnected'}`}
            style={{ backgroundColor: handDetected ? 'var(--sb-success)' : 'var(--sb-text-tertiary)' }}
            aria-hidden="true"
          />
          <span className="sb-camera-status-text">
            {handDetected ? 'Hand detected' : 'No hand detected'}
          </span>
        </div>

        {isRecording ? (
          <div className="sb-camera-status-center recording-badge">
            Recording {motionFrameCount}/30
          </div>
        ) : motionState === 'processing' ? (
          <div className="sb-camera-status-center processing-badge">
            Processing sequence…
          </div>
        ) : motionState === 'result' ? (
          <div className={`sb-camera-status-center ${motionResult === 'Not recognized' || motionResult === 'Not enough frames' ? 'error-badge' : 'result-badge'}`}>
            {motionResult === 'Not recognized' || motionResult === 'Not enough frames' ? `⚠️ ${motionResult} — try again` : `✓ ${motionResult}`}
          </div>
        ) : null}

        <div className="sb-camera-status-right">
          <span className="sb-camera-mode-text">{modeOverlayText}</span>
        </div>
      </div>
    </div>
  );
});
