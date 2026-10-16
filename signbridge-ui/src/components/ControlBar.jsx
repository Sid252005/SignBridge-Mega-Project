import React from 'react';

/**
 * ControlBar adhering strictly to Section 4.7 of the design blueprint.
 * Bottom-anchored toolbar with action groups: Voice/Persistence, Mode/Language toggles,
 * motion recording trigger, and right-aligned SOS emergency trigger.
 */
export const ControlBar = React.memo(function ControlBar({
  currentWord,
  currentLanguage,
  currentMode,
  speak,
  speechEnabled,
  toggleSpeech,
  saveLog,
  switchLanguage,
  switchMode,
  triggerSOS,
  motionState,
  startMotionRecording,
  cancelMotionRecording
}) {
  const isLetters = currentMode === 'letters';
  const langToggleLabel = currentLanguage === 'ASL' ? 'ASL → ISL' : 'ISL → ASL';
  const modeToggleLabel = isLetters ? 'Switch to Words' : 'Switch to Letters';

  const isMotionActive = motionState === 'countdown' || motionState === 'recording';
  const motionBtnLabel = isMotionActive
    ? 'Cancel Recording'
    : (isLetters ? 'Record J/Z' : 'Record Motion Sign');

  return (
    <footer className="sb-control-bar" role="toolbar" aria-label="Translation controls">
      {/* Group 1: Voice & Persistence */}
      <div className="sb-control-group">
        <button
          type="button"
          className="sb-btn sb-btn-primary"
          onClick={speak}
          disabled={!currentWord}
          id="btn-speak"
          aria-label="Speak current phrase"
        >
          Speak
        </button>

        <button
          type="button"
          className={`sb-btn ${speechEnabled ? 'sb-btn-speech-active' : 'sb-btn-default'}`}
          onClick={toggleSpeech}
          id="btn-toggle-speech"
          aria-label={`Toggle automatic speech. Currently ${speechEnabled ? 'enabled' : 'disabled'}`}
          title={speechEnabled ? 'Automatic speech is ON (speaks recognized signs)' : 'Automatic speech is OFF'}
        >
          {speechEnabled ? '🔊 Speech ON' : '🔇 Speech OFF'}
        </button>

        <button
          type="button"
          className="sb-btn sb-btn-default"
          onClick={saveLog}
          id="btn-save-log"
          aria-label="Save conversation history"
        >
          Save
        </button>
      </div>

      <div className="sb-control-divider" aria-hidden="true" />

      {/* Group 2: Mode, Language & Motion Capture */}
      <div className="sb-control-group">
        <button
          type="button"
          className="sb-btn sb-btn-default"
          onClick={switchLanguage}
          disabled={!isLetters}
          id="btn-toggle-language"
          aria-label={`Toggle language. Currently ${currentLanguage}`}
          title={!isLetters ? 'Language switching is only available in Letters mode' : undefined}
        >
          {langToggleLabel}
        </button>

        <button
          type="button"
          className="sb-btn sb-btn-default"
          onClick={switchMode}
          id="btn-toggle-mode"
          aria-label={`Toggle mode. Currently ${currentMode}`}
        >
          {modeToggleLabel}
        </button>

        {/* Motion Sign trigger (enabled in both letters and words mode) */}
        <button
          type="button"
          className={`sb-btn ${isMotionActive ? 'sb-btn-danger' : 'sb-btn-default'}`}
          onClick={isMotionActive ? cancelMotionRecording : startMotionRecording}
          id="btn-motion-record"
          aria-label={motionBtnLabel}
        >
          {motionBtnLabel}
        </button>
      </div>

      <div className="sb-control-spacer" />

      {/* Group 3: Emergency Action */}
      <div className="sb-control-group right">
        <button
          type="button"
          className="sb-btn sb-btn-danger sb-btn-sos"
          onClick={triggerSOS}
          id="btn-sos"
          aria-label="Trigger Emergency SOS Alert"
        >
          SOS
        </button>
      </div>
    </footer>
  );
});
