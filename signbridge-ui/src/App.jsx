import React, { useEffect, useState } from 'react';
import { useSignBridge } from './hooks/useSignBridge';
import { Header } from './components/Header';
import { CameraPanel } from './components/CameraPanel';
import { PredictionPanel } from './components/PredictionPanel';
import { WordBuilder } from './components/WordBuilder';
import { Suggestions } from './components/Suggestions';
import { ConversationLog } from './components/ConversationLog';
import { ControlBar } from './components/ControlBar';
import { SOSAlert } from './components/SOSAlert';
import './App.css';

export function App() {
  const {
    connected,
    prediction,
    confidence,
    handDetected,
    currentWord,
    currentLanguage,
    currentMode,
    conversationLog,
    sosActive,
    sosProgress,
    dismissSosAlert,
    sendFrame,
    addLetter,
    backspace,
    clearWord,
    setWord,
    speak,
    speechEnabled,
    toggleSpeech,
    switchLanguage,
    switchMode,
    saveLog,
    triggerSOS,
    deleteLogEntry,
    motionState,
    motionCountdown,
    motionFrameCount,
    motionResult,
    startMotionRecording,
    cancelMotionRecording,
  } = useSignBridge();

  const [mobileHistoryOpen, setMobileHistoryOpen] = useState(false);

  // Keyboard Shortcuts (Section 16)
  useEffect(() => {
    function handleKeyDown(event) {
      const tag = event.target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
        return;
      }
      if (event.repeat) {
        return;
      }

      // Space: Add current prediction
      if (event.code === 'Space') {
        event.preventDefault();
        addLetter();
      }
      // Delete / Backspace: Backspace
      else if (event.code === 'Backspace' || event.code === 'Delete') {
        event.preventDefault();
        backspace();
      }
      // Escape: Clear word
      else if (event.code === 'Escape') {
        event.preventDefault();
        clearWord();
      }
      // S / s: Speak
      else if (event.key === 's' || event.key === 'S') {
        event.preventDefault();
        speak();
      }
      // V / v: Toggle Speech ON/OFF
      else if (event.key === 'v' || event.key === 'V') {
        event.preventDefault();
        toggleSpeech();
      }
      // T / t: Toggle Language
      else if ((event.key === 't' || event.key === 'T') && currentMode === 'letters') {
        event.preventDefault();
        switchLanguage();
      }
      // P / p: Toggle Mode
      else if (event.key === 'p' || event.key === 'P') {
        event.preventDefault();
        switchMode();
      }
      // N / n: Toggle Motion Sign Recording
      else if (event.key === 'n' || event.key === 'N') {
        event.preventDefault();
        if (motionState === 'countdown' || motionState === 'recording') {
          cancelMotionRecording();
        } else {
          startMotionRecording();
        }
      }
    }

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [addLetter, backspace, clearWord, speak, toggleSpeech, switchLanguage, switchMode, currentMode, motionState, startMotionRecording, cancelMotionRecording]);

  return (
    <div className="sb-app-shell">
      {/* Fixed Workspace Header */}
      <Header
        connected={connected}
        currentLanguage={currentLanguage}
        currentMode={currentMode}
      />

      {/* Backend connection notice if waiting */}
      {!connected && (
        <div className="sb-reconnecting-banner" role="status">
          <span className="sb-spinner" aria-hidden="true" />
          <span>Connecting to SignBridge AI backend on localhost:8000…</span>
        </div>
      )}

      {/* Main Workspace: Asymmetric Two-Column Grid */}
      <main className="sb-workspace-body" id="main-content">
        {/* Left Column: Natural Camera Panel */}
        <div className="sb-column-camera">
          <CameraPanel
            connected={connected}
            handDetected={handDetected}
            currentLanguage={currentLanguage}
            currentMode={currentMode}
            sendFrame={sendFrame}
            motionState={motionState}
            motionCountdown={motionCountdown}
            motionFrameCount={motionFrameCount}
            motionResult={motionResult}
            sosProgress={sosProgress}
          />
        </div>

        {/* Right Column: Unified Translation Workspace */}
        <div className="sb-column-translation">
          <PredictionPanel
            prediction={prediction}
            confidence={confidence}
            handDetected={handDetected}
            currentMode={currentMode}
          />

          <WordBuilder
            currentWord={currentWord}
            addLetter={addLetter}
            backspace={backspace}
            clearWord={clearWord}
            prediction={prediction}
          />

          <Suggestions
            currentWord={currentWord}
            currentMode={currentMode}
            setWord={setWord}
          />

          {/* Collapsible toggle visible on narrow screens */}
          <button
            type="button"
            className="sb-mobile-history-toggle"
            onClick={() => setMobileHistoryOpen((prev) => !prev)}
            aria-expanded={mobileHistoryOpen}
          >
            History ({conversationLog.length} messages) {mobileHistoryOpen ? '▲' : '▼'}
          </button>

          <div className={`sb-history-wrapper ${mobileHistoryOpen ? 'mobile-open' : ''}`}>
            <ConversationLog
              conversationLog={conversationLog}
              deleteLogEntry={deleteLogEntry}
            />
          </div>
        </div>
      </main>

      {/* Bottom Fixed Control Bar */}
      <ControlBar
        currentWord={currentWord}
        currentLanguage={currentLanguage}
        currentMode={currentMode}
        speak={speak}
        speechEnabled={speechEnabled}
        toggleSpeech={toggleSpeech}
        saveLog={saveLog}
        switchLanguage={switchLanguage}
        switchMode={switchMode}
        triggerSOS={triggerSOS}
        motionState={motionState}
        startMotionRecording={startMotionRecording}
        cancelMotionRecording={cancelMotionRecording}
      />

      {/* Emergency Alert Overlay */}
      <SOSAlert
        active={sosActive}
        onDismiss={dismissSosAlert}
      />
    </div>
  );
}

export default App;
