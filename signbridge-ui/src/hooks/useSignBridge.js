import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Text-to-Speech helper using browser Web Speech API
 */
export function speakText(text) {
  if (!text || typeof window === 'undefined' || !('speechSynthesis' in window)) {
    return;
  }
  try {
    const synth = window.speechSynthesis;
    if (synth.paused) {
      synth.resume();
    }
    synth.cancel();

    // Defer utterance slightly so cancel() completes cleanly in Chromium
    setTimeout(() => {
      try {
        if (synth.paused) {
          synth.resume();
        }
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.95;
        utterance.pitch = 1.0;
        utterance.lang = 'en-US';

        const voices = synth.getVoices();
        if (voices && voices.length > 0) {
          const enVoice = voices.find(v => v.lang.startsWith('en') && !v.localService) || voices.find(v => v.lang.startsWith('en'));
          if (enVoice) utterance.voice = enVoice;
        }

        synth.speak(utterance);
      } catch (err) {
        console.warn('[TTS] Utterance error:', err);
      }
    }, 20);
  } catch (err) {
    console.warn('[TTS] Synthesis error:', err);
  }
}

/**
 * Primary custom hook managing WebSocket connection, frame dispatch,
 * application state, prediction throttling, and motion-word recording state machine.
 */
export function useSignBridge() {
  // Connection state
  const [connected, setConnected] = useState(false);

  // Throttled prediction state (updated at ~5 Hz to keep renders calm)
  const [prediction, setPrediction] = useState('—');
  const [confidence, setConfidence] = useState(0);
  const [handDetected, setHandDetected] = useState(false);

  // Application state (from backend "state" messages)
  const [currentWord, setCurrentWord] = useState('');
  const [currentLanguage, setCurrentLanguage] = useState('ASL');
  const [currentMode, setCurrentMode] = useState('letters');
  const [conversationLog, setConversationLog] = useState([]);
  const [sosActive, setSosActive] = useState(false);
  const [sosProgress, setSosProgress] = useState(0);

  // Automatic speech toggle and stability state
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const speechEnabledRef = useRef(true);
  const currentModeRef = useRef('letters');
  const stablePredictionRef = useRef('—');
  const stableCountRef = useRef(0);
  const lastSpokenRef = useRef('');
  const speechCooldownRef = useRef(false);
  const speechCooldownTimerRef = useRef(null);

  // Keep speechEnabledRef synced
  useEffect(() => {
    speechEnabledRef.current = speechEnabled;
  }, [speechEnabled]);

  // Keep currentModeRef synced & reset stability on mode switch
  useEffect(() => {
    currentModeRef.current = currentMode;
    stablePredictionRef.current = '—';
    stableCountRef.current = 0;
    lastSpokenRef.current = '';
  }, [currentMode]);

  // Reset stability on language switch
  useEffect(() => {
    stablePredictionRef.current = '—';
    stableCountRef.current = 0;
    lastSpokenRef.current = '';
  }, [currentLanguage]);

  const toggleSpeech = useCallback(() => {
    setSpeechEnabled((prev) => {
      const next = !prev;
      if (!next && typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
      return next;
    });
  }, []);

  // Motion recording state machine
  // states: "idle" | "countdown" | "recording" | "processing" | "result" | "error"
  const [motionState, setMotionState] = useState('idle');
  const [motionCountdown, setMotionCountdown] = useState(3);
  const [motionFrameCount, setMotionFrameCount] = useState(0);
  const [motionResult, setMotionResult] = useState('');

  // Internal references
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttemptRef = useRef(0);
  const sosTimeoutRef = useRef(null);

  // High-frequency prediction refs
  const latestPredictionRef = useRef('—');
  const latestConfidenceRef = useRef(0);
  const latestHandDetectedRef = useRef(false);

  // Motion recording refs
  const motionBufferRef = useRef([]);
  const motionCountdownTimerRef = useRef(null);
  const motionResultTimerRef = useRef(null);
  const motionTimeoutRef = useRef(null);
  const motionWatchdogRef = useRef(null);
  const motionStateRef = useRef('idle');

  // Keep motionStateRef synced
  useEffect(() => {
    motionStateRef.current = motionState;
  }, [motionState]);

  // Handle SOS activation with 10s auto-dismiss
  const triggerSosAlert = useCallback(() => {
    setSosActive(true);
    speakText('Emergency. Help needed.');
    if (sosTimeoutRef.current) {
      clearTimeout(sosTimeoutRef.current);
    }
    sosTimeoutRef.current = setTimeout(() => {
      setSosActive(false);
    }, 10000);
  }, []);

  const dismissSosAlert = useCallback(() => {
    if (sosTimeoutRef.current) {
      clearTimeout(sosTimeoutRef.current);
    }
    setSosActive(false);
  }, []);

  // Send raw payload over WebSocket if open
  const sendMessage = useCallback((payload) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload));
    }
  }, []);

  // Frame sender
  const sendFrame = useCallback((imageSrc) => {
    if (!imageSrc) return;
    sendMessage({
      type: 'frame',
      data: imageSrc
    });
  }, [sendMessage]);

  // Action dispatches
  const addLetter = useCallback((explicitLetter) => {
    const letterToAdd = explicitLetter !== undefined ? explicitLetter : (latestPredictionRef.current !== '—' ? latestPredictionRef.current : '');
    sendMessage({
      type: 'action',
      action: 'add_letter',
      letter: letterToAdd
    });
  }, [sendMessage]);

  const backspace = useCallback(() => {
    sendMessage({ type: 'action', action: 'backspace' });
  }, [sendMessage]);

  const clearWord = useCallback(() => {
    sendMessage({ type: 'action', action: 'clear' });
  }, [sendMessage]);

  const setWord = useCallback((word) => {
    sendMessage({ type: 'action', action: 'set_word', word });
  }, [sendMessage]);

  const speak = useCallback(() => {
    if (currentWord) {
      speakText(currentWord);
      sendMessage({ type: 'action', action: 'speak' });
    }
  }, [currentWord, sendMessage]);

  const switchLanguage = useCallback(() => {
    sendMessage({ type: 'action', action: 'switch_language' });
  }, [sendMessage]);

  const switchMode = useCallback(() => {
    sendMessage({ type: 'action', action: 'switch_mode' });
  }, [sendMessage]);

  const saveLog = useCallback(() => {
    sendMessage({ type: 'action', action: 'save_log' });
  }, [sendMessage]);

  const triggerSOS = useCallback(() => {
    sendMessage({ type: 'action', action: 'sos' });
  }, [sendMessage]);

  const sendMotionSequence = useCallback((sequence) => {
    sendMessage({
      type: 'action',
      action: 'motion_sequence',
      sequence
    });
  }, [sendMessage]);

  const deleteLogEntry = useCallback((index) => {
    sendMessage({
      type: 'action',
      action: 'delete_log_entry',
      index
    });
  }, [sendMessage]);

  // Motion recording controls
  const cancelMotionRecording = useCallback(() => {
    if (motionCountdownTimerRef.current) {
      clearInterval(motionCountdownTimerRef.current);
    }
    if (motionResultTimerRef.current) {
      clearTimeout(motionResultTimerRef.current);
    }
    if (motionTimeoutRef.current) {
      clearTimeout(motionTimeoutRef.current);
    }
    if (motionWatchdogRef.current) {
      clearTimeout(motionWatchdogRef.current);
    }
    motionBufferRef.current = [];
    setMotionState('idle');
    setMotionCountdown(3);
    setMotionFrameCount(0);
    setMotionResult('');
  }, []);

  const startMotionRecording = useCallback(() => {
    cancelMotionRecording();
    setMotionState('countdown');
    setMotionCountdown(3);
    setPrediction('—');
    setConfidence(0);

    let count = 3;
    motionCountdownTimerRef.current = setInterval(() => {
      count -= 1;
      if (count > 0) {
        setMotionCountdown(count);
      } else {
        clearInterval(motionCountdownTimerRef.current);
        motionBufferRef.current = [];
        setMotionFrameCount(0);
        setMotionState('recording');

        // Recording duration window: if 30 frames haven't arrived within 1.8s
        if (motionTimeoutRef.current) clearTimeout(motionTimeoutRef.current);
        motionTimeoutRef.current = setTimeout(() => {
          if (motionStateRef.current === 'recording') {
            const buf = motionBufferRef.current;
            if (buf.length >= 15) {
              setMotionState('processing');
              if (motionWatchdogRef.current) clearTimeout(motionWatchdogRef.current);
              motionWatchdogRef.current = setTimeout(() => {
                if (motionStateRef.current === 'processing') {
                  setMotionState('idle');
                  setMotionResult('');
                }
              }, 4000);
              sendMessage({
                type: 'action',
                action: 'motion_sequence',
                sequence: buf.slice(0, 30)
              });
            } else {
              setMotionResult('Not enough frames');
              setMotionState('result');
              if (motionResultTimerRef.current) clearTimeout(motionResultTimerRef.current);
              motionResultTimerRef.current = setTimeout(() => {
                setMotionState('idle');
                setMotionResult('');
              }, 2000);
            }
          }
        }, 1800);
      }
    }, 1000);
  }, [cancelMotionRecording, sendMessage]);

  // WebSocket connect & lifecycle
  useEffect(() => {
    let isUnmounted = false;

    function connect() {
      if (isUnmounted) return;

      const envWsUrl = import.meta.env.VITE_WS_URL;
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.hostname || 'localhost';
      const wsUrl = envWsUrl || (
        window.location.port === '5173'
          ? `${wsProtocol}//${host}:8000/ws`
          : `${wsProtocol}//${window.location.host}/ws`
      );

      try {
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          if (isUnmounted) return;
          console.log('[WebSocket] Connected to SignBridge AI backend');
          setConnected(true);
          reconnectAttemptRef.current = 0;
        };

        ws.onmessage = (event) => {
          if (isUnmounted) return;
          try {
            const data = JSON.parse(event.data);

            if (data.type === 'state') {
              if (data.current_word !== undefined) setCurrentWord(data.current_word);
              if (data.conversation_log !== undefined) setConversationLog(data.conversation_log);
              if (data.current_language !== undefined) setCurrentLanguage(data.current_language);
              if (data.current_mode !== undefined) setCurrentMode(data.current_mode);
              if (typeof data.sos_progress === 'number') setSosProgress(data.sos_progress);
              if (data.sos) {
                setSosProgress(0);
                triggerSosAlert();
              }
            } else if (data.type === 'prediction') {
              const predVal = data.letter || '—';
              const confVal = typeof data.confidence === 'number' ? data.confidence : 0;
              const handVal = Boolean(data.hand_detected);
              if (typeof data.sos_progress === 'number') {
                setSosProgress(data.sos_progress);
              }

              // ── 1. MOTION SEQUENCE RESPONSE ARRIVED ──
              if (motionStateRef.current === 'processing') {
                if (motionWatchdogRef.current) clearTimeout(motionWatchdogRef.current);
                const recognizedWord = data.letter && data.letter !== '—' ? data.letter : '';
                if (recognizedWord) {
                  setMotionResult(recognizedWord);
                  setMotionState('result');

                  // Immediately lock hero prediction and confidence to the recognized motion word!
                  latestPredictionRef.current = recognizedWord;
                  latestConfidenceRef.current = confVal;
                  latestHandDetectedRef.current = true;
                  setPrediction(recognizedWord);
                  setConfidence(confVal);
                  setHandDetected(true);

                  // Always announce recognized motion words with voice
                  speakText(recognizedWord);

                  if (motionResultTimerRef.current) clearTimeout(motionResultTimerRef.current);
                  motionResultTimerRef.current = setTimeout(() => {
                    setMotionState('idle');
                    setMotionResult('');
                  }, 3200);
                } else {
                  setMotionResult('Not recognized');
                  setMotionState('result');
                  if (motionResultTimerRef.current) clearTimeout(motionResultTimerRef.current);
                  motionResultTimerRef.current = setTimeout(() => {
                    setMotionState('idle');
                    setMotionResult('');
                  }, 2500);
                }
                return;
              }

              // ── 2. MOTION GESTURE ACTIVE (COUNTDOWN, RECORDING, OR RESULT DISPLAY) ──
              // Prevent fast static predictions from overriding or flickering on screen!
              if (motionStateRef.current !== 'idle') {
                if (motionStateRef.current === 'recording') {
                  if (handVal && Array.isArray(data.landmarks) && data.landmarks.length === 63) {
                    motionBufferRef.current.push(data.landmarks);
                    const framesNow = motionBufferRef.current.length;
                    setMotionFrameCount(framesNow);

                    if (framesNow >= 30) {
                      if (motionTimeoutRef.current) clearTimeout(motionTimeoutRef.current);
                      setMotionState('processing');
                      if (motionWatchdogRef.current) clearTimeout(motionWatchdogRef.current);
                      motionWatchdogRef.current = setTimeout(() => {
                        if (motionStateRef.current === 'processing') {
                          setMotionState('idle');
                          setMotionResult('');
                        }
                      }, 4000);
                      sendMessage({
                        type: 'action',
                        action: 'motion_sequence',
                        sequence: motionBufferRef.current.slice(0, 30)
                      });
                    }
                  }
                }
                // Suppress static single-frame predictions from overwriting hero panel while gesturing or viewing result!
                return;
              }

              // ── 3. NORMAL IDLE MODE (STATIC INFERENCE) ──
              latestPredictionRef.current = predVal;
              latestConfidenceRef.current = confVal;
              latestHandDetectedRef.current = handVal;
            }
          } catch (err) {
            console.error('[WebSocket] Error parsing message:', err);
          }
        };

        ws.onclose = () => {
          if (isUnmounted) return;
          setConnected(false);
          // Exponential backoff: 3s, 6s, 12s, max 15s
          const attempt = reconnectAttemptRef.current;
          const delay = Math.min(3000 * Math.pow(2, attempt), 15000);
          reconnectAttemptRef.current += 1;
          console.warn(`[WebSocket] Disconnected. Reconnecting in ${delay / 1000}s (attempt ${reconnectAttemptRef.current})...`);
          reconnectTimeoutRef.current = setTimeout(connect, delay);
        };

        ws.onerror = (err) => {
          console.error('[WebSocket] Error:', err);
          ws.close();
        };
      } catch (err) {
        console.error('[WebSocket] Failed to instantiate:', err);
        reconnectTimeoutRef.current = setTimeout(connect, 3000);
      }
    }

    connect();

    return () => {
      isUnmounted = true;
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (sosTimeoutRef.current) clearTimeout(sosTimeoutRef.current);
      if (motionCountdownTimerRef.current) clearInterval(motionCountdownTimerRef.current);
      if (motionResultTimerRef.current) clearTimeout(motionResultTimerRef.current);
      if (speechCooldownTimerRef.current) clearTimeout(speechCooldownTimerRef.current);
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [sendMessage, triggerSosAlert]);

  // Throttled state updater (~5.5 Hz) and automatic speech stability tracker
  useEffect(() => {
    const throttleInterval = setInterval(() => {
      // If motion gesture is active (countdown, recording, processing, or showing result), do not overwrite hero prediction!
      if (motionStateRef.current !== 'idle') {
        return;
      }

      const pred = latestPredictionRef.current;
      const conf = latestConfidenceRef.current;
      const hand = latestHandDetectedRef.current;

      setPrediction(pred);
      setConfidence(conf);
      setHandDetected(hand);

      // Automatic Speech Engine
      if (speechEnabledRef.current) {
        if (!hand || !pred || pred === '—') {
          stableCountRef.current = 0;
          stablePredictionRef.current = '—';
        } else {
          if (pred === stablePredictionRef.current) {
            stableCountRef.current += 1;
          } else {
            stablePredictionRef.current = pred;
            stableCountRef.current = 1;
          }

          const isLetters = currentModeRef.current === 'letters';
          const stableThreshold = isLetters ? 8 : 6; // ~1.44s letters, ~1.08s words
          const minConfidence = isLetters ? 60 : 70;
          const cooldownMs = isLetters ? 1500 : 2000;

          if (
            stableCountRef.current >= stableThreshold &&
            pred !== lastSpokenRef.current &&
            conf >= minConfidence &&
            !speechCooldownRef.current
          ) {
            speakText(pred);
            lastSpokenRef.current = pred;
            speechCooldownRef.current = true;
            if (speechCooldownTimerRef.current) clearTimeout(speechCooldownTimerRef.current);
            speechCooldownTimerRef.current = setTimeout(() => {
              speechCooldownRef.current = false;
            }, cooldownMs);
          }
        }
      }
    }, 180);

    return () => {
      clearInterval(throttleInterval);
      if (speechCooldownTimerRef.current) clearTimeout(speechCooldownTimerRef.current);
    };
  }, []);

  return {
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
    sendMotionSequence,
    deleteLogEntry,
    motionState,
    motionCountdown,
    motionFrameCount,
    motionResult,
    startMotionRecording,
    cancelMotionRecording,
  };
}
