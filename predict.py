"""
predict.py -- SignBridge AI Real-Time Prediction (v7)
======================================================
Startup optimizations:
  [+] All TF/oneDNN log spam suppressed immediately
  [+] Camera opens and shows "Loading..." screen FIRST
  [+] TFLite interpreter used instead of full Keras (10x faster)

Runtime improvements:
  [+] Normalized landmarks (matches training pipeline)
  [+] 20-frame weighted majority-vote smoothing (no flickering)
  [+] Visual confidence bar on screen
  [+] Both-hand support (auto-mirrors left hand)
  [+] Hand label shown on HUD (Left / Right)

Voice System (v4):
  [+] Letter announced automatically when detection changes
  [+] SPACE  -> adds letter + speaks it
  [+] T key  -> speaks current word as a whole word
  [+] R key  -> spells current word letter by letter
  [+] Non-blocking TTS via background thread (zero lag on camera)

Motion Gestures (v6 — key-triggered):
  [+] J and Z detected via LSTM motion model (30-frame sequences)
  [+] Press M key to start recording, perform J/Z gesture
  [+] Visual progress bar during recording
  [+] No automatic motion detection (eliminates Z hallucination)
  [+] Full A-Z support (all 26 ASL letters)

Common Words (v7 — NEW):
  [+] 23 common word signs (HELP, STOP, WATER, etc.)
  [+] Press P to toggle between Letters and Words mode
  [+] Auto-speaks detected word when stable
  [+] Green [WORDS] badge in HUD

Controls:
  SPACE      = add detected letter/word to sentence
  BACKSPACE  = delete last letter (letters mode) / last word (words mode)
  C          = clear word/sentence
  M          = record J/Z motion gesture
  T          = speak current WORD  (e.g. "hello")
  R          = spell current word  (e.g. "H... E... L... L... O")
  V          = toggle voice ON/OFF
  P          = toggle Letters/Words mode
  Q          = quit
"""

# ── Suppress ALL TensorFlow/oneDNN log noise BEFORE any TF import ────────────
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']   = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS']  = '0'
os.environ['GLOG_minloglevel']       = '3'

import warnings
warnings.filterwarnings('ignore')

# ── Open camera IMMEDIATELY so user sees something right away ─────────────────
import cv2
import time as _time

def open_camera(indices=(0, 1, 2), retries=3, delay=0.5):
    """
    Try multiple camera indices with retries.
    Returns the first working VideoCapture, or None.
    """
    for attempt in range(1, retries + 1):
        for idx in indices:
            c = cv2.VideoCapture(idx)
            if c.isOpened():
                ret, _ = c.read()
                if ret:
                    print(f"Camera opened on index {idx} (attempt {attempt})")
                    return c
                c.release()
        _time.sleep(delay)
    return None

cap = open_camera()
if cap is None:
    # Show a standalone error window before exiting
    import numpy as _np
    err_frame = _np.zeros((480, 640, 3), dtype='uint8')
    cv2.putText(err_frame, "ERROR: No camera found!",
                (40, 200), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(err_frame, "Close other apps using the camera and retry.",
                (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(err_frame, "Press any key to exit...",
                (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
    cv2.imshow("SignBridge AI", err_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    raise SystemExit("No working camera found. Close other apps and retry.")

WINDOW_NAME = "SignBridge AI"
_window_focused = [False]  # mutable so no 'global' needed inside the loop

def show_loading(message="Loading SignBridge AI..."):
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
    else:
        frame = __import__('numpy').zeros((480, 640, 3), dtype='uint8')
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, message,
                (40, frame.shape[0]//2),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 210, 255), 2)
    cv2.putText(frame, "Please wait...",
                (40, frame.shape[0]//2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(1)

show_loading("Loading AI model...")

# ── Now load the heavy stuff ──────────────────────────────────────────────────
import numpy as np
show_loading("Loading AI model...")

import tensorflow as tf
show_loading("Initializing voice engine...")

# ── Voice engine (non-blocking background thread) ─────────────────────────────
import pyttsx3
import threading
import queue

_tts_queue   = queue.Queue()
_voice_on    = [True]   # mutable so it can be toggled inside the loop

def _tts_worker():
    """Runs in a daemon thread — consumes speech requests from the queue."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 165)    # comfortable speaking speed
    engine.setProperty('volume', 1.0)
    # Use a clear, natural voice if available
    voices = engine.getProperty('voices')
    for v in voices:
        if 'zira' in v.name.lower() or 'david' in v.name.lower():
            engine.setProperty('voice', v.id)
            break
    while True:
        text = _tts_queue.get()
        if text is None:          # sentinel → shutdown
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass                  # never crash the main app over TTS

_tts_thread = threading.Thread(target=_tts_worker, daemon=True)
_tts_thread.start()

def speak(text: str, flush: bool = False):
    """
    Queue a speech request.
    flush=True clears any pending speech first (use for letter announcements
    so old requests don't pile up behind a fast hand movement).
    """
    if not _voice_on[0]:
        return
    if flush:
        # Drain the queue — discard stale requests
        while not _tts_queue.empty():
            try:
                _tts_queue.get_nowait()
            except queue.Empty:
                break
    _tts_queue.put(text)

def spell_word(word: str):
    """Speak each letter of the word with a short pause between them."""
    if not word:
        speak("No letters yet", flush=True)
        return
    # Spell e.g. "HELLO" as "H... E... L... L... O"
    spaced = ".  ".join(list(word))
    speak(spaced, flush=True)

show_loading("Starting camera...")

import mediapipe as mp
from collections import deque, Counter
from utils import normalize_landmarks, mirror_hand, normalize_isl_landmarks

# ── Load static TFLite model (A-Y, 24 signs) ─────────────────────────────────
TFLITE_PATH = "signbridge_model.tflite"
KERAS_PATH  = "signbridge_model.h5"

if os.path.exists(TFLITE_PATH):
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    USE_TFLITE = True
    print("Static model loaded (TFLite - fast mode)")
else:
    keras_model = tf.keras.models.load_model(KERAS_PATH)
    USE_TFLITE  = False
    print("Static model loaded (Keras fallback)")

classes = np.load("label_encoder.npy", allow_pickle=True)

# ── Load motion LSTM model (J & Z) ───────────────────────────────────────────
MOTION_TFLITE_PATH = "motion_model.tflite"
MOTION_KERAS_PATH  = "motion_model.h5"
MOTION_SEQUENCE_LEN = 30        # must match training
MOTION_AVAILABLE    = False

if os.path.exists(MOTION_TFLITE_PATH):
    motion_interpreter = tf.lite.Interpreter(model_path=MOTION_TFLITE_PATH)
    motion_interpreter.allocate_tensors()
    motion_input_details  = motion_interpreter.get_input_details()
    motion_output_details = motion_interpreter.get_output_details()
    motion_classes = np.load("motion_label_encoder.npy", allow_pickle=True)
    MOTION_AVAILABLE = True
    print(f"Motion model loaded (TFLite) — classes: {motion_classes}")
elif os.path.exists(MOTION_KERAS_PATH):
    motion_keras_model = tf.keras.models.load_model(MOTION_KERAS_PATH)
    motion_classes = np.load("motion_label_encoder.npy", allow_pickle=True)
    MOTION_AVAILABLE = True
    print(f"Motion model loaded (Keras fallback) — classes: {motion_classes}")
else:
    print("Motion model not found — J and Z detection disabled.")
    print("  Run motion_data_collection.py + train_motion_model.py to enable.")

# ── Load ISL model (A-Z, 26 signs, two-hand input) ───────────────────────────
ISL_TFLITE_PATH = "isl_model.tflite"
ISL_KERAS_PATH  = "isl_model.h5"
ISL_USE_TFLITE  = False
isl_available   = False

if os.path.exists(ISL_TFLITE_PATH):
    isl_interpreter = tf.lite.Interpreter(model_path=ISL_TFLITE_PATH)
    isl_interpreter.allocate_tensors()
    isl_input_details  = isl_interpreter.get_input_details()
    isl_output_details = isl_interpreter.get_output_details()
    ISL_USE_TFLITE = True
    isl_available  = True
    print("ISL model loaded (TFLite - fast mode)")
elif os.path.exists(ISL_KERAS_PATH):
    isl_keras_model = tf.keras.models.load_model(ISL_KERAS_PATH)
    isl_available  = True
    print("ISL model loaded (Keras fallback)")
else:
    print("ISL model not found — run train_isl_model.py to enable ISL mode.")

if isl_available and os.path.exists("isl_label_encoder.npy"):
    isl_classes = np.load("isl_label_encoder.npy", allow_pickle=True)
    print(f"ISL classes: {isl_classes}")
elif isl_available:
    print("WARNING: isl_label_encoder.npy not found — ISL disabled.")
    isl_available = False

# ── Load Common Words model (23 word signs) ──────────────────────────────────
WORDS_TFLITE_PATH = "words_model.tflite"
WORDS_KERAS_PATH  = "words_model.h5"
WORDS_USE_TFLITE  = False
words_available   = False

if os.path.exists(WORDS_TFLITE_PATH):
    words_interpreter = tf.lite.Interpreter(model_path=WORDS_TFLITE_PATH)
    words_interpreter.allocate_tensors()
    words_input_details  = words_interpreter.get_input_details()
    words_output_details = words_interpreter.get_output_details()
    WORDS_USE_TFLITE = True
    words_available  = True
    print("Words model loaded (TFLite - fast mode)")
elif os.path.exists(WORDS_KERAS_PATH):
    words_keras_model = tf.keras.models.load_model(WORDS_KERAS_PATH)
    words_available   = True
    print("Words model loaded (Keras fallback)")
else:
    print("Words model not found — run train_words_model.py to enable words mode.")

if words_available and os.path.exists("words_label_encoder.npy"):
    words_classes = np.load("words_label_encoder.npy", allow_pickle=True)
    print(f"Words model loaded — {len(words_classes)} signs ready")
elif words_available:
    print("WARNING: words_label_encoder.npy not found — words mode disabled.")
    words_available = False

# ── Load Motion Words model (7 motion word signs) ────────────────────────────
WORDS_MOTION_TFLITE_PATH = "words_motion_model.tflite"
WORDS_MOTION_KERAS_PATH  = "words_motion_model.h5"
words_motion_available   = False

if os.path.exists(WORDS_MOTION_TFLITE_PATH):
    wm_interpreter = tf.lite.Interpreter(model_path=WORDS_MOTION_TFLITE_PATH)
    wm_interpreter.allocate_tensors()
    wm_input_details  = wm_interpreter.get_input_details()
    wm_output_details = wm_interpreter.get_output_details()
    words_motion_available = True
    print("Words motion model loaded (TFLite - fast mode)")
elif os.path.exists(WORDS_MOTION_KERAS_PATH):
    wm_keras_model = tf.keras.models.load_model(WORDS_MOTION_KERAS_PATH)
    words_motion_available = True
    print("Words motion model loaded (Keras fallback)")
else:
    print("Words motion model not found — run train_words_motion_model.py")

if words_motion_available and os.path.exists("words_motion_label_encoder.npy"):
    words_motion_label_encoder = np.load("words_motion_label_encoder.npy",
                                         allow_pickle=True)
    print(f"Words motion model loaded — {len(words_motion_label_encoder)} signs ready")
elif words_motion_available:
    print("WARNING: words_motion_label_encoder.npy not found — motion words disabled.")
    words_motion_available = False

# ── Inference helpers ─────────────────────────────────────────────────────────
def predict_sign(norm_landmarks: np.ndarray) -> tuple[str, float]:
    """Run static sign inference. Returns (letter, confidence_percent)."""
    inp = np.array([norm_landmarks], dtype=np.float32)
    if USE_TFLITE:
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_details[0]['index'])[0]
    else:
        probs = keras_model.predict(inp, verbose=0)[0]
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx] * 100)

def predict_motion(sequence: np.ndarray) -> tuple[str, float]:
    """
    Run motion gesture inference on a (30, 63) sequence.
    Returns (letter, confidence_percent) for J or Z.
    """
    inp = np.array([sequence], dtype=np.float32)
    if os.path.exists(MOTION_TFLITE_PATH):
        motion_interpreter.set_tensor(motion_input_details[0]['index'], inp)
        motion_interpreter.invoke()
        probs = motion_interpreter.get_tensor(motion_output_details[0]['index'])[0]
    else:
        probs = motion_keras_model.predict(inp, verbose=0)[0]
    idx = int(np.argmax(probs))
    return motion_classes[idx], float(probs[idx] * 100)

# ── ISL inference helper ──────────────────────────────────────────────────────
def predict_isl(result):
    """
    ISL two-hand prediction: extract both hands, normalize, mirror left hand,
    run ISL model, apply weighted smoothing.
    Returns (letter, confidence_percent) or ("", 0.0) if no hands detected.
    """
    global isl_hands_count

    if not (result and result.hand_landmarks):
        isl_hands_count = 0
        return "", 0.0

    num_detected = len(result.hand_landmarks)
    isl_hands_count = num_detected

    right_raw = np.zeros(63, dtype=np.float32)
    left_raw  = np.zeros(63, dtype=np.float32)
    left_detected = False

    for hi in range(num_detected):
        hand_lms = result.hand_landmarks[hi]
        raw_lm = [coord for lm in hand_lms for coord in (lm.x, lm.y, lm.z)]
        h_label = result.handedness[hi][0].display_name
        if h_label == "Left":        # flipped frame: "Left" = user's right hand
            right_raw = np.array(raw_lm, dtype=np.float32)
        else:                        # "Right" = user's left hand
            left_raw = np.array(raw_lm, dtype=np.float32)
            left_detected = True

    # Normalize both hands (right directly, left will be mirrored)
    isl_input = normalize_isl_landmarks(right_raw, left_raw)
    if left_detected:
        isl_input[63:] = mirror_hand(isl_input[63:])

    # Run ISL inference
    inp = np.array([isl_input], dtype=np.float32)
    if ISL_USE_TFLITE:
        isl_interpreter.set_tensor(isl_input_details[0]['index'], inp)
        isl_interpreter.invoke()
        probs = isl_interpreter.get_tensor(isl_output_details[0]['index'])[0]
    else:
        probs = isl_keras_model.predict(inp, verbose=0)[0]

    idx = int(np.argmax(probs))
    letter = isl_classes[idx]
    conf = float(probs[idx] * 100)

    # Weighted smoothing (same formula as ASL, separate buffer)
    isl_prediction_buf.append(letter)
    isl_conf_buf.append(conf)

    vote_counter = Counter()
    buf_list = list(isl_prediction_buf)
    for vi, vl in enumerate(buf_list):
        weight = 1.0 + vi * 0.15
        vote_counter[vl] += weight
    smoothed_letter = vote_counter.most_common(1)[0][0]

    conf_list = list(isl_conf_buf)
    weights = np.array([1.0 + i * 0.15 for i in range(len(conf_list))])
    smoothed_conf = float(np.average(conf_list, weights=weights))

    return smoothed_letter, smoothed_conf


# ── Common Words inference helper (TWO-HAND, 126-dim) ────────────────────────
WORDS_CONFIDENCE_THRESHOLD = 75.0  # suppress flickering low-confidence word predictions

def predict_words(result):
    """
    Common Words prediction: extract BOTH hands, normalize via ISL-style,
    mirror left hand, run 126-dim through words model, weighted smoothing.
    Returns (word, confidence_percent) or ("", 0.0) if no hand detected.
    """
    global words_hands_count

    if not (result and result.hand_landmarks):
        words_hands_count = 0
        return "", 0.0

    num_detected = len(result.hand_landmarks)
    words_hands_count = num_detected

    right_raw = np.zeros(63, dtype=np.float32)
    left_raw  = np.zeros(63, dtype=np.float32)
    left_detected = False

    for hi in range(num_detected):
        hand_lms = result.hand_landmarks[hi]
        raw_lm = [coord for lm in hand_lms for coord in (lm.x, lm.y, lm.z)]
        h_label = result.handedness[hi][0].display_name
        if h_label == "Left":        # flipped frame: "Left" = user's right hand
            right_raw = np.array(raw_lm, dtype=np.float32)
        else:                        # "Right" = user's left hand
            left_raw = np.array(raw_lm, dtype=np.float32)
            left_detected = True

    # Normalize both hands (same as ISL pipeline)
    words_input = normalize_isl_landmarks(right_raw, left_raw)
    if left_detected:
        words_input[63:] = mirror_hand(words_input[63:])

    # Run words model inference (TFLite preferred, Keras fallback)
    inp = np.array([words_input], dtype=np.float32)
    if WORDS_USE_TFLITE:
        words_interpreter.set_tensor(words_input_details[0]['index'], inp)
        words_interpreter.invoke()
        probs = words_interpreter.get_tensor(words_output_details[0]['index'])[0]
    else:
        probs = words_keras_model.predict(inp, verbose=0)[0]

    idx = int(np.argmax(probs))
    word = words_classes[idx]
    conf = float(probs[idx] * 100)

    # Confidence gate: suppress low-confidence flickering predictions
    if conf < WORDS_CONFIDENCE_THRESHOLD:
        return "", 0.0

    # Weighted smoothing using SEPARATE words buffer
    words_prediction_buf.append(word)
    words_conf_buf.append(conf)

    vote_counter = Counter()
    buf_list = list(words_prediction_buf)
    for vi, vl in enumerate(buf_list):
        weight = 1.0 + vi * 0.15
        vote_counter[vl] += weight
    smoothed_word = vote_counter.most_common(1)[0][0]

    conf_list = list(words_conf_buf)
    weights = np.array([1.0 + i * 0.15 for i in range(len(conf_list))])
    smoothed_conf = float(np.average(conf_list, weights=weights))

    return smoothed_word, smoothed_conf


# ── Motion Words inference helper (7 motion word signs) ──────────────────────
def predict_words_motion(sequence):
    """
    Run motion words inference on a (30, 63) sequence.
    Returns (word, confidence) where confidence is 0.0–1.0.
    """
    if not words_motion_available:
        return ("", 0.0)
    arr = np.array(sequence, dtype=np.float32)
    arr = arr.reshape(1, 30, 63)
    # TFLite inference (same pattern as predict_motion for J/Z)
    if os.path.exists(WORDS_MOTION_TFLITE_PATH):
        wm_interpreter.set_tensor(wm_input_details[0]['index'], arr)
        wm_interpreter.invoke()
        output = wm_interpreter.get_tensor(wm_output_details[0]['index'])
    else:
        output = wm_keras_model.predict(arr, verbose=0)
    idx = np.argmax(output[0])
    confidence = float(output[0][idx])
    word = words_motion_label_encoder[idx]
    return (word, confidence)


# ── Motion Words frame landmark extraction (for recording) ──────────────────
def get_frame_landmarks(result):
    """
    Extract normalized, hand-mirrored landmarks from a MediaPipe result.
    Returns np.zeros(63) if no hand detected (keeps recording going).
    """
    if not result or not result.hand_landmarks:
        return np.zeros(63)
    hand = result.hand_landmarks[0]
    raw = [c for lm in hand for c in (lm.x, lm.y, lm.z)]
    normalized = normalize_landmarks(raw)
    if (result.handedness and
            result.handedness[0][0].category_name == "Left"):
        pts = np.array(normalized).reshape(21, 3)
        normalized = mirror_hand(pts).flatten().tolist()
    return np.array(normalized)


# ── MediaPipe setup ───────────────────────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

latest_result = None

def save_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=save_result
)

# ── Config ────────────────────────────────────────────────────────────────────
SMOOTHING_WINDOW     = 20        # larger window for smoother, more stable output
MIN_CONFIDENCE       = 45.0      # slightly higher threshold to reduce noise
LETTER_STABLE_FRAMES = 10        # frames the same letter must hold before announcing

# Dynamically generate a unique vibrant color for each sign (HSV wheel)
import colorsys
_all_signs = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # full A-Z for color mapping
SIGN_COLORS = {}
for _i, _s in enumerate(_all_signs):
    h = _i / len(_all_signs)
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    SIGN_COLORS[_s] = (int(b * 255), int(g * 255), int(r * 255))  # BGR
DEFAULT_COLOR = (255, 255, 255)

# ── Draw helpers ──────────────────────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_skeleton(frame, result, color=(0, 255, 120)):
    if not (result and result.hand_landmarks):
        return
    h, w, _ = frame.shape
    pts = result.hand_landmarks[0]
    xy  = [(int(lm.x * w), int(lm.y * h)) for lm in pts]
    for a, b in CONNECTIONS:
        cv2.line(frame, xy[a], xy[b], color, 2)
    for x, y in xy:
        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 3, color, -1)

def draw_confidence_bar(frame, confidence, color, x=10, y=110, width=260, height=18):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
    filled = int((confidence / 100.0) * width)
    cv2.rectangle(frame, (x, y), (x + filled, y + height), color, -1)
    cv2.putText(frame, f"{confidence:.1f}%",
                (x + width + 8, y + height - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

def extract_raw_landmarks(result):
    if result and result.hand_landmarks:
        hand = result.hand_landmarks[0]
        return [coord for lm in hand for coord in (lm.x, lm.y, lm.z)]
    return None

# ── Main loop state ───────────────────────────────────────────────────────────
timestamp           = 0
prediction_buf      = deque(maxlen=SMOOTHING_WINDOW)
conf_buf            = deque(maxlen=SMOOTHING_WINDOW)
word                = ""
sentence_words      = []          # list of completed words
last_spoken_letter  = ""          # track last announced letter
stable_count        = 0           # how many frames current_letter has been stable

# ── Motion detection state (KEY-TRIGGERED, not automatic) ─────────────────────
# The motion LSTM is a binary J/Z classifier — it outputs J or Z for ANY input.
# Automatic detection causes Z hallucination on every hand movement.
# Solution: user presses M to start motion recording, performs J/Z, system classifies.
MOTION_RECORDING     = False       # True while recording motion frames
MOTION_RECORD_BUF    = []          # collects frames during recording
MOTION_RECORD_TARGET = 30          # frames to record (matches training)
motion_active_letter = ""          # detected motion letter result
motion_active_conf   = 0.0         # confidence of detection
motion_display_frames = 0          # frames to keep showing motion result
MOTION_DISPLAY_LEN   = 30          # show motion result for ~1 second

# ── ISL mode state ────────────────────────────────────────────────────────────
current_language    = "ASL"        # "ASL" or "ISL"
isl_prediction_buf  = deque(maxlen=SMOOTHING_WINDOW)
isl_conf_buf        = deque(maxlen=SMOOTHING_WINDOW)
isl_hands_count     = 0            # how many hands detected in ISL mode

# ── Common Words mode state ───────────────────────────────────────────────────
current_mode          = "letters"   # "letters" or "words"
words_prediction_buf  = deque(maxlen=SMOOTHING_WINDOW)
words_conf_buf        = deque(maxlen=SMOOTHING_WINDOW)
words_stable_count    = 0           # frames current word has been stable
words_last_spoken     = ""          # last auto-spoken word (prevents spam)
words_last_detected   = ""          # last smoothed word for stable tracking
words_hands_count     = 0           # how many hands detected in words mode
WORDS_STABLE_FRAMES   = 15          # frames the same word must hold before auto-speak

# ── Motion Words recording state (N key in words mode) ────────────────────────
words_motion_buf            = []     # frames collected during recording
words_motion_recording      = False  # True while actively recording 30 frames
words_motion_countdown      = 0      # countdown value (3, 2, 1, 0)
words_motion_countdown_start = 0.0   # time.time() when countdown started
words_motion_result_text    = ""     # text to display after prediction
words_motion_result_timer   = 0      # frames remaining to show result
words_motion_last_spoken    = ""     # prevents repeat announcements

motion_status       = "enabled (M key)" if MOTION_AVAILABLE else "disabled (model not found)"
isl_status          = "available (L key)" if isl_available else "disabled (model not found)"
words_status        = "available (P key)" if words_available else "disabled (model not found)"
words_motion_status = "available (N key)" if words_motion_available else "disabled (model not found)"
print("=" * 58)
print("  SignBridge AI — Ready!  (A-Z, all 26 signs)")
print(f"  Motion gestures (J/Z): {motion_status}")
print(f"  ISL mode: {isl_status}")
print(f"  Words mode: {words_status}")
print(f"  Motion words: {words_motion_status}")
print("  SPACE=add letter | BKSP=delete | C=clear word")
print("  M=detect J/Z | N=motion word | T=speak | R=spell | V=voice")
print("  L=toggle ASL/ISL | P=toggle Words | Q=quit")
print("=" * 58 + "\n")

# Announce app start
speak("SignBridge AI ready. Show a sign to begin.", flush=True)

FRAME_FAIL_LIMIT = 30   # tolerate up to 30 consecutive bad frames (~1 sec)
frame_fail_count = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_fail_count += 1
            if frame_fail_count >= FRAME_FAIL_LIMIT:
                err_frame = np.zeros((480, 640, 3), dtype='uint8')
                cv2.putText(err_frame, "Camera disconnected!",
                            (40, 200), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(err_frame, f"Lost {FRAME_FAIL_LIMIT} frames in a row.",
                            (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
                cv2.putText(err_frame, "Press any key to exit...",
                            (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
                cv2.imshow("SignBridge AI", err_frame)
                cv2.waitKey(0)
                speak("Camera lost. Exiting.", flush=True)
                break
            _time.sleep(0.03)
            continue
        frame_fail_count = 0

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_image, timestamp)

        # ── Predict ───────────────────────────────────────────────────────────
        current_letter = ""
        current_conf   = 0.0
        is_left_hand   = False
        current_word_prediction = ""   # words mode detected word
        current_word_conf       = 0.0  # words mode confidence

        # ── WORDS MODE prediction ─────────────────────────────────────────────
        if current_mode == "words" and words_available:

            # ── Motion words countdown + recording logic ──────────────────────
            if words_motion_countdown > 0:
                # Handle countdown (3, 2, 1 → start recording)
                elapsed = _time.time() - words_motion_countdown_start
                current_count = 3 - int(elapsed)
                if current_count <= 0:
                    words_motion_countdown = 0
                    words_motion_recording = True
                    words_motion_buf = []
                else:
                    words_motion_countdown = current_count

            elif words_motion_recording:
                # Handle active recording — collect 30 frames
                frame_lm = get_frame_landmarks(latest_result)
                words_motion_buf.append(frame_lm)

                if len(words_motion_buf) >= 30:
                    # Recording complete — run inference
                    words_motion_recording = False
                    word_m, conf_m = predict_words_motion(words_motion_buf)
                    words_motion_buf = []

                    MOTION_CONFIDENCE_THRESHOLD = 0.80
                    if conf_m >= MOTION_CONFIDENCE_THRESHOLD:
                        words_motion_result_text = f"{word_m} ({conf_m*100:.0f}%)"
                        words_motion_result_timer = 60
                        if word_m != words_motion_last_spoken:
                            speak(word_m, flush=True)
                            words_motion_last_spoken = word_m
                        print(f"  [MOTION WORD] Detected: {word_m} @ {conf_m*100:.1f}%")
                    else:
                        words_motion_result_text = "Not recognized — try again"
                        words_motion_result_timer = 60
                        words_motion_last_spoken = ""
                        print(f"  [MOTION WORD] Low confidence ({conf_m*100:.1f}%), try again")

            # Countdown timer for result display
            if words_motion_result_timer > 0:
                words_motion_result_timer -= 1

            # ── Static words prediction (only when NOT recording motion) ──────
            if not words_motion_recording and words_motion_countdown == 0:
                current_word_prediction, current_word_conf = predict_words(latest_result)
            else:
                current_word_prediction, current_word_conf = "", 0.0

            # Auto-speak when word is stable for WORDS_STABLE_FRAMES
            if current_word_prediction and current_word_conf >= MIN_CONFIDENCE:
                if current_word_prediction == words_last_detected:
                    words_stable_count += 1
                else:
                    words_stable_count = 1
                    words_last_detected = current_word_prediction

                if words_stable_count == WORDS_STABLE_FRAMES:
                    # Only speak if this is a NEW word (not the same one just spoken)
                    if current_word_prediction != words_last_spoken:
                        speak(current_word_prediction, flush=True)
                        words_last_spoken = current_word_prediction
                        print(f"  [WORDS] Auto-spoke: {current_word_prediction}")
                    # Reset stable count after speaking to avoid re-triggering
                    words_stable_count = 0
            else:
                words_stable_count = 0
                # Reset last spoken when no confident detection (allows re-speaking)
                if not current_word_prediction:
                    words_last_spoken = ""

            # Detect handedness for skeleton drawing
            if latest_result and latest_result.handedness:
                handedness_label = latest_result.handedness[0][0].display_name
                is_left_hand = (handedness_label == "Right")

        # ── LETTERS MODE prediction (existing ASL/ISL logic, unchanged) ───────
        elif current_mode == "letters":

            # ── ISL Prediction (two-hand, 126-dim) ───────────────────────────
            if current_language == "ISL" and isl_available:
                current_letter, current_conf = predict_isl(latest_result)

            # ── ASL Prediction (static signs A-Y) ─────────────────────────────
            raw = extract_raw_landmarks(latest_result) if current_language == "ASL" else None

            if latest_result and latest_result.handedness:
                handedness_label = latest_result.handedness[0][0].display_name
                is_left_hand = (handedness_label == "Right")

            if raw:
                norm_data = normalize_landmarks(raw)
                if is_left_hand:
                    norm_data = mirror_hand(norm_data)

                # ── Motion recording: if active, collect frames ───────────────
                if MOTION_RECORDING and MOTION_AVAILABLE:
                    MOTION_RECORD_BUF.append(norm_data.copy())
                    if len(MOTION_RECORD_BUF) >= MOTION_RECORD_TARGET:
                        # Recording complete — run LSTM classifier
                        motion_seq = np.array(MOTION_RECORD_BUF, dtype=np.float32)
                        m_letter, m_conf = predict_motion(motion_seq)
                        print(f"  [MOTION] Detected: {m_letter} @ {m_conf:.1f}%")

                        if m_conf >= 65.0:  # reasonable threshold for triggered detection
                            motion_active_letter = m_letter
                            motion_active_conf = m_conf
                            motion_display_frames = MOTION_DISPLAY_LEN
                            speak(m_letter, flush=True)
                        else:
                            print(f"  [MOTION] Confidence too low ({m_conf:.1f}%), try again")
                            speak("Not recognized, try again", flush=True)

                        # Reset recording state
                        MOTION_RECORDING = False
                        MOTION_RECORD_BUF = []
                        prediction_buf.clear()
                        conf_buf.clear()

                # ── Normal static prediction (only when NOT recording motion) ─
                if not MOTION_RECORDING:
                    letter, conf = predict_sign(norm_data)
                    prediction_buf.append(letter)
                    conf_buf.append(conf)

                    # Weighted majority vote: recent frames count more
                    vote_counter = Counter()
                    buf_list = list(prediction_buf)
                    for vi, vl in enumerate(buf_list):
                        weight = 1.0 + vi * 0.15
                        vote_counter[vl] += weight
                    current_letter = vote_counter.most_common(1)[0][0]

                    # Confidence: weighted average favoring recent frames
                    conf_list = list(conf_buf)
                    weights = np.array([1.0 + i * 0.15 for i in range(len(conf_list))])
                    current_conf = float(np.average(conf_list, weights=weights))

            # ── Motion display: show motion result if active ──────────────────
            if motion_display_frames > 0 and current_language == "ASL":
                current_letter = motion_active_letter
                current_conf = motion_active_conf
                motion_display_frames -= 1
                if motion_display_frames == 0:
                    prediction_buf.clear()
                    conf_buf.clear()

            # ── Voice: auto-announce when letter becomes stable ───────────────
            if current_letter and current_conf >= MIN_CONFIDENCE:
                if current_letter == last_spoken_letter:
                    stable_count += 1
                else:
                    stable_count = 1
                    last_spoken_letter = current_letter

                if stable_count == LETTER_STABLE_FRAMES:
                    speak(current_letter, flush=True)
            else:
                stable_count = 0

        # ── Draw skeleton ─────────────────────────────────────────────────────
        if current_mode == "words":
            # Words mode: draw BOTH hands (green = right, blue = left)
            if latest_result and latest_result.hand_landmarks:
                h_sk, w_sk, _ = frame.shape
                for hi in range(len(latest_result.hand_landmarks)):
                    h_label = latest_result.handedness[hi][0].display_name
                    skel_color = (0, 255, 120) if h_label == "Left" else (255, 180, 0)
                    pts = latest_result.hand_landmarks[hi]
                    xy = [(int(lm.x * w_sk), int(lm.y * h_sk)) for lm in pts]
                    for a, b in CONNECTIONS:
                        cv2.line(frame, xy[a], xy[b], skel_color, 2)
                    for x, y in xy:
                        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
                        cv2.circle(frame, (x, y), 3, skel_color, -1)
        elif current_language == "ISL" and isl_available:
            # ISL: draw both hands: GREEN = user's right, BLUE = user's left
            if latest_result and latest_result.hand_landmarks:
                h_sk, w_sk, _ = frame.shape
                for hi in range(len(latest_result.hand_landmarks)):
                    h_label = latest_result.handedness[hi][0].display_name
                    skel_color = (0, 255, 120) if h_label == "Left" else (255, 180, 0)
                    pts = latest_result.hand_landmarks[hi]
                    xy = [(int(lm.x * w_sk), int(lm.y * h_sk)) for lm in pts]
                    for a, b in CONNECTIONS:
                        cv2.line(frame, xy[a], xy[b], skel_color, 2)
                    for x, y in xy:
                        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
                        cv2.circle(frame, (x, y), 3, skel_color, -1)
        else:
            sign_color = SIGN_COLORS.get(current_letter, DEFAULT_COLOR)
            draw_skeleton(frame, latest_result, color=sign_color)

        # ── HUD overlay (top panel) ───────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (340, 150), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # ── WORDS MODE HUD ────────────────────────────────────────────────────
        if current_mode == "words":
            # Green [WORDS] badge — no language sub-badge
            cv2.rectangle(frame, (6, 4), (95, 24), (0, 160, 80), -1)
            cv2.putText(frame, "WORDS",
                        (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Hand count indicator (same as ISL)
            hands_txt = f"{words_hands_count} hand{'s' if words_hands_count != 1 else ''}"
            cv2.putText(frame, hands_txt,
                        (200, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1)

            # Voice status
            voice_txt   = "Voice: ON" if _voice_on[0] else "Voice: OFF"
            voice_color = (50, 220, 120) if _voice_on[0] else (80, 80, 80)
            cv2.putText(frame, voice_txt,
                        (175, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, voice_color, 1)

            # ── Motion words countdown overlay ────────────────────────────────
            if words_motion_countdown > 0:
                # Show large countdown number center screen
                count_text = str(words_motion_countdown)
                text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_DUPLEX, 4.0, 5)[0]
                cx = (frame.shape[1] - text_size[0]) // 2
                cy = (frame.shape[0] + text_size[1]) // 2
                cv2.putText(frame, count_text,
                            (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 4.0,
                            (255, 255, 255), 5)
                # "Get ready..." below
                cv2.putText(frame, "Get ready...",
                            (frame.shape[1] // 2 - 80, cy + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 255), 2)

            # ── Motion words recording indicator ──────────────────────────────
            elif words_motion_recording:
                # Red border around full frame
                cv2.rectangle(frame, (0, 0),
                              (frame.shape[1]-1, frame.shape[0]-1),
                              (0, 0, 255), 4)
                # Recording progress (white text on red background)
                rec_text = f"RECORDING: {len(words_motion_buf)}/30"
                rec_size = cv2.getTextSize(rec_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                rec_x = (frame.shape[1] - rec_size[0]) // 2
                cv2.rectangle(frame, (rec_x - 10, 30), (rec_x + rec_size[0] + 10, 58),
                              (0, 0, 200), -1)
                cv2.putText(frame, rec_text,
                            (rec_x, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)

            # ── Motion word result display (takes priority over static) ────────
            elif words_motion_result_timer > 0:
                if "Not recognized" in words_motion_result_text:
                    # Orange text for failed detection
                    cv2.putText(frame, "Try again",
                                (18, 95), cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 140, 255), 4)
                    cv2.putText(frame, "Motion Word",
                                (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 1)
                else:
                    # Extract word name from result text "WORD (XX%)"
                    detected_word = words_motion_result_text.split(" (")[0]
                    # Green text — same style as static word detection
                    cv2.putText(frame, detected_word,
                                (18, 95), cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 230, 120), 4)
                    cv2.putText(frame, "Motion Word",
                                (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
                    # Show confidence info below
                    cv2.putText(frame, words_motion_result_text,
                                (18, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 220, 100), 1)

            # ── Normal static word prediction display ─────────────────────────
            elif current_word_prediction and current_word_conf >= MIN_CONFIDENCE:
                cv2.putText(frame, current_word_prediction,
                            (18, 95), cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 230, 120), 4)
                cv2.putText(frame, "Detected Word",
                            (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
                draw_confidence_bar(frame, current_word_conf, (0, 220, 100))
            else:
                # Show "?" when confidence is below threshold (instead of flickering words)
                cv2.putText(frame, "?",
                            (18, 95), cv2.FONT_HERSHEY_DUPLEX, 3.5, (80, 80, 80), 4)
                cv2.putText(frame, "Show a word sign",
                            (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)

        # ── LETTERS MODE HUD (existing, unchanged) ───────────────────────────
        else:
            # Language badge (top-left corner)
            if current_language == "ASL":
                cv2.putText(frame, "[ASL]",
                            (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
            else:
                cv2.putText(frame, "[ISL]",
                            (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)

            # Hand label
            if current_language == "ISL" and isl_available:
                hands_txt = f"{isl_hands_count} hand{'s' if isl_hands_count != 1 else ''}"
                cv2.putText(frame, hands_txt,
                            (200, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1)
            else:
                hand_label = "Left Hand" if is_left_hand else "Right Hand"
                hand_color = (100, 200, 255) if is_left_hand else (100, 255, 180)
                cv2.putText(frame, hand_label,
                            (200, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)

            # Voice status indicator
            voice_txt   = "Voice: ON" if _voice_on[0] else "Voice: OFF"
            voice_color = (50, 220, 120) if _voice_on[0] else (80, 80, 80)
            cv2.putText(frame, voice_txt,
                        (175, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, voice_color, 1)

            # ── Motion recording indicator ────────────────────────────────────
            sign_color = SIGN_COLORS.get(current_letter, DEFAULT_COLOR)
            if MOTION_RECORDING:
                progress = len(MOTION_RECORD_BUF) / MOTION_RECORD_TARGET
                bar_w = 260
                filled = int(progress * bar_w)
                cv2.putText(frame, "RECORDING J/Z...",
                            (18, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
                cv2.rectangle(frame, (10, 55), (10 + bar_w, 70), (40, 40, 40), -1)
                cv2.rectangle(frame, (10, 55), (10 + filled, 70), (0, 140, 255), -1)
                cv2.putText(frame, f"{int(progress * 100)}%",
                            (10 + bar_w + 8, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            elif current_letter and current_conf >= MIN_CONFIDENCE:
                cv2.putText(frame, current_letter,
                            (18, 95), cv2.FONT_HERSHEY_DUPLEX, 3.5, sign_color, 5)
                detect_label = "Motion Gesture" if current_letter in ('J', 'Z') and current_language == "ASL" else "Detected Sign"
                cv2.putText(frame, detect_label,
                            (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
                draw_confidence_bar(frame, current_conf, sign_color)
            else:
                cv2.putText(frame, "---",
                            (18, 95), cv2.FONT_HERSHEY_DUPLEX, 3.5, (80, 80, 80), 4)
                cv2.putText(frame, "Show a sign (A-Z)",
                            (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)

        # ── Word builder panel (bottom) ───────────────────────────────────────
        h_frame    = frame.shape[0]
        word_panel = frame.copy()
        cv2.rectangle(word_panel, (0, h_frame - 70), (frame.shape[1], h_frame),
                      (20, 20, 20), -1)
        cv2.addWeighted(word_panel, 0.75, frame, 0.25, 0, frame)

        display_word = word if word else "_"
        cv2.putText(frame, "Word: " + display_word,
                    (10, h_frame - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 230, 180), 2)
        # Context-aware bottom hint bar
        if current_mode == "words":
            cv2.putText(frame, "SPACE=Commit | T=Speak | N=Motion word | P=Letters mode | Q=Quit",
                        (10, h_frame - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (130, 130, 130), 1)
        else:
            cv2.putText(frame, "SPACE=add  BKSP=del  C=clear  M=J/Z  T=speak  R=spell  V=voice  L=Lang  P=Words  Q=quit",
                        (10, h_frame - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.27, (130, 130, 130), 1)

        cv2.imshow(WINDOW_NAME, frame)

        # Bring window to front once so it receives keyboard focus
        if not _window_focused[0]:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(1)
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 0)
            _window_focused[0] = True

        # ── Key handling ──────────────────────────────────────────────────────
        # NOTE: click the OpenCV window first so it has keyboard focus.
        # Keys are only captured when the SignBridge AI window is active.
        try:
            key = cv2.waitKey(30) & 0xFF
        except KeyboardInterrupt:
            speak("Goodbye! Session ended.", flush=True)
            break

        if key in (ord('q'), ord('Q')):
            speak("Goodbye! Session ended.", flush=True)
            break

        # ── P KEY: Toggle Letters ↔ Words mode ────────────────────────────────
        elif key in (ord('p'), ord('P')):
            if current_mode == "letters":
                # Switch to words mode
                if not words_available:
                    print("  [MODE] Words model not found — run train_words_model.py")
                    speak("Words model not available", flush=True)
                else:
                    current_mode = "words"
                    words_prediction_buf.clear()
                    words_conf_buf.clear()
                    words_stable_count = 0
                    words_last_spoken = ""
                    words_last_detected = ""
                    speak("Words mode", flush=True)
                    print("  [MODE] Switched to Words mode")
            else:
                # Switch back to letters mode
                current_mode = "letters"
                prediction_buf.clear()
                conf_buf.clear()
                last_spoken_letter = ""
                stable_count = 0
                speak("Letters mode", flush=True)
                print("  [MODE] Switched to Letters mode")

        elif key in (ord('m'), ord('M')) and current_mode == "words":
            # M key not available in words mode — tell user to use N
            print("  Use N for motion words in words mode")
            speak("Use N for motion words", flush=True)

        elif key in (ord('m'), ord('M')) and current_mode == "letters" and current_language == "ASL" and MOTION_AVAILABLE and not MOTION_RECORDING:
            # Start motion recording for J/Z detection (ASL only)
            MOTION_RECORDING = True
            MOTION_RECORD_BUF = []
            print("  [MOTION] Recording started — perform J or Z gesture now...")
            speak("Recording. Show J or Z now.", flush=True)

        elif key in (ord('m'), ord('M')) and current_mode == "letters" and current_language == "ISL":
            # ISL handles J and Z as static signs — no motion recording needed
            print("  ISL mode: J and Z are static signs, no motion recording needed.")
            speak("J and Z are static signs in ISL", flush=True)

        # ── N KEY: Record motion word (words mode only) ──────────────────
        elif key in (ord('n'), ord('N')):
            if current_mode == "words":
                if not words_motion_available:
                    # Model not found — show error for 2 seconds
                    words_motion_result_text = "Words motion model not found"
                    words_motion_result_timer = 60  # ~2 seconds at 30fps
                    print("  [MOTION WORD] Model not found — run train_words_motion_model.py")
                elif words_motion_recording or words_motion_countdown > 0:
                    # Already recording — ignore
                    pass
                else:
                    # Start countdown → then record 30 frames
                    words_motion_buf = []
                    words_motion_recording = False
                    words_motion_countdown = 3
                    words_motion_countdown_start = _time.time()
                    words_motion_last_spoken = ""
                    print("  [MOTION WORD] Countdown started — perform motion word...")
                    speak("Get ready", flush=True)
            # In letters mode: N key does nothing silently

        elif key == ord(' '):
            # ── SPACE: commit letter or word ──────────────────────────────
            if current_mode == "words":
                # In words mode: append entire detected word to sentence
                if current_word_prediction and current_word_conf >= MIN_CONFIDENCE:
                    if word:
                        word += " " + current_word_prediction
                    else:
                        word = current_word_prediction
                    print(f"  Added word: {current_word_prediction}  |  Sentence: '{word}'")
                    speak(current_word_prediction, flush=True)
            else:
                # In letters mode: append single letter (existing behavior)
                if current_letter and current_conf >= MIN_CONFIDENCE:
                    word += current_letter
                    print(f"  Added: {current_letter}  |  Word: '{word}'")
                    speak(current_letter, flush=True)

        elif key == 8:  # BACKSPACE
            if word:
                if current_mode == "words":
                    # In words mode: remove last word (space-separated)
                    parts = word.rsplit(" ", 1)
                    removed = parts[-1] if len(parts) > 1 else word
                    word = parts[0] if len(parts) > 1 else ""
                    print(f"  Deleted word '{removed}'  |  Sentence: '{word}'")
                    speak(f"Deleted {removed}", flush=True)
                else:
                    # In letters mode: remove last letter (existing behavior)
                    removed = word[-1]
                    word = word[:-1]
                    print(f"  Deleted '{removed}'  |  Word: '{word}'")
                    speak(f"Deleted {removed}", flush=True)

        elif key in (ord('c'), ord('C')):
            print(f"  Cleared (was: '{word}')")
            speak("Cleared", flush=True)
            word = ""

        elif key in (ord('t'), ord('T')):
            if word:
                print(f"  [Voice] Speaking: '{word}'")
                speak(word, flush=True)
            else:
                speak("No word yet", flush=True)

        elif key in (ord('r'), ord('R')):
            if word:
                print(f"  [Voice] Spelling word: '{word}'")
                spell_word(word)
            else:
                speak("No word to spell", flush=True)

        elif key in (ord('v'), ord('V')):
            _voice_on[0] = not _voice_on[0]
            status = "on" if _voice_on[0] else "off"
            print(f"  Voice turned {status}")
            if _voice_on[0]:
                speak("Voice on", flush=True)

        elif key in (ord('l'), ord('L')):
            # ── Toggle ASL ↔ ISL (disabled in words mode) ─────────────
            if current_mode == "words":
                # Silently ignore L key in words mode
                pass
            elif current_language == "ASL":
                if isl_available:
                    current_language = "ISL"
                    word = ""
                    prediction_buf.clear()
                    conf_buf.clear()
                    isl_prediction_buf.clear()
                    isl_conf_buf.clear()
                    last_spoken_letter = ""
                    stable_count = 0
                    MOTION_RECORDING = False
                    MOTION_RECORD_BUF = []
                    speak("Switched to ISL", flush=True)
                    print("  [MODE] Switched to ISL")
                else:
                    print("  [MODE] ISL model not found — run train_isl_model.py")
                    speak("ISL model not available", flush=True)
            else:
                current_language = "ASL"
                word = ""
                prediction_buf.clear()
                conf_buf.clear()
                isl_prediction_buf.clear()
                isl_conf_buf.clear()
                last_spoken_letter = ""
                stable_count = 0
                speak("Switched to ASL", flush=True)
                print("  [MODE] Switched to ASL")

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()

# Send shutdown sentinel to TTS thread
_tts_queue.put(None)

print(f"\nSession ended. Final word: '{word}'")