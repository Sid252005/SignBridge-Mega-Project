"""
backend.py — SignBridge AI FastAPI Backend
==========================================
High-performance WebSocket & REST backend providing real-time Sign Language
Inference (ASL, ISL, Common Words, and Motion Gestures) for SignBridge AI.

Key Features:
  - MediaPipe Tasks API (LIVE_STREAM mode, no mp.solutions)
  - Seamless loading of 5 models (TFLite with Keras .h5 fallback)
  - Real-time hand landmark normalization & mirroring (via utils.py)
  - Weighted temporal smoothing on predictions
  - Full WebSocket protocol (/ws) for frame streaming & UI actions
  - REST endpoints (/health, /log)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '3'

import warnings
warnings.filterwarnings('ignore')

import sys
import base64
import json
import time
import asyncio
from collections import deque, Counter

import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import mediapipe as mp

from utils import normalize_landmarks, mirror_hand, normalize_isl_landmarks

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP & CORS
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SignBridge AI Backend",
    description="Real-time Sign Language Recognition Backend API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL & ENCODER GLOBALS
# ─────────────────────────────────────────────────────────────────────────────
# Model file paths
ASL_TFLITE_PATH          = "signbridge_model.tflite"
ASL_KERAS_PATH           = "signbridge_model.h5"
ASL_ENCODER_PATH         = "label_encoder.npy"

ISL_TFLITE_PATH          = "isl_model.tflite"
ISL_KERAS_PATH           = "isl_model.h5"
ISL_ENCODER_PATH         = "isl_label_encoder.npy"

WORDS_TFLITE_PATH        = "words_model.tflite"
WORDS_KERAS_PATH         = "words_model.h5"
WORDS_ENCODER_PATH       = "words_label_encoder.npy"

MOTION_TFLITE_PATH       = "motion_model.tflite"
MOTION_KERAS_PATH        = "motion_model.h5"
MOTION_ENCODER_PATH      = "motion_label_encoder.npy"

WORDS_MOTION_TFLITE_PATH = "words_motion_model.tflite"
WORDS_MOTION_KERAS_PATH  = "words_motion_model.h5"
WORDS_MOTION_ENCODER_PATH = "words_motion_label_encoder.npy"

# Interpreters / Models
asl_interpreter          = None
asl_keras_model          = None
asl_classes              = None
asl_available            = False

isl_interpreter          = None
isl_keras_model          = None
isl_classes              = None
isl_available            = False

words_interpreter        = None
words_keras_model        = None
words_classes            = None
words_available          = False

motion_interpreter       = None
motion_keras_model       = None
motion_classes           = None
motion_available         = False

words_motion_interpreter = None
words_motion_keras_model = None
words_motion_classes     = None
words_motion_available   = False

# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE TASKS API SETUP
# ─────────────────────────────────────────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

latest_result    = None
latest_result_id = 0
landmarker       = None

def _save_result_callback(result, output_image, timestamp_ms: int):
    """MediaPipe async callback to store latest detection."""
    global latest_result, latest_result_id
    latest_result = result
    latest_result_id += 1

def init_mediapipe():
    """Initializes MediaPipe HandLandmarker with Tasks API."""
    global landmarker
    task_path = "hand_landmarker.task"
    if not os.path.exists(task_path):
        print(f"[WARNING] MediaPipe model asset '{task_path}' not found!")
        return

    try:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=task_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=_save_result_callback
        )
        landmarker = HandLandmarker.create_from_options(options)
        print("[OK] MediaPipe HandLandmarker initialized (Tasks API - LIVE_STREAM).")
    except Exception as e:
        print(f"[ERROR] Failed to initialize MediaPipe HandLandmarker: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ALL 5 MODELS AT STARTUP
# ─────────────────────────────────────────────────────────────────────────────
def _load_single_model(tflite_path, keras_path, encoder_path, name):
    """
    Helper to load a TFLite model or fallback to Keras .h5, along with label encoder.
    Returns (interpreter, keras_model, classes, status_str, is_available)
    """
    interpreter = None
    keras_model = None
    classes     = None
    status_str  = "NOT FOUND"
    is_avail    = False

    # 1. Load label encoder
    if os.path.exists(encoder_path):
        try:
            classes = np.load(encoder_path, allow_pickle=True)
        except Exception as e:
            print(f"[WARNING] Error loading {encoder_path}: {e}")

    # 2. Try TFLite interpreter
    if os.path.exists(tflite_path):
        try:
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            status_str = "TFLite"
            is_avail   = True
        except Exception as e:
            print(f"[WARNING] Failed to load {tflite_path}: {e}")

    # 3. Fallback to Keras .h5
    if not is_avail and os.path.exists(keras_path):
        try:
            keras_model = tf.keras.models.load_model(keras_path)
            status_str  = "Keras .h5"
            is_avail    = True
        except Exception as e:
            print(f"[WARNING] Failed to load {keras_path}: {e}")

    # If classes missing, disable model
    if is_avail and (classes is None or len(classes) == 0):
        print(f"[WARNING] Model {name} loaded but encoder {encoder_path} is missing!")
        is_avail = False
        status_str = "NO ENCODER"

    return interpreter, keras_model, classes, status_str, is_avail


def load_all_models():
    """Loads all 5 models and encoders, printing a formatted startup status table."""
    global asl_interpreter, asl_keras_model, asl_classes, asl_available
    global isl_interpreter, isl_keras_model, isl_classes, isl_available
    global words_interpreter, words_keras_model, words_classes, words_available
    global motion_interpreter, motion_keras_model, motion_classes, motion_available
    global words_motion_interpreter, words_motion_keras_model, words_motion_classes, words_motion_available

    print("\n" + "=" * 62)
    print("           SignBridge AI — Loading Models...")
    print("=" * 62)

    # 1. ASL Static Model
    asl_interpreter, asl_keras_model, asl_classes, asl_stat, asl_available = \
        _load_single_model(ASL_TFLITE_PATH, ASL_KERAS_PATH, ASL_ENCODER_PATH, "ASL Static")

    # 2. ISL Model
    isl_interpreter, isl_keras_model, isl_classes, isl_stat, isl_available = \
        _load_single_model(ISL_TFLITE_PATH, ISL_KERAS_PATH, ISL_ENCODER_PATH, "ISL")

    # 3. Words Static Model
    words_interpreter, words_keras_model, words_classes, words_stat, words_available = \
        _load_single_model(WORDS_TFLITE_PATH, WORDS_KERAS_PATH, WORDS_ENCODER_PATH, "Words Static")

    # 4. Motion Letters (J/Z)
    motion_interpreter, motion_keras_model, motion_classes, motion_stat, motion_available = \
        _load_single_model(MOTION_TFLITE_PATH, MOTION_KERAS_PATH, MOTION_ENCODER_PATH, "Motion J/Z")

    # 5. Motion Words
    words_motion_interpreter, words_motion_keras_model, words_motion_classes, wm_stat, words_motion_available = \
        _load_single_model(WORDS_MOTION_TFLITE_PATH, WORDS_MOTION_KERAS_PATH, WORDS_MOTION_ENCODER_PATH, "Words Motion")

    # Status Table Display
    print(f"{'Model Name':<24} | {'Format':<10} | {'Status':<10} | {'Classes':<8}")
    print("-" * 62)
    models_info = [
        ("ASL Letters (A-Y)", asl_stat, "READY" if asl_available else "OFF", len(asl_classes) if asl_classes is not None else 0),
        ("ISL Letters (A-Z)", isl_stat, "READY" if isl_available else "OFF", len(isl_classes) if isl_classes is not None else 0),
        ("Words Static (23)", words_stat, "READY" if words_available else "OFF", len(words_classes) if words_classes is not None else 0),
        ("Motion Letters (J/Z)", motion_stat, "READY" if motion_available else "OFF", len(motion_classes) if motion_classes is not None else 0),
        ("Motion Words (7)", wm_stat, "READY" if words_motion_available else "OFF", len(words_motion_classes) if words_motion_classes is not None else 0),
    ]
    for name, fmt, st, cls_cnt in models_info:
        print(f"{name:<24} | {fmt:<10} | {st:<10} | {cls_cnt:<8}")
    print("=" * 62 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION STATE & SMOOTHING BUFFERS
# ─────────────────────────────────────────────────────────────────────────────
current_language = "ASL"   # "ASL" or "ISL"
current_mode     = "letters"   # "letters" or "words"
current_word     = ""          # word being built
conversation_log = []          # list of {time, text} dicts

SMOOTHING_WINDOW = 15
asl_pred_buf     = deque(maxlen=SMOOTHING_WINDOW)
asl_conf_buf     = deque(maxlen=SMOOTHING_WINDOW)

isl_pred_buf     = deque(maxlen=SMOOTHING_WINDOW)
isl_conf_buf     = deque(maxlen=SMOOTHING_WINDOW)

words_pred_buf   = deque(maxlen=SMOOTHING_WINDOW)
words_conf_buf   = deque(maxlen=SMOOTHING_WINDOW)

frame_timestamp_counter = 0

# Thresholds
WORDS_CONFIDENCE_THRESHOLD  = 0.75   # Static words threshold (0.0 to 1.0)
MOTION_CONFIDENCE_THRESHOLD = 0.65   # Motion words threshold (0.0 to 1.0)

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def run_tflite(interpreter, input_data: np.ndarray) -> np.ndarray:
    """Executes a forward pass on a TFLite interpreter."""
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(
        input_details[0]['index'],
        input_data.astype(np.float32)
    )
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])


def predict_letter(result, language: str = "ASL") -> tuple[str, float]:
    """
    Extract landmarks from MediaPipe result, normalize, mirror if left hand,
    run through ASL or ISL model, and apply weighted majority vote smoothing.
    Returns (letter, confidence) where confidence is in 0.0 - 1.0, or ("", 0.0).
    """
    if not (result and result.hand_landmarks and len(result.hand_landmarks) > 0):
        return ("", 0.0)

    if language == "ISL":
        if not isl_available:
            return ("", 0.0)

        num_detected = len(result.hand_landmarks)
        right_raw = np.zeros(63, dtype=np.float32)
        left_raw  = np.zeros(63, dtype=np.float32)
        left_detected = False

        for hi in range(num_detected):
            hand_lms = result.hand_landmarks[hi]
            raw_lm   = [coord for lm in hand_lms for coord in (lm.x, lm.y, lm.z)]
            h_label  = result.handedness[hi][0].display_name if result.handedness else "Right"
            if h_label == "Left":       # In mirrored camera frame, "Left" is user's right hand
                right_raw = np.array(raw_lm, dtype=np.float32)
            else:                       # "Right" is user's left hand
                left_raw = np.array(raw_lm, dtype=np.float32)
                left_detected = True

        isl_input = normalize_isl_landmarks(right_raw, left_raw)
        if left_detected:
            isl_input[63:] = mirror_hand(isl_input[63:])

        inp = np.array([isl_input], dtype=np.float32)
        if isl_interpreter is not None:
            probs = run_tflite(isl_interpreter, inp)[0]
        elif isl_keras_model is not None:
            probs = isl_keras_model.predict(inp, verbose=0)[0]
        else:
            return ("", 0.0)

        idx  = int(np.argmax(probs))
        letter = isl_classes[idx]
        conf   = float(probs[idx])

        # Weighted smoothing
        isl_pred_buf.append(letter)
        isl_conf_buf.append(conf)

        vote_counter = Counter()
        for vi, vl in enumerate(isl_pred_buf):
            vote_counter[vl] += 1.0 + vi * 0.15
        smoothed_letter = vote_counter.most_common(1)[0][0]

        conf_list = list(isl_conf_buf)
        weights   = np.array([1.0 + i * 0.15 for i in range(len(conf_list))])
        smoothed_conf = float(np.average(conf_list, weights=weights))

        return smoothed_letter, smoothed_conf

    else:  # ASL Mode (A-Y static)
        if not asl_available:
            return ("", 0.0)

        hand_lms = result.hand_landmarks[0]
        raw      = [coord for lm in hand_lms for coord in (lm.x, lm.y, lm.z)]
        norm_data = normalize_landmarks(raw)

        # Handedness check: if left hand, mirror along X to match right-hand trained model
        if result.handedness and len(result.handedness) > 0:
            h_label = result.handedness[0][0].display_name
            if h_label == "Right":  # In mirrored frame, "Right" tag is user's left hand
                norm_data = mirror_hand(norm_data)

        inp = np.array([norm_data], dtype=np.float32)
        if asl_interpreter is not None:
            probs = run_tflite(asl_interpreter, inp)[0]
        elif asl_keras_model is not None:
            probs = asl_keras_model.predict(inp, verbose=0)[0]
        else:
            return ("", 0.0)

        idx  = int(np.argmax(probs))
        letter = asl_classes[idx]
        conf   = float(probs[idx])

        # Weighted smoothing
        asl_pred_buf.append(letter)
        asl_conf_buf.append(conf)

        vote_counter = Counter()
        for vi, vl in enumerate(asl_pred_buf):
            vote_counter[vl] += 1.0 + vi * 0.15
        smoothed_letter = vote_counter.most_common(1)[0][0]

        conf_list = list(asl_conf_buf)
        weights   = np.array([1.0 + i * 0.15 for i in range(len(conf_list))])
        smoothed_conf = float(np.average(conf_list, weights=weights))

        return smoothed_letter, smoothed_conf


def predict_words_static(result) -> tuple[str, float]:
    """
    Two-hand 126-dim Common Words static prediction.
    Applies normalize_isl_landmarks, mirrors left hand if detected,
    runs inference, applies weighted smoothing, and enforces 0.75 confidence threshold.
    Returns (word, confidence) or ("", 0.0) if below threshold.
    """
    if not (result and result.hand_landmarks and len(result.hand_landmarks) > 0):
        return ("", 0.0)
    if not words_available:
        return ("", 0.0)

    num_detected = len(result.hand_landmarks)
    right_raw = np.zeros(63, dtype=np.float32)
    left_raw  = np.zeros(63, dtype=np.float32)
    left_detected = False

    for hi in range(num_detected):
        hand_lms = result.hand_landmarks[hi]
        raw_lm   = [coord for lm in hand_lms for coord in (lm.x, lm.y, lm.z)]
        h_label  = result.handedness[hi][0].display_name if result.handedness else "Right"
        if h_label == "Left":
            right_raw = np.array(raw_lm, dtype=np.float32)
        else:
            left_raw = np.array(raw_lm, dtype=np.float32)
            left_detected = True

    words_input = normalize_isl_landmarks(right_raw, left_raw)
    if left_detected:
        words_input[63:] = mirror_hand(words_input[63:])

    inp = np.array([words_input], dtype=np.float32)
    if words_interpreter is not None:
        probs = run_tflite(words_interpreter, inp)[0]
    elif words_keras_model is not None:
        probs = words_keras_model.predict(inp, verbose=0)[0]
    else:
        return ("", 0.0)

    idx  = int(np.argmax(probs))
    word = words_classes[idx]
    conf = float(probs[idx])

    # Suppress low-confidence predictions
    if conf < WORDS_CONFIDENCE_THRESHOLD:
        return ("", 0.0)

    words_pred_buf.append(word)
    words_conf_buf.append(conf)

    vote_counter = Counter()
    for vi, vl in enumerate(words_pred_buf):
        vote_counter[vl] += 1.0 + vi * 0.15
    smoothed_word = vote_counter.most_common(1)[0][0]

    conf_list = list(words_conf_buf)
    weights   = np.array([1.0 + i * 0.15 for i in range(len(conf_list))])
    smoothed_conf = float(np.average(conf_list, weights=weights))

    return smoothed_word, smoothed_conf


def predict_words_motion(sequence_30_frames: list) -> tuple[str, float]:
    """
    Inference for a motion sequence.
    Resamples sequences (>= 15 frames) to exactly 30 frames using linear interpolation.
    In words mode: classifies into 7 motion words (words_motion_model).
    In letters mode: classifies into J / Z (motion_model).
    Tests both direct orientation and mirrored orientation for robust dual-hand recognition.
    Returns (word_or_letter, confidence) where confidence is 0.0 - 1.0.
    """
    if not sequence_30_frames or len(sequence_30_frames) < 15:
        return ("", 0.0)

    n_frames = len(sequence_30_frames)
    seq_np = np.array(sequence_30_frames, dtype=np.float32)

    # Resample to exactly 30 frames if not already 30
    if n_frames != 30:
        old_t = np.linspace(0, 1, n_frames)
        new_t = np.linspace(0, 1, 30)
        resampled = np.zeros((30, 63), dtype=np.float32)
        for j in range(63):
            resampled[:, j] = np.interp(new_t, old_t, seq_np[:, j])
        seq_arr = resampled.reshape(1, 30, 63)
    else:
        seq_arr = seq_np[:30].reshape(1, 30, 63)

    if current_mode == "words" and words_motion_available:
        def _eval_motion_words(inp_seq):
            if words_motion_interpreter is not None:
                return run_tflite(words_motion_interpreter, inp_seq)[0]
            elif words_motion_keras_model is not None:
                return words_motion_keras_model.predict(inp_seq, verbose=0)[0]
            return None

        probs = _eval_motion_words(seq_arr)
        if probs is None:
            return ("", 0.0)

        idx  = int(np.argmax(probs))
        conf = float(probs[idx])

        # Dual-hand check: if confidence is below threshold, test mirrored hand orientation along X
        if conf < MOTION_CONFIDENCE_THRESHOLD:
            mirrored_seq = seq_arr.copy()
            mirrored_seq[:, :, 0::3] = -mirrored_seq[:, :, 0::3]
            probs_m = _eval_motion_words(mirrored_seq)
            if probs_m is not None:
                idx_m = int(np.argmax(probs_m))
                conf_m = float(probs_m[idx_m])
                if conf_m > conf:
                    idx = idx_m
                    conf = conf_m

        if conf >= MOTION_CONFIDENCE_THRESHOLD:
            return words_motion_classes[idx], conf
        return ("", conf)

    elif motion_available:
        def _eval_motion_letters(inp_seq):
            if motion_interpreter is not None:
                return run_tflite(motion_interpreter, inp_seq)[0]
            elif motion_keras_model is not None:
                return motion_keras_model.predict(inp_seq, verbose=0)[0]
            return None

        probs = _eval_motion_letters(seq_arr)
        if probs is None:
            return ("", 0.0)

        idx  = int(np.argmax(probs))
        conf = float(probs[idx])

        if conf < 0.65:
            mirrored_seq = seq_arr.copy()
            mirrored_seq[:, :, 0::3] = -mirrored_seq[:, :, 0::3]
            probs_m = _eval_motion_letters(mirrored_seq)
            if probs_m is not None:
                idx_m = int(np.argmax(probs_m))
                conf_m = float(probs_m[idx_m])
                if conf_m > conf:
                    idx = idx_m
                    conf = conf_m

        if conf >= 0.65:
            return motion_classes[idx], conf
        return ("", conf)

    return ("", 0.0)

# ─────────────────────────────────────────────────────────────────────────────
# AUTOMATIC EMERGENCY (SOS) DETECTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
sos_wrist_history        = deque(maxlen=60)
sos_wave_confirm_count   = 0
sos_help_sign_counter    = 0
sos_cooldown_until       = 0.0

# Letter 'S' 3-repetition detector state
sos_s_reps               = 0      # Current confirmed repetition count (0, 1, 2)
sos_s_hold_count         = 0      # Consecutive frames holding 'S'
sos_s_release_count      = 0      # Consecutive frames releasing 'S'
sos_s_in_hold            = False  # Whether currently locked into an active hold
sos_s_start_time         = 0.0    # Timestamp when rep 1 was confirmed
SOS_S_WINDOW_SECONDS     = 10.0   # Window to complete all 3 repetitions
SOS_S_MIN_HOLD_FRAMES    = 6      # ~0.6s hold at 10fps
SOS_S_MIN_RELEASE_FRAMES = 4      # ~0.4s release at 10fps
SOS_S_MIN_CONFIDENCE     = 0.70   # Min confidence to register 'S'


def update_sos_s_detector(is_s_detected: bool, conf: float) -> tuple[int, bool]:
    """
    State machine detecting 3 distinct repetitions of the letter 'S' sign.
    Requires holding 'S' for >= 6 frames, releasing for >= 4 frames, done 3 times within 10s.
    Returns (current_reps, triggered_bool).
    """
    global sos_s_reps, sos_s_hold_count, sos_s_release_count, sos_s_in_hold, sos_s_start_time

    now = time.time()

    # Reset if total completion window exceeded
    if sos_s_reps > 0 and (now - sos_s_start_time > SOS_S_WINDOW_SECONDS):
        sos_s_reps = 0
        sos_s_hold_count = 0
        sos_s_release_count = 0
        sos_s_in_hold = False
        sos_s_start_time = 0.0

    valid_s = is_s_detected and (conf >= SOS_S_MIN_CONFIDENCE)

    if valid_s:
        sos_s_release_count = 0
        if not sos_s_in_hold:
            sos_s_hold_count += 1
            if sos_s_hold_count >= SOS_S_MIN_HOLD_FRAMES:
                # Confirmed a repetition!
                sos_s_reps += 1
                sos_s_in_hold = True
                sos_s_hold_count = 0
                if sos_s_reps == 1:
                    sos_s_start_time = now

                print(f"[SOS-S] Repetition {sos_s_reps}/3 confirmed (conf: {conf:.2f})")

                if sos_s_reps >= 3:
                    # 3 repetitions achieved!
                    sos_s_reps = 0
                    sos_s_in_hold = False
                    sos_s_start_time = 0.0
                    return 3, True
    else:
        sos_s_hold_count = 0
        if sos_s_in_hold:
            sos_s_release_count += 1
            if sos_s_release_count >= SOS_S_MIN_RELEASE_FRAMES:
                # User released the sign, ready for next repetition
                sos_s_in_hold = False
                sos_s_release_count = 0

    return sos_s_reps, False


def check_emergency_wave(result) -> bool:
    """
    Detects rapid lateral hand waving distress signal.
    Requires >= 4 direction reversals and amplitude >= 0.15 sustained over multiple frames.
    """
    global sos_wave_confirm_count
    if not result or not result.hand_landmarks or len(result.hand_landmarks) == 0:
        sos_wave_confirm_count = max(0, sos_wave_confirm_count - 1)
        return False

    now = time.time()
    wrist = result.hand_landmarks[0][0]
    sos_wrist_history.append((wrist.x, now))

    cutoff = now - 2.5
    recent = [pt for pt in sos_wrist_history if pt[1] >= cutoff]

    if len(recent) < 12:
        return False

    xs = [pt[0] for pt in recent]
    amplitude = max(xs) - min(xs)
    if amplitude < 0.15:
        sos_wave_confirm_count = max(0, sos_wave_confirm_count - 1)
        return False

    reversals = 0
    dx_sign = 0
    deadband = 0.02
    for i in range(1, len(xs)):
        diff = xs[i] - xs[i - 1]
        if abs(diff) > deadband:
            new_sign = 1 if diff > 0 else -1
            if dx_sign != 0 and new_sign != dx_sign:
                reversals += 1
            dx_sign = new_sign

    if reversals >= 4:
        sos_wave_confirm_count += 1
        if sos_wave_confirm_count >= 14:  # ~1.4s sustained waving at 10fps
            sos_wave_confirm_count = 0
            sos_wrist_history.clear()
            return True
    else:
        sos_wave_confirm_count = max(0, sos_wave_confirm_count - 1)

    return False

# ─────────────────────────────────────────────────────────────────────────────
# REST HTTP ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Health check endpoint exposing system & model availability status."""
    return {
        "status": "ok",
        "models": {
            "asl": asl_available,
            "isl": isl_available,
            "words": words_available,
            "motion": motion_available,
            "words_motion": words_motion_available
        },
        "current_language": current_language,
        "current_mode": current_mode,
        "active_word": current_word
    }


@app.get("/log")
async def get_conversation_log():
    """Returns conversation log history as JSON."""
    return conversation_log


@app.delete("/log")
async def clear_conversation_log():
    """Clears conversation log history."""
    global conversation_log
    conversation_log = []
    return {"status": "ok", "message": "Conversation log cleared"}

# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET ENDPOINT — /ws
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main real-time communication channel.
    Accepts webcam frames, performs MediaPipe detection and ML inference,
    and handles user actions from the UI.
    """
    global current_language, current_mode, current_word, conversation_log
    global frame_timestamp_counter, sos_cooldown_until, sos_help_sign_counter
    global sos_s_reps, sos_s_hold_count, sos_s_release_count, sos_s_in_hold

    await websocket.accept()

    last_sent_result_id = -1

    # Send initial state snapshot on connection
    await websocket.send_text(json.dumps({
        "type": "state",
        "current_word": current_word,
        "conversation_log": conversation_log,
        "current_language": current_language,
        "current_mode": current_mode,
        "sos": False
    }))

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except Exception:
                continue

            msg_type = message.get("type")

            # ── 1. FRAME MESSAGE: Webcam frame from React ────────────────────
            if msg_type == "frame":
                img_data_str = message.get("data", "")
                if not img_data_str:
                    continue

                if "," in img_data_str:
                    img_data_str = img_data_str.split(",")[1]

                try:
                    img_bytes = base64.b64decode(img_data_str)
                    nparr     = np.frombuffer(img_bytes, np.uint8)
                    frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception:
                    continue

                if frame is None:
                    continue

                # MediaPipe Tasks API async detection
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_timestamp_counter += 1
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                if landmarker is not None:
                    landmarker.detect_async(mp_image, frame_timestamp_counter)
                    # Tiny non-blocking yield to allow the background callback to fire
                    await asyncio.sleep(0.002)

                # Check if result is fresh from MediaPipe callback
                is_fresh_result = (latest_result_id != last_sent_result_id)
                if is_fresh_result:
                    last_sent_result_id = latest_result_id

                # Run prediction based on current mode
                if current_mode == "letters":
                    letter, conf = predict_letter(latest_result, current_language)
                elif current_mode == "words":
                    letter, conf = predict_words_static(latest_result)
                else:
                    letter, conf = "", 0.0

                conf_percent = round(conf * 100, 1) if conf <= 1.0 else round(conf, 1)

                # Only include landmarks if result is fresh — prevents duplicate frames in motion buffer
                landmarks_for_client = None
                if is_fresh_result and latest_result and latest_result.hand_landmarks and len(latest_result.hand_landmarks) > 0:
                    hand_lms = latest_result.hand_landmarks[0]
                    raw_lm = [coord for lm in hand_lms for coord in (lm.x, lm.y, lm.z)]
                    norm_lm = normalize_landmarks(raw_lm)
                    if latest_result.handedness and len(latest_result.handedness) > 0:
                        h_name = getattr(latest_result.handedness[0][0], 'display_name', '') or getattr(latest_result.handedness[0][0], 'category_name', '')
                        if h_name == "Left":
                            norm_lm = mirror_hand(norm_lm)
                    landmarks_for_client = norm_lm.tolist()

                # Check for 'S' sign for SOS 3-repetition detector
                is_s_sign = False
                s_sign_conf = 0.0
                if current_mode == "letters":
                    if letter == "S":
                        is_s_sign = True
                        s_sign_conf = conf
                elif latest_result and latest_result.hand_landmarks and len(latest_result.hand_landmarks) > 0:
                    # In Words mode, evaluate ASL letter model in background for 'S' emergency sign
                    s_let, s_cf = predict_letter(latest_result, "ASL")
                    if s_let == "S":
                        is_s_sign = True
                        s_sign_conf = s_cf

                # Automatic SOS Gesture & Sign Detection
                auto_sos = False
                sos_reason = "EMERGENCY - HELP NEEDED (AUTO-DETECTED)"
                current_s_reps = 0
                now_ts = time.time()
                if now_ts >= sos_cooldown_until:
                    # 1. Primary SOS Trigger: Letter 'S' sign performed 3 distinct times
                    current_s_reps, s_3_triggered = update_sos_s_detector(is_s_sign, s_sign_conf)
                    if s_3_triggered:
                        auto_sos = True
                        sos_reason = "EMERGENCY - HELP NEEDED (S SIGN x3)"
                    # 2. Universal rapid lateral wave distress gesture
                    elif check_emergency_wave(latest_result):
                        auto_sos = True
                        sos_reason = "EMERGENCY - HELP NEEDED (WAVE DETECTED)"
                    # 3. Words mode sustained high-confidence HELP sign
                    elif current_mode == "words" and letter == "HELP" and conf >= 0.85:
                        sos_help_sign_counter += 1
                        if sos_help_sign_counter >= 20:  # ~2 seconds sustained at 10fps
                            auto_sos = True
                            sos_help_sign_counter = 0
                            sos_reason = "EMERGENCY - HELP NEEDED (HELP SIGN)"
                    else:
                        sos_help_sign_counter = max(0, sos_help_sign_counter - 1)
                else:
                    sos_help_sign_counter = 0
                    current_s_reps = 0

                if auto_sos:
                    t_str = time.strftime("%H:%M")
                    conversation_log.append({
                        "time": t_str,
                        "text": sos_reason
                    })
                    sos_cooldown_until = time.time() + 60.0
                    print(f"[SOS] {sos_reason} at {t_str}")
                    await websocket.send_text(json.dumps({
                        "type": "state",
                        "current_word": current_word,
                        "conversation_log": conversation_log,
                        "current_language": current_language,
                        "current_mode": current_mode,
                        "sos": True,
                        "sos_progress": 0
                    }))

                await websocket.send_text(json.dumps({
                    "type": "prediction",
                    "letter": letter,
                    "confidence": conf_percent,
                    "word": current_word,
                    "mode": current_mode,
                    "language": current_language,
                    "hand_detected": bool(
                        latest_result and
                        latest_result.hand_landmarks and
                        len(latest_result.hand_landmarks) > 0
                    ),
                    "landmarks": landmarks_for_client,
                    "sos_progress": current_s_reps
                }))

            # ── 2. ACTION MESSAGE: User button / gesture action ──────────────
            elif msg_type == "action":
                action = message.get("action", "")
                sos_flag = False

                if action == "add_letter":
                    letter_to_add = message.get("letter")
                    if not letter_to_add and latest_result:
                        if current_mode == "letters":
                            det_letter, _ = predict_letter(latest_result, current_language)
                        else:
                            det_letter, _ = predict_words_static(latest_result)
                        if det_letter:
                            letter_to_add = det_letter
                    if letter_to_add:
                        current_word += str(letter_to_add)

                elif action == "backspace":
                    if len(current_word) > 0:
                        current_word = current_word[:-1]

                elif action == "clear":
                    current_word = ""

                elif action == "set_word":
                    current_word = message.get("word", "")

                elif action == "speak":
                    if current_word:
                        t_str = time.strftime("%H:%M")
                        conversation_log.append({
                            "time": t_str,
                            "text": current_word
                        })

                elif action == "switch_language":
                    if current_mode == "letters":
                        current_language = "ISL" if current_language == "ASL" else "ASL"
                        asl_pred_buf.clear()
                        asl_conf_buf.clear()
                        isl_pred_buf.clear()
                        isl_conf_buf.clear()

                elif action == "switch_mode":
                    current_mode = "words" if current_mode == "letters" else "letters"
                    current_word = ""
                    words_pred_buf.clear()
                    words_conf_buf.clear()
                    asl_pred_buf.clear()
                    asl_conf_buf.clear()

                elif action == "save_log":
                    t_str = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"conversation_{t_str}.txt"
                    try:
                        with open(filename, "w", encoding="utf-8") as f:
                            for entry in conversation_log:
                                f.write(f"[{entry.get('time', '')}] {entry.get('text', '')}\n")
                        print(f"[OK] Saved conversation log to {filename}")
                    except Exception as e:
                        print(f"[ERROR] Could not save log file: {e}")

                elif action == "sos":
                    t_str = time.strftime("%H:%M")
                    conversation_log.append({
                        "time": t_str,
                        "text": "EMERGENCY - HELP NEEDED"
                    })
                    sos_cooldown_until = time.time() + 60.0
                    sos_s_reps = 0
                    sos_s_hold_count = 0
                    sos_s_release_count = 0
                    sos_s_in_hold = False
                    sos_flag = True

                elif action == "motion_sequence":
                    sequence = message.get("sequence", message.get("data", []))
                    pred_word, m_conf = predict_words_motion(sequence) if sequence else ("", 0.0)
                    conf_pct = round(m_conf * 100, 1) if m_conf <= 1.0 else round(m_conf, 1)

                    if pred_word:
                        current_word = pred_word
                        words_pred_buf.clear()
                        words_conf_buf.clear()
                        asl_pred_buf.clear()
                        asl_conf_buf.clear()
                        t_str = time.strftime("%H:%M")
                        conversation_log.append({
                            "time": t_str,
                            "text": pred_word
                        })
                        print(f"[MOTION RECOGNIZED] Detected '{pred_word}' with {conf_pct}% confidence")
                        await websocket.send_text(json.dumps({
                            "type": "prediction",
                            "letter": pred_word,
                            "confidence": conf_pct,
                            "word": current_word,
                            "mode": current_mode,
                            "language": current_language,
                            "hand_detected": True,
                            "motion_result": pred_word,
                            "motion_status": "success"
                        }))
                    else:
                        print(f"[MOTION UNRECOGNIZED] Below threshold or insufficient frames ({conf_pct}%)")
                        await websocket.send_text(json.dumps({
                            "type": "prediction",
                            "letter": "",
                            "confidence": conf_pct,
                            "word": current_word,
                            "mode": current_mode,
                            "language": current_language,
                            "hand_detected": True,
                            "motion_result": "Not recognized",
                            "motion_status": "failed"
                        }))

                elif action == "delete_log_entry":
                    idx = message.get("index")
                    if idx is not None and 0 <= idx < len(conversation_log):
                        conversation_log.pop(idx)

                # Send updated state back to client
                await websocket.send_text(json.dumps({
                    "type": "state",
                    "current_word": current_word,
                    "conversation_log": conversation_log,
                    "current_language": current_language,
                    "current_mode": current_mode,
                    "sos": sos_flag
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket Notice] Connection closed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP & SHUTDOWN LIFECYCLE
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Initializes models and MediaPipe landmarker on server boot."""
    load_all_models()
    init_mediapipe()
    print(">>> SignBridge AI Backend is running and ready for connections! <<<")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanly releases MediaPipe resources."""
    global landmarker
    if landmarker is not None:
        try:
            landmarker.close()
            print("[OK] MediaPipe landmarker closed.")
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# SERVE PRODUCTION FRONTEND (if signbridge-ui/dist exists)
# ─────────────────────────────────────────────────────────────────────────────
dist_path = os.path.join(os.path.dirname(__file__), "signbridge-ui", "dist")
if os.path.exists(dist_path):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=dist_path, html=True), name="static")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

