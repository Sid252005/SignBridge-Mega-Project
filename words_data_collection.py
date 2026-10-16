"""
words_data_collection.py — SignBridge AI Common Words Data Collection
=====================================================================
Collects hand landmark data for 23 common word signs (static gestures).
Uses TWO-HAND detection since most word signs require both hands.

Each sample is a 126-number vector (same as ISL):
  - First 63 numbers: Right hand (normalized)
  - Last 63 numbers:  Left hand (normalized + mirrored)
  - Missing hand → 63 zeros (zero-padded)

Signs: HELP, PAIN, DOCTOR, POLICE, STOP, WATER, FOOD, SICK, HURT,
       MEDICINE, HUNGRY, TIRED, DIZZY, FEVER, PLEASE, WAIT, MORE,
       HOME, PHONE, MOTHER, FATHER, BATHROOM, DANGER

Controls:
  SPACE = start collecting current sign
  Q     = quit anytime (progress is saved, resumes next run)

Usage:
  python words_data_collection.py
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

from utils import normalize_isl_landmarks, normalize_landmarks, mirror_hand

# ── Sign guidance tips for each common word ───────────────────────────────────
SIGN_TIPS = {
    "HELP":      "Flat hand on fist, lift both up together",
    "PAIN":      "Index fingers point at each other, twist in opposite directions",
    "DOCTOR":    "Tap wrist (pulse point) with fingertips of D-hand",
    "POLICE":    "C-hand taps chest near badge area",
    "STOP":      "Flat hand chops down onto open palm",
    "WATER":     "W handshape, tap index finger on chin",
    "FOOD":      "Flat O-hand taps lips (fingertips to mouth)",
    "SICK":      "Middle finger taps forehead (bent 5-hand)",
    "HURT":      "Index fingers point at each other near pain area, twist",
    "MEDICINE":  "Middle finger circles on open palm (grinding motion)",
    "HUNGRY":    "Claw hand slides down chest from upper to lower",
    "TIRED":     "Bent hands on chest, fingers drop/fall forward",
    "DIZZY":     "Claw hand circles beside head",
    "FEVER":     "Back of hand rests on forehead (checking temperature)",
    "PLEASE":    "Flat hand circles on chest",
    "WAIT":      "Both hands up, palms facing you, wiggle fingers",
    "MORE":      "Flat O-hands tap together (fingertips touch)",
    "HOME":      "Flat O-hand touches cheek then jaw (eat + sleep)",
    "PHONE":     "Y-hand (thumb+pinky) held to ear",
    "MOTHER":    "Open hand, thumb taps chin",
    "FATHER":    "Open hand, thumb taps forehead",
    "BATHROOM":  "T-hand shakes side to side",
    "DANGER":    "A-hand (fist + thumb) pushes up past other hand",
}

# ── MediaPipe setup (Tasks API ONLY — never use mp.solutions) ─────────────────
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

model_path    = "hand_landmarker.task"
latest_result = None

def save_result(result, output_image, timestamp_ms):
    """Async callback — stores the latest hand detection result."""
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,                            # TWO hands — most words need both
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=save_result
)

# ── Settings ──────────────────────────────────────────────────────────────────
SIGNS = [
    "HELP", "PAIN", "DOCTOR", "POLICE", "STOP",
    "WATER", "FOOD", "SICK", "HURT", "MEDICINE",
    "HUNGRY", "TIRED", "DIZZY", "FEVER", "PLEASE",
    "WAIT", "MORE", "HOME", "PHONE", "MOTHER",
    "FATHER", "BATHROOM", "DANGER"
]
SAMPLES_PER_SIGN = 150
DATA_DIR = os.path.join("dataset", "words")
os.makedirs(DATA_DIR, exist_ok=True)       # create dataset/words/ if missing

# ── Hand skeleton connections (21 landmarks per hand) ─────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_hand_skeleton(frame, landmarks, color):
    """Draw full skeleton (lines + dots) for a single hand."""
    h, w, _ = frame.shape
    xy = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    # Connection lines
    for a, b in CONNECTIONS:
        cv2.line(frame, xy[a], xy[b], color, 2)
    # Landmark dots: white outline + colored center
    for x, y in xy:
        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 3, color, -1)


# ── Helper: extract 126 normalized landmarks from both hands ──────────────────
def extract_both_hands(result):
    """
    Extract and normalize landmarks for both hands.

    Hand identification (in horizontally-flipped frame):
      MediaPipe "Left"  = user's RIGHT hand → normalize directly
      MediaPipe "Right" = user's LEFT hand  → normalize then mirror

    Returns:
      - 126-number normalized array (right_63 + left_63), num_hands
      - Or (None, 0) if no hands detected
    """
    if not (result and result.hand_landmarks):
        return None, 0

    num_detected = len(result.hand_landmarks)
    right_raw     = np.zeros(63, dtype=np.float32)
    left_raw      = np.zeros(63, dtype=np.float32)
    left_detected = False

    for hi in range(num_detected):
        hand_lms = result.hand_landmarks[hi]
        raw_lm   = [coord for lm in hand_lms for coord in (lm.x, lm.y, lm.z)]

        # In flipped frame: MediaPipe "Left" = user's right hand
        h_label = result.handedness[hi][0].category_name
        if h_label == "Left":
            right_raw = np.array(raw_lm, dtype=np.float32)
        else:
            left_raw = np.array(raw_lm, dtype=np.float32)
            left_detected = True

    # Normalize both hands using ISL-style normalization
    combined = normalize_isl_landmarks(right_raw, left_raw)

    # Mirror the left hand part (second 63 values) if detected
    if left_detected:
        combined[63:] = mirror_hand(combined[63:])

    return combined, num_detected


# ── Camera helper ─────────────────────────────────────────────────────────────
def open_camera(index=0, retries=5, delay=1.0):
    """Try to open the camera, retrying if it's temporarily busy."""
    for attempt in range(retries):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[CAM] Camera opened on attempt {attempt + 1}")
                return cap
            cap.release()
        print(f"[CAM] Camera not ready, retrying in {delay}s... ({attempt+1}/{retries})")
        time.sleep(delay)
    print("ERROR: Could not open camera after multiple retries.")
    print("       Make sure no other app is using the camera.")
    exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────
cap = open_camera()
timestamp = 0

# Count how many signs already have saved data (for resumable progress)
already_done = sum(
    1 for s in SIGNS
    if os.path.exists(os.path.join(DATA_DIR, f"{s}.npy"))
)

print(f"\n{'='*60}")
print(f"  SignBridge AI — Common Words Data Collection (2-Hand)")
print(f"  Words to collect  : {len(SIGNS)}")
print(f"  Already collected : {already_done}/{len(SIGNS)}")
print(f"  Samples per word  : {SAMPLES_PER_SIGN}")
print(f"  Feature size      : 126 (63 per hand)")
print(f"  Data directory    : {DATA_DIR}")
print(f"{'='*60}\n")

with HandLandmarker.create_from_options(options) as landmarker:
    for idx, sign in enumerate(SIGNS):
        save_path = os.path.join(DATA_DIR, f"{sign}.npy")

        # ── Skip if this word's data already exists (resumable) ───────────
        if os.path.exists(save_path):
            existing = np.load(save_path)
            print(f"[SKIP] {sign} already collected ({len(existing)} samples).")
            continue

        collected = []
        tip = SIGN_TIPS.get(sign, "")
        print(f"\n>>> NEXT SIGN: {sign}  ({idx+1}/{len(SIGNS)})")
        print(f"    Tip: {tip}")
        print(f"    Show the sign with BOTH hands and press SPACE...")

        # ── Wait-for-spacebar loop: preview camera + show instructions ────
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"    [WARN] Camera read failed. Retrying...")
                cap.release()
                cap = open_camera()
                continue
            frame = cv2.flip(frame, 1)         # horizontal mirror (selfie view)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp += 1
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_image, timestamp)

            # Draw full skeleton for ALL detected hands (with connection lines)
            hands_status = "No hands detected"
            if latest_result and latest_result.hand_landmarks:
                num_h = len(latest_result.hand_landmarks)
                hands_status = f"Both hands visible ({num_h})" if num_h >= 2 else "One hand only"
                for hi in range(num_h):
                    h_label = latest_result.handedness[hi][0].category_name
                    # GREEN for user's right hand, BLUE for user's left
                    color = (0, 255, 100) if h_label == "Left" else (255, 180, 0)
                    draw_hand_skeleton(frame, latest_result.hand_landmarks[hi], color)

            # HUD: dark overlay + sign info
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 130), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(frame, f"Word: {sign}  ({idx+1}/{len(SIGNS)})",
                        (12, 35), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 210, 255), 2)
            cv2.putText(frame, tip,
                        (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 180), 1)
            # Show how many hands are detected
            hands_color = (0, 255, 100) if "Both" in hands_status else (0, 140, 255)
            cv2.putText(frame, hands_status,
                        (12, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hands_color, 1)
            cv2.putText(frame, "Press SPACE when ready to collect  |  Q = quit",
                        (12, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 200, 255), 1)

            cv2.imshow("SignBridge AI — Words Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):       # SPACE = start collecting
                break
            if key == ord('q'):       # Q = quit everything
                cap.release()
                cv2.destroyAllWindows()
                print("\n[QUIT] Data collection stopped by user.")
                exit(0)

        # ── Collect SAMPLES_PER_SIGN frames with normalized landmarks ─────
        print(f"    Collecting {SAMPLES_PER_SIGN} samples for {sign}...")
        while len(collected) < SAMPLES_PER_SIGN:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cap = open_camera()
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp += 1
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_image, timestamp)

            # Extract + normalize both hands into 126-dim feature vector
            combined, num_hands = extract_both_hands(latest_result)
            if combined is not None:
                collected.append(combined)

            # Draw full hand skeletons (with connection lines + dots)
            hands_status = "No hands"
            if latest_result and latest_result.hand_landmarks:
                num_h = len(latest_result.hand_landmarks)
                hands_status = f"{num_h} hand{'s' if num_h > 1 else ''}"
                for hi in range(num_h):
                    h_label = latest_result.handedness[hi][0].category_name
                    color = (0, 255, 100) if h_label == "Left" else (255, 180, 0)
                    draw_hand_skeleton(frame, latest_result.hand_landmarks[hi], color)

            # HUD overlay with progress
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 130), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(frame, f"Collecting: {sign}  ({idx+1}/{len(SIGNS)})",
                        (12, 35), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 100), 2)
            cv2.putText(frame, tip,
                        (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 180), 1)

            # Hands detected indicator
            cv2.putText(frame, hands_status,
                        (12, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1)

            # Progress bar
            bar_w = frame.shape[1] - 24
            progress = int((len(collected) / SAMPLES_PER_SIGN) * bar_w)
            cv2.rectangle(frame, (12, 72), (12 + bar_w, 95), (50, 50, 50), -1)
            cv2.rectangle(frame, (12, 72), (12 + progress, 95), (0, 220, 100), -1)
            pct_txt = f"{len(collected)}/{SAMPLES_PER_SIGN}"
            cv2.putText(frame, pct_txt,
                        (12 + bar_w - 80, 89), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

            cv2.imshow("SignBridge AI — Words Data Collection", frame)
            cv2.waitKey(1)

        # ── Save collected normalized landmarks ───────────────────────────
        np.save(save_path, np.array(collected, dtype=np.float32))
        print(f"    [OK] Saved {sign}.npy — {len(collected)} samples (126-dim)")

cap.release()
cv2.destroyAllWindows()

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Common Words Data Collection Complete!")
total = 0
for s in SIGNS:
    p = os.path.join(DATA_DIR, f"{s}.npy")
    if os.path.exists(p):
        n = len(np.load(p))
        total += n
        print(f"  {s:12s}: {n} samples")
    else:
        print(f"  {s:12s}: MISSING")
print(f"\n  Total raw samples : {total}")
print(f"  Feature size      : 126 (63 per hand)")
print(f"  Augmented (x3)    : ~{total * 3}")
print(f"{'='*60}")
print("\nNext step: python train_words_model.py")
