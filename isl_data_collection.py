"""
isl_data_collection.py -- SignBridge AI ISL Data Collection
=============================================================
Collects Indian Sign Language (ISL) fingerspelling data.
Unlike ASL, ISL treats ALL 26 letters as static poses (including J and Z).
Uses TWO-HAND detection for ISL signs that involve both hands.

Each sample is a 126-number vector:
  - First 63 numbers: Right hand (normalized)
  - Last 63 numbers:  Left hand (normalized + mirrored)
  - Missing hand → 63 zeros

Controls:
  SPACE = Start collecting current sign
  Q     = Quit anytime (progress is saved)
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

from utils import normalize_isl_landmarks, normalize_landmarks, mirror_hand

# ── ISL sign guidance tips ────────────────────────────────────
ISL_SIGN_TIPS = {
    'A': "ISL sign for A — refer to ISL fingerspelling chart",
    'B': "ISL sign for B — refer to ISL fingerspelling chart",
    'C': "ISL sign for C — refer to ISL fingerspelling chart",
    'D': "ISL sign for D — refer to ISL fingerspelling chart",
    'E': "ISL sign for E — refer to ISL fingerspelling chart",
    'F': "ISL sign for F — refer to ISL fingerspelling chart",
    'G': "ISL sign for G — refer to ISL fingerspelling chart",
    'H': "ISL sign for H — refer to ISL fingerspelling chart",
    'I': "ISL sign for I — refer to ISL fingerspelling chart",
    'J': "ISL sign for J — static pose (unlike ASL motion)",
    'K': "ISL sign for K — refer to ISL fingerspelling chart",
    'L': "ISL sign for L — refer to ISL fingerspelling chart",
    'M': "ISL sign for M — refer to ISL fingerspelling chart",
    'N': "ISL sign for N — refer to ISL fingerspelling chart",
    'O': "ISL sign for O — refer to ISL fingerspelling chart",
    'P': "ISL sign for P — refer to ISL fingerspelling chart",
    'Q': "ISL sign for Q — refer to ISL fingerspelling chart",
    'R': "ISL sign for R — refer to ISL fingerspelling chart",
    'S': "ISL sign for S — refer to ISL fingerspelling chart",
    'T': "ISL sign for T — refer to ISL fingerspelling chart",
    'U': "ISL sign for U — refer to ISL fingerspelling chart",
    'V': "ISL sign for V — refer to ISL fingerspelling chart",
    'W': "ISL sign for W — refer to ISL fingerspelling chart",
    'X': "ISL sign for X — refer to ISL fingerspelling chart",
    'Y': "ISL sign for Y — refer to ISL fingerspelling chart",
    'Z': "ISL sign for Z — static pose (unlike ASL motion)",
}

# ── MediaPipe setup (Tasks API ONLY — never mp.solutions) ────
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

model_path = "hand_landmarker.task"
latest_result = None

def save_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,                  # ISL needs both hands
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=save_result
)

# ── Settings ─────────────────────────────────────────────────
SIGNS            = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")   # All 26 letters for ISL
SAMPLES_PER_SIGN = 200
DATA_DIR         = os.path.join("dataset", "ISL")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Hand skeleton connections ────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_hand_skeleton(frame, landmarks, color):
    """Draw skeleton for a single hand with specified color."""
    h, w, _ = frame.shape
    xy = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, xy[a], xy[b], color, 2)
    for x, y in xy:
        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 3, color, -1)


# ── Helper: extract 126 ISL numbers from both hands ──────────
def extract_isl_landmarks(result):
    """
    Extract and normalize landmarks for ISL (both hands).

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
    right_raw    = np.zeros(63, dtype=np.float32)
    left_raw     = np.zeros(63, dtype=np.float32)
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

    # Normalize both hands using normalize_isl_landmarks
    combined = normalize_isl_landmarks(right_raw, left_raw)

    # Mirror the left hand part (second 63 values) if detected
    if left_detected:
        combined[63:] = mirror_hand(combined[63:])

    return combined, num_detected


# ── Main ──────────────────────────────────────────────────────
def open_camera(index=0, retries=5, delay=1.0):
    """Try to open the camera, retrying if it's temporarily busy."""
    import time
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
    print("       Make sure no other app (OBS, Teams, previous script) is using it.")
    exit(1)

cap = open_camera()
timestamp = 0

# Count how many are already done
already_done = sum(1 for s in SIGNS
                   if os.path.exists(os.path.join(DATA_DIR, f"{s}.npy")))

print(f"\n{'='*60}")
print(f"  SignBridge AI — ISL Data Collection (A-Z, 26 Signs)")
print(f"  Already collected: {already_done}/{len(SIGNS)} signs")
print(f"  Samples per sign : {SAMPLES_PER_SIGN}")
print(f"  NOTE: ISL treats ALL 26 letters as static poses")
print(f"{'='*60}\n")

with HandLandmarker.create_from_options(options) as landmarker:
    for idx, sign in enumerate(SIGNS):
        save_path = os.path.join(DATA_DIR, f"{sign}.npy")

        # Skip if already collected (resume support)
        if os.path.exists(save_path):
            existing = np.load(save_path)
            print(f"[SKIP] {sign} already collected ({len(existing)} samples).")
            continue

        collected = []
        tip = ISL_SIGN_TIPS.get(sign, "")
        print(f"\n>>> GET READY FOR ISL SIGN: {sign}  ({idx+1}/{len(SIGNS)})")
        print(f"    Tip: {tip}")
        print(f"    Show the sign and press SPACE to start collecting...")

        # ── Wait for spacebar ────────────────────────────────
        space_pressed = False
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"    [WARN] Camera read failed while waiting for sign {sign}. Retrying camera...")
                cap.release()
                cap = open_camera()
                continue
            frame = cv2.flip(frame, 1)   # Natural mirror feel
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp += 1
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_image, timestamp)

            # Draw hand skeletons with color coding
            hands_status = "No hands detected"
            if latest_result and latest_result.hand_landmarks:
                num_h = len(latest_result.hand_landmarks)
                hands_status = f"Both hands visible ({num_h})" if num_h >= 2 else "One hand only"
                for hi in range(num_h):
                    h_label = latest_result.handedness[hi][0].category_name
                    # GREEN for user's right hand, BLUE for user's left
                    color = (0, 255, 100) if h_label == "Left" else (255, 180, 0)
                    draw_hand_skeleton(frame, latest_result.hand_landmarks[hi], color)

            # HUD: sign info
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 135), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(frame, f"ISL Sign: {sign}  ({idx+1}/{len(SIGNS)})",
                        (12, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 180, 0), 2)
            cv2.putText(frame, tip,
                        (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 180), 1)
            cv2.putText(frame, hands_status,
                        (12, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 220, 255), 1)
            cv2.putText(frame, "Press SPACE when ready to collect",
                        (12, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 200, 255), 1)

            cv2.imshow("SignBridge AI — ISL Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):     # spacebar = start collecting
                space_pressed = True
                break
            if key == ord('q'):     # q = quit everything
                cap.release()
                cv2.destroyAllWindows()
                exit()

        if not space_pressed:
            continue   # camera failed — retry this sign

        # ── Collect 200 frames ───────────────────────────────
        print(f"    Collecting {SAMPLES_PER_SIGN} samples for ISL {sign}...")
        cam_fail = False
        while len(collected) < SAMPLES_PER_SIGN:
            ret, frame = cap.read()
            if not ret:
                print(f"    [WARN] Camera read failed during collection of {sign}.")
                cam_fail = True
                break
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp += 1
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_image, timestamp)

            isl_data, num_h = extract_isl_landmarks(latest_result)
            if isl_data is not None:
                collected.append(isl_data)

            # Draw hand skeletons with color coding
            hands_status = "No hands detected"
            if latest_result and latest_result.hand_landmarks:
                n_h = len(latest_result.hand_landmarks)
                hands_status = f"Both hands visible ({n_h})" if n_h >= 2 else "One hand only"
                for hi in range(n_h):
                    h_label = latest_result.handedness[hi][0].category_name
                    color = (0, 255, 100) if h_label == "Left" else (255, 180, 0)
                    draw_hand_skeleton(frame, latest_result.hand_landmarks[hi], color)

            # HUD overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 140), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(frame, f"Collecting ISL: {sign}  ({idx+1}/{len(SIGNS)})",
                        (12, 30), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 100), 2)
            cv2.putText(frame, tip,
                        (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 220, 180), 1)
            cv2.putText(frame, hands_status,
                        (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 220, 255), 1)

            # Progress bar
            bar_w = frame.shape[1] - 24
            progress = int((len(collected) / SAMPLES_PER_SIGN) * bar_w)
            cv2.rectangle(frame, (12, 85), (12 + bar_w, 108), (50, 50, 50), -1)
            cv2.rectangle(frame, (12, 85), (12 + progress, 108), (0, 220, 100), -1)
            cv2.putText(frame, f"{len(collected)}/{SAMPLES_PER_SIGN}",
                        (12, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("SignBridge AI — ISL Data Collection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):   # allow quit during collection too
                if collected:
                    np.save(save_path, np.array(collected, dtype=np.float32))
                    print(f"    [PARTIAL] Saved {len(collected)} samples for ISL {sign}")
                cap.release()
                cv2.destroyAllWindows()
                exit()

        if cam_fail:
            print(f"    [ERROR] Camera failed during {sign}. Trying to reconnect...")
            cap.release()
            import time; time.sleep(2)
            cap = open_camera()
            continue   # retry this sign

        # ── Save ─────────────────────────────────────────────
        if collected:
            np.save(save_path, np.array(collected, dtype=np.float32))
            print(f"    [OK] Saved ISL {sign}.npy — {len(collected)} samples (126-dim each)")

cap.release()
cv2.destroyAllWindows()

# ── Final summary ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ISL Data Collection Complete!")
total = 0
for s in SIGNS:
    p = os.path.join(DATA_DIR, f"{s}.npy")
    if os.path.exists(p):
        n = len(np.load(p))
        total += n
        print(f"  {s}: {n} samples")
    else:
        print(f"  {s}: MISSING")
print(f"\n  Total raw samples : {total}")
print(f"  Each sample       : 126 numbers (2 hands x 63)")
print("=" * 60)
print("\nNext step: Run train_isl_model.py to train the ISL model.")
