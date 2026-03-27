import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ── MediaPipe setup ──────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "hand_landmarker.task"
latest_result = None

def save_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=save_result
)

# ── Settings ─────────────────────────────────────────────────
SIGNS = list("ABCDE")
SAMPLES_PER_SIGN = 100      # frames to capture per letter
DATA_DIR = "dataset"
os.makedirs(DATA_DIR, exist_ok=True)

# ── Helper: extract 63 numbers from landmarks ─────────────────
def extract_landmarks(result):
    if result and result.hand_landmarks:
        hand = result.hand_landmarks[0]
        return [coord for lm in hand for coord in (lm.x, lm.y, lm.z)]
    return None

# ── Main ──────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
timestamp = 0

with HandLandmarker.create_from_options(options) as landmarker:
    for sign in SIGNS:
        save_path = os.path.join(DATA_DIR, f"{sign}.npy")

        # Skip if already collected
        if os.path.exists(save_path):
            print(f"[SKIP] {sign} already collected.")
            continue

        collected = []
        print(f"\n>>> GET READY FOR SIGN: {sign}")
        print(f"    Show the sign and press SPACE to start collecting...")

        # ── Wait for spacebar ────────────────────────────────
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp += 1
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_image, timestamp)

            # Draw landmarks
            if latest_result and latest_result.hand_landmarks:
                for lm in latest_result.hand_landmarks[0]:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            cv2.putText(frame, f"Sign: {sign} | Press SPACE to collect",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.imshow("Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):   # spacebar = start collecting
                break
            if key == ord('q'):   # q = quit everything
                cap.release()
                cv2.destroyAllWindows()
                exit()

        # ── Collect 100 frames ───────────────────────────────
        print(f"    Collecting {SAMPLES_PER_SIGN} samples for {sign}...")
        while len(collected) < SAMPLES_PER_SIGN:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp += 1
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_image, timestamp)

            landmarks = extract_landmarks(latest_result)
            if landmarks:
                collected.append(landmarks)

            # Draw landmarks
            if latest_result and latest_result.hand_landmarks:
                for lm in latest_result.hand_landmarks[0]:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            # Progress bar
            progress = int((len(collected) / SAMPLES_PER_SIGN) * 400)
            cv2.rectangle(frame, (10, 60), (410, 90), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 60), (10 + progress, 90), (0, 255, 0), -1)
            cv2.putText(frame, f"Collecting {sign}: {len(collected)}/{SAMPLES_PER_SIGN}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Data Collection", frame)
            cv2.waitKey(1)

        # ── Save ─────────────────────────────────────────────
        np.save(save_path, np.array(collected))
        print(f"    ✅ Saved {sign}.npy — {len(collected)} samples")

cap.release()
cv2.destroyAllWindows()
print("\n🎉 All signs collected! Dataset saved in /dataset folder.")