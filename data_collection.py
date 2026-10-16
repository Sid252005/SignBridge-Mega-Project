import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ── Sign guidance tips for confusable letters ─────────────────
SIGN_TIPS = {
    'A': "Fist with thumb beside index finger",
    'B': "Four fingers straight up, thumb tucked",
    'C': "Curved fingers forming a 'C'",
    'D': "Index up, other fingers curve to thumb",
    'E': "All fingers curled, thumb tucked under",
    'F': "Index+thumb circle, others spread out",
    'G': "Index & thumb point sideways (like a gun)",
    'H': "Index & middle finger point sideways together",
    'I': "Pinky straight up, fist closed",
    'K': "Index up, middle finger angled out, thumb between them",
    'L': "Index up, thumb out (like letter L)",
    'M': "Three fingers fold over tucked thumb",
    'N': "Two fingers fold over tucked thumb",
    'O': "All fingers + thumb form an 'O'",
    'P': "Like K but hand pointing down",
    'Q': "Like G but hand pointing down",
    'R': "Index and middle fingers crossed",
    'S': "Fist with thumb over fingers",
    'T': "Thumb between index and middle fingers (fist)",
    'U': "Index and middle fingers together, pointing up",
    'V': "Index and middle fingers in a V",
    'W': "Index, middle, ring fingers spread (W shape)",
    'X': "Index finger bent/hooked",
    'Y': "Pinky and thumb out, other fingers curled",
}
                                                                       
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
# 24 static ASL signs — J and Z are MOTION-based, handled in Phase 2
SIGNS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
SAMPLES_PER_SIGN = 150      # 150 frames per letter for better accuracy
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

# Count how many are already done
already_done = sum(1 for s in SIGNS if os.path.exists(os.path.join(DATA_DIR, f"{s}.npy")))
print(f"\n{'='*55}")
print(f"  SignBridge AI — Data Collection (A-Y, 24 Static Signs)")
print(f"  Already collected: {already_done}/{len(SIGNS)} signs")
print(f"  Samples per sign : {SAMPLES_PER_SIGN}")
print(f"  NOTE: J and Z are SKIPPED (motion-based signs)")
print(f"{'='*55}\n")

with HandLandmarker.create_from_options(options) as landmarker:
    for idx, sign in enumerate(SIGNS):
        save_path = os.path.join(DATA_DIR, f"{sign}.npy")

        # Skip if already collected
        if os.path.exists(save_path):
            existing = np.load(save_path)
            print(f"[SKIP] {sign} already collected ({len(existing)} samples).")
            continue

        collected = []
        tip = SIGN_TIPS.get(sign, "")
        remaining = len(SIGNS) - idx
        print(f"\n>>> GET READY FOR SIGN: {sign}  ({idx+1}/{len(SIGNS)})")
        print(f"    Tip: {tip}")
        print(f"    Show the sign and press SPACE to start collecting...")

        # ── Wait for spacebar ────────────────────────────────
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp += 1
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_image, timestamp)

            # Draw hand skeleton
            if latest_result and latest_result.hand_landmarks:
                for lm in latest_result.hand_landmarks[0]:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 100), -1)

            # HUD: sign info
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 110), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(frame, f"Sign: {sign}  ({idx+1}/{len(SIGNS)})",
                        (12, 40), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 210, 255), 2)
            cv2.putText(frame, tip,
                        (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 220, 180), 1)
            cv2.putText(frame, "Press SPACE when ready to collect",
                        (12, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 200, 255), 1)

            cv2.imshow("SignBridge AI — Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):   # spacebar = start collecting
                break
            if key == ord('q'):   # q = quit everything
                cap.release()
                cv2.destroyAllWindows()
                exit()

        # ── Collect 150 frames ───────────────────────────────
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

            # Draw hand skeleton
            if latest_result and latest_result.hand_landmarks:
                for lm in latest_result.hand_landmarks[0]:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 100), -1)

            # HUD overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(frame, f"Collecting: {sign}  ({idx+1}/{len(SIGNS)})",
                        (12, 38), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 100), 2)
            cv2.putText(frame, tip,
                        (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 220, 180), 1)

            # Progress bar
            bar_w = frame.shape[1] - 24
            progress = int((len(collected) / SAMPLES_PER_SIGN) * bar_w)
            cv2.rectangle(frame, (12, 75), (12 + bar_w, 98), (50, 50, 50), -1)
            cv2.rectangle(frame, (12, 75), (12 + progress, 98), (0, 220, 100), -1)
            cv2.putText(frame, f"{len(collected)}/{SAMPLES_PER_SIGN}",
                        (12, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("SignBridge AI — Data Collection", frame)
            cv2.waitKey(1)

        # ── Save ─────────────────────────────────────────────
        np.save(save_path, np.array(collected))
        print(f"    [OK] Saved {sign}.npy — {len(collected)} samples")

cap.release()
cv2.destroyAllWindows()

# ── Final summary ─────────────────────────────────────────────
print("\n" + "="*55)
print("  Data Collection Complete!")
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
print(f"  Augmented (x15)   : ~{total * 15}")
print("="*55)