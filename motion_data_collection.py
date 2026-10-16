"""
motion_data_collection.py -- SignBridge AI Motion Data Collection
=================================================================
Collects motion gesture data for J and Z (ASL motion-based signs).

How it works:
  - J and Z are NOT single-frame poses — they require tracing a path.
  - Each sample is a SEQUENCE of 30 frames (~1 second at 30fps).
  - User presses SPACE to start recording, performs the gesture,
    and recording auto-stops after 30 frames.
  - Saves as dataset/J_motion.npy and dataset/Z_motion.npy
    (separate files from static sign data — no conflicts).

Controls:
  SPACE     = Start recording a motion sample
  N         = Skip to next sign
  Q         = Quit

NOTE: This script does NOT modify or interfere with the existing
      static sign data (A.npy through Y.npy).
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ── Motion sign guidance ──────────────────────────────────────
MOTION_TIPS = {
    'J': "Start with PINKY UP (like letter I), then trace a J-hook downward and curl inward",
    'Z': "Start with INDEX FINGER pointing out, then trace a zigzag Z shape in the air",
}

MOTION_DETAIL = {
    'J': [
        "1. Hold pinky up, fist closed (like 'I')",
        "2. Move hand DOWN",
        "3. Curl pinky inward (J-hook shape)",
    ],
    'Z': [
        "1. Point index finger out",
        "2. Move RIGHT (top of Z)",
        "3. Move DOWN-LEFT (diagonal)",
        "4. Move RIGHT again (bottom of Z)",
    ],
}

# ── MediaPipe setup ───────────────────────────────────────────
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

# ── Settings ──────────────────────────────────────────────────
MOTION_SIGNS = ['J', 'Z']
SEQUENCE_LENGTH = 30          # 30 frames per motion sample (~1 second)
SAMPLES_PER_SIGN = 100        # 100 motion recordings per letter
DATA_DIR = "dataset"
os.makedirs(DATA_DIR, exist_ok=True)

# Fingertip indices for trail drawing (MediaPipe landmark indices)
# Index 8 = index fingertip, Index 20 = pinky fingertip
KEY_FINGERTIP = {
    'J': 20,   # pinky tip (J uses pinky)
    'Z': 8,    # index tip (Z uses index finger)
}

# ── Helper: extract 63 numbers from landmarks ─────────────────
def extract_landmarks(result):
    if result and result.hand_landmarks:
        hand = result.hand_landmarks[0]
        return [coord for lm in hand for coord in (lm.x, lm.y, lm.z)]
    return None

def get_fingertip_pos(result, fingertip_idx, frame_shape):
    """Get pixel position of a specific fingertip for trail drawing."""
    if result and result.hand_landmarks:
        lm = result.hand_landmarks[0][fingertip_idx]
        h, w = frame_shape[:2]
        return (int(lm.x * w), int(lm.y * h))
    return None

# ── Draw helpers ──────────────────────────────────────────────
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
    xy = [(int(lm.x * w), int(lm.y * h)) for lm in pts]
    for a, b in CONNECTIONS:
        cv2.line(frame, xy[a], xy[b], color, 2)
    for x, y in xy:
        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 3, color, -1)

def draw_trail(frame, trail_points, color=(0, 200, 255)):
    """Draw the fingertip motion trail as a fading line."""
    if len(trail_points) < 2:
        return
    for i in range(1, len(trail_points)):
        if trail_points[i - 1] is None or trail_points[i] is None:
            continue
        # Fade from dim to bright as trail progresses
        alpha = i / len(trail_points)
        thickness = max(2, int(alpha * 5))
        b, g, r = color
        faded_color = (int(b * alpha), int(g * alpha), int(r * alpha))
        cv2.line(frame, trail_points[i - 1], trail_points[i], faded_color, thickness)
    # Draw current position as a bright dot
    if trail_points[-1] is not None:
        cv2.circle(frame, trail_points[-1], 8, color, -1)
        cv2.circle(frame, trail_points[-1], 10, (255, 255, 255), 2)

# ── Main ──────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
timestamp = 0

# Count how many are already done
already_done = sum(1 for s in MOTION_SIGNS
                   if os.path.exists(os.path.join(DATA_DIR, f"{s}_motion.npy")))

print(f"\n{'='*60}")
print(f"  SignBridge AI — Motion Data Collection (J & Z)")
print(f"  Already collected: {already_done}/{len(MOTION_SIGNS)} motion signs")
print(f"  Samples per sign : {SAMPLES_PER_SIGN} recordings")
print(f"  Frames per sample: {SEQUENCE_LENGTH} frames (~1 sec)")
print(f"{'='*60}\n")

with HandLandmarker.create_from_options(options) as landmarker:
    for idx, sign in enumerate(MOTION_SIGNS):
        save_path = os.path.join(DATA_DIR, f"{sign}_motion.npy")

        # Load existing samples so we can APPEND new ones
        existing_sequences = []
        if os.path.exists(save_path):
            existing_data = np.load(save_path)
            existing_sequences = [existing_data[i] for i in range(len(existing_data))]
            print(f"[LOADED] {sign}: {len(existing_sequences)} existing samples")
            if len(existing_sequences) >= SAMPLES_PER_SIGN:
                print(f"[SKIP] {sign} already has {len(existing_sequences)}/{SAMPLES_PER_SIGN} samples.")
                continue

        all_sequences = list(existing_sequences)  # start with existing data
        tip = MOTION_TIPS[sign]
        detail_lines = MOTION_DETAIL[sign]
        fingertip_idx = KEY_FINGERTIP[sign]

        remaining = SAMPLES_PER_SIGN - len(all_sequences)
        print(f"\n>>> MOTION SIGN: {sign}  ({idx+1}/{len(MOTION_SIGNS)})")
        print(f"    Tip: {tip}")
        for line in detail_lines:
            print(f"    {line}")
        print(f"    Already have: {len(existing_sequences)} samples")
        print(f"    Need {remaining} more (target: {SAMPLES_PER_SIGN})")
        print(f"    Press SPACE to start each recording...")

        while len(all_sequences) < SAMPLES_PER_SIGN:
            recording = False
            current_sequence = []       # frames captured during this recording
            trail_points = []           # fingertip positions for trail

            # ── Wait for SPACE to start recording ─────────────────
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp += 1
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                landmarker.detect_async(mp_image, timestamp)

                # Draw hand skeleton
                draw_skeleton(frame, latest_result, color=(0, 255, 200))

                # ── HUD: Ready state ──────────────────────────────
                overlay = frame.copy()
                h_frame, w_frame = frame.shape[:2]
                cv2.rectangle(overlay, (0, 0), (w_frame, 160), (15, 15, 15), -1)
                cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

                cv2.putText(frame, f"MOTION: {sign}  ({idx+1}/{len(MOTION_SIGNS)})",
                            (12, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 210, 255), 2)
                cv2.putText(frame, tip,
                            (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 180), 1)

                # Show detail steps
                for i, line in enumerate(detail_lines):
                    cv2.putText(frame, line,
                                (12, 80 + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.42, (200, 200, 160), 1)

                cv2.putText(frame, f"Recorded: {len(all_sequences)}/{SAMPLES_PER_SIGN}",
                            (12, 80 + len(detail_lines) * 20 + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

                # Bottom bar
                bot_overlay = frame.copy()
                cv2.rectangle(bot_overlay, (0, h_frame - 40), (w_frame, h_frame),
                              (20, 20, 20), -1)
                cv2.addWeighted(bot_overlay, 0.7, frame, 0.3, 0, frame)
                cv2.putText(frame, "Press SPACE to start recording | N=next sign | Q=quit",
                            (10, h_frame - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

                cv2.imshow("SignBridge AI — Motion Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    recording = True
                    current_sequence = []
                    trail_points = []
                    break
                if key == ord('n'):
                    # Skip to next sign
                    print(f"    [SKIP] Skipping {sign} (collected {len(all_sequences)} so far)")
                    break
                if key == ord('q'):
                    # Save what we have and quit
                    if all_sequences:
                        np.save(save_path, np.array(all_sequences))
                        print(f"    [PARTIAL] Saved {len(all_sequences)} samples for {sign}")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            if not recording:
                break   # user pressed N to skip this sign

            # ── Record SEQUENCE_LENGTH frames ─────────────────────
            countdown_start = time.time()
            # Brief 3-2-1 countdown
            for count in [3, 2, 1]:
                countdown_end = time.time() + 0.4
                while time.time() < countdown_end:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    timestamp += 1
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    landmarker.detect_async(mp_image, timestamp)
                    draw_skeleton(frame, latest_result, color=(0, 255, 200))

                    # Countdown overlay
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (15, 15, 15), -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                    cv2.putText(frame, f"Get ready... {count}",
                                (12, 42), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2)
                    cv2.imshow("SignBridge AI — Motion Data Collection", frame)
                    cv2.waitKey(1)

            # NOW record the actual motion
            while len(current_sequence) < SEQUENCE_LENGTH:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp += 1
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                landmarker.detect_async(mp_image, timestamp)

                landmarks = extract_landmarks(latest_result)
                if landmarks:
                    current_sequence.append(landmarks)
                    # Track fingertip for trail
                    tip_pos = get_fingertip_pos(latest_result, fingertip_idx, frame.shape)
                    trail_points.append(tip_pos)

                # Draw hand + trail
                draw_skeleton(frame, latest_result, color=(0, 255, 100))
                draw_trail(frame, trail_points, color=(0, 200, 255))

                # HUD overlay
                overlay = frame.copy()
                h_frame = frame.shape[0]
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], 105), (15, 15, 15), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                cv2.putText(frame, f"RECORDING {sign}  (sample {len(all_sequences)+1}/{SAMPLES_PER_SIGN})",
                            (12, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 80, 255), 2)
                cv2.putText(frame, "Perform the gesture NOW!",
                            (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 100), 1)

                # Progress bar
                bar_w = frame.shape[1] - 24
                progress = int((len(current_sequence) / SEQUENCE_LENGTH) * bar_w)
                cv2.rectangle(frame, (12, 68), (12 + bar_w, 88), (50, 50, 50), -1)
                cv2.rectangle(frame, (12, 68), (12 + progress, 88), (0, 80, 255), -1)
                cv2.putText(frame, f"Frame {len(current_sequence)}/{SEQUENCE_LENGTH}",
                            (12, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                cv2.imshow("SignBridge AI — Motion Data Collection", frame)
                cv2.waitKey(1)

            # Validate and save the sequence
            if len(current_sequence) == SEQUENCE_LENGTH:
                all_sequences.append(current_sequence)
                print(f"    [OK] Sample {len(all_sequences)}/{SAMPLES_PER_SIGN} recorded for {sign}")
            else:
                print(f"    [WARN] Short sequence ({len(current_sequence)} frames), discarding")

        # ── Save all sequences for this sign ──────────────────
        if all_sequences:
            np.save(save_path, np.array(all_sequences, dtype=np.float32))
            print(f"    [SAVED] {sign}_motion.npy — {len(all_sequences)} sequences × {SEQUENCE_LENGTH} frames")

cap.release()
cv2.destroyAllWindows()

# ── Final summary ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Motion Data Collection Complete!")
for s in MOTION_SIGNS:
    p = os.path.join(DATA_DIR, f"{s}_motion.npy")
    if os.path.exists(p):
        data = np.load(p)
        print(f"  {s}: {data.shape[0]} sequences × {data.shape[1]} frames × {data.shape[2]} landmarks")
    else:
        print(f"  {s}: MISSING")
print("=" * 60)
print("\nNext step: Run train_motion_model.py to train the LSTM model.")
