"""
words_motion_collection.py -- SignBridge AI Motion Words Data Collection
=========================================================================
Collects motion gesture data for 7 common ASL motion-based word signs.

Each word is a motion sign requiring a sequence of frames (not a single pose).
Follows the same architecture as the J/Z motion data collection system.

How it works:
  - Each sample is a SEQUENCE of 30 frames (~1 second at 30fps).
  - User presses SPACE to start collecting, performs the sign repeatedly.
  - 80 samples are recorded per word.
  - Saves as dataset/word_motions/WORD.npy

Controls:
  SPACE = Start collecting for current word
  S     = Skip current word (save what's collected so far)
  Q     = Quit anytime

NOTE: If a word's .npy file already exists, it is skipped automatically.
      This makes collection resumable if interrupted.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

from utils import normalize_landmarks, mirror_hand

# ── Sign performance tips (shown on screen) ──────────────────────────────────
SIGN_TIPS = {
    "HELLO":     "Flat hand at forehead, wave outward",
    "GOODBYE":   "Open hand, wave fingers down and up",
    "THANKYOU":  "Flat hand at chin, move forward",
    "SORRY":     "A fist, circle once on chest",
    "YES":       "S fist nod up and down twice",
    "NO":        "Index+middle snap against thumb twice",
    "AMBULANCE": "Both A fists, rotate forward like wheels"
}

# ── MediaPipe Tasks API setup ────────────────────────────────────────────────
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
    num_hands=1,
    result_callback=save_result
)

# ── Settings ─────────────────────────────────────────────────────────────────
WORD_MOTIONS = [
    "HELLO",
    "GOODBYE",
    "THANKYOU",
    "SORRY",
    "YES",
    "NO",
    "AMBULANCE"
]

DATA_DIR = "dataset/word_motions"
os.makedirs(DATA_DIR, exist_ok=True)

SAMPLES_PER_WORD  = 50
SEQUENCE_LENGTH   = 30

# ── Hand skeleton drawing ────────────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_skeleton(frame, result, color=(0, 255, 120)):
    """Draw hand landmarks and connections on frame."""
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


# ── Landmark extraction (per-frame) ─────────────────────────────────────────
def get_frame_landmarks(result):
    """
    Extract normalized, hand-mirrored landmarks from a MediaPipe result.
    Returns np.zeros(63) if no hand detected (keeps recording going —
    partial occlusion is realistic and improves model robustness).
    """
    if not result or not result.hand_landmarks:
        return np.zeros(63)
    hand = result.hand_landmarks[0]
    raw = [c for lm in hand
           for c in (lm.x, lm.y, lm.z)]
    normalized = normalize_landmarks(raw)
    # Mirror left hand so all data looks like right hand
    if (result.handedness and
            result.handedness[0][0].category_name == "Left"):
        pts = np.array(normalized).reshape(21, 3)
        normalized = mirror_hand(pts).flatten().tolist()
    return np.array(normalized)


# ── Main collection ──────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit(1)

timestamp = 0
WINDOW_NAME = "SignBridge AI — Motion Words Collection"

# Count already collected words
already_done = sum(1 for w in WORD_MOTIONS
                   if os.path.exists(os.path.join(DATA_DIR, f"{w}.npy")))

print(f"\n{'='*60}")
print(f"  SignBridge AI — Motion Words Data Collection")
print(f"  Already collected: {already_done}/{len(WORD_MOTIONS)} words")
print(f"  Samples per word : {SAMPLES_PER_WORD} recordings")
print(f"  Frames per sample: {SEQUENCE_LENGTH} frames (~1 sec)")
print(f"{'='*60}\n")

with HandLandmarker.create_from_options(options) as landmarker:
    for word_idx, word in enumerate(WORD_MOTIONS):
        save_path = os.path.join(DATA_DIR, f"{word}.npy")

        # ── Skip logic: resume if interrupted ─────────────────────────────
        if os.path.exists(save_path):
            print(f"Skipping {word} — already collected")
            continue

        samples = []  # collected sequences for this word
        tip = SIGN_TIPS[word]

        print(f"\n>>> Word {word_idx+1} of {len(WORD_MOTIONS)}: {word}")
        print(f"    Tip: {tip}")
        print(f"    Press SPACE to begin collecting, S=skip, Q=quit")

        # ══════════════════════════════════════════════════════════════════
        # PHASE 1 — READY SCREEN: show tips, wait for SPACE
        # ══════════════════════════════════════════════════════════════════
        user_started = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp += 1
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_image, timestamp)

            # Draw hand skeleton in green
            draw_skeleton(frame, latest_result, color=(0, 255, 200))

            h_frame, w_frame = frame.shape[:2]

            # Dark overlay at top
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w_frame, 130), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

            # Word title (large)
            cv2.putText(frame, f"Word {word_idx+1} of {len(WORD_MOTIONS)}: {word}",
                        (12, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 210, 255), 2)

            # Sign tip (yellow)
            cv2.putText(frame, tip,
                        (12, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1)

            # Instruction
            cv2.putText(frame, "Practice sign, then press SPACE to begin",
                        (12, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

            # Bottom hint bar
            bot_overlay = frame.copy()
            cv2.rectangle(bot_overlay, (0, h_frame - 35), (w_frame, h_frame),
                          (20, 20, 20), -1)
            cv2.addWeighted(bot_overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, "SPACE=start | S=skip this word | Q=quit",
                        (10, h_frame - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                user_started = True
                break
            elif key in (ord('s'), ord('S')):
                # Skip this word
                print(f"    [SKIP] Skipping {word}")
                break
            elif key in (ord('q'), ord('Q')):
                # Save partial and quit
                if samples:
                    np.save(save_path, np.array(samples, dtype=np.float32))
                    print(f"    [PARTIAL] Saved {len(samples)} samples for {word}")
                cap.release()
                cv2.destroyAllWindows()
                print("\nCollection stopped by user.")
                exit()

        if not user_started:
            continue  # user pressed S to skip

        # ══════════════════════════════════════════════════════════════════
        # PHASE 2 — COUNTDOWN (3, 2, 1, GO!)
        # ══════════════════════════════════════════════════════════════════
        countdown_start = time.time()
        while True:
            elapsed = time.time() - countdown_start
            if elapsed >= 3.0:
                break  # countdown finished

            current_count = 3 - int(elapsed)

            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp += 1
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_image, timestamp)

            draw_skeleton(frame, latest_result, color=(0, 255, 200))

            h_frame, w_frame = frame.shape[:2]

            # Large countdown number (center screen)
            text = str(current_count)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 4.0, 5)[0]
            text_x = (w_frame - text_size[0]) // 2
            text_y = (h_frame + text_size[1]) // 2
            cv2.putText(frame, text,
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 4.0,
                        (255, 255, 255), 5)

            # "Get ready..." below
            cv2.putText(frame, "Get ready...",
                        (w_frame // 2 - 80, text_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 255), 2)

            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(1)

        # Brief "GO!" flash
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            h_frame, w_frame = frame.shape[:2]
            text_size = cv2.getTextSize("GO!", cv2.FONT_HERSHEY_DUPLEX, 3.0, 5)[0]
            text_x = (w_frame - text_size[0]) // 2
            text_y = (h_frame + text_size[1]) // 2
            cv2.putText(frame, "GO!",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 3.0,
                        (0, 255, 0), 5)
            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(300)

        # ══════════════════════════════════════════════════════════════════
        # PHASE 3 — RECORDING LOOP (80 samples × 30 frames each)
        # ══════════════════════════════════════════════════════════════════
        skip_word = False
        while len(samples) < SAMPLES_PER_WORD:
            sequence = []  # one 30-frame sequence

            # Record exactly 30 frames
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp += 1
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                landmarker.detect_async(mp_image, timestamp)

                # Extract landmarks for this frame
                lm = get_frame_landmarks(latest_result)
                sequence.append(lm)

                h_frame, w_frame = frame.shape[:2]

                # (a) RED border rectangle around full frame
                cv2.rectangle(frame, (0, 0),
                              (w_frame - 1, h_frame - 1), (0, 0, 255), 4)

                # Draw hand skeleton
                draw_skeleton(frame, latest_result, color=(0, 255, 100))

                # HUD overlay at top
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w_frame, 60), (15, 15, 15), -1)
                cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

                # (b) Recording status text
                cv2.putText(frame,
                            f"Recording: {frame_num+1}/{SEQUENCE_LENGTH} | "
                            f"Sample {len(samples)+1}/{SAMPLES_PER_WORD}",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

                # Word label
                cv2.putText(frame, f"Word {word_idx+1}/7: {word}",
                            (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 210, 255), 1)

                cv2.imshow(WINDOW_NAME, frame)
                cv2.waitKey(1)

            # (c) Append completed sequence
            if len(sequence) == SEQUENCE_LENGTH:
                samples.append(sequence)
                print(f"    ✓ Sample {len(samples)}/{SAMPLES_PER_WORD} captured")

                # Show brief confirmation
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    h_frame, w_frame = frame.shape[:2]
                    cv2.putText(frame,
                                f"Sample {len(samples)} captured",
                                (w_frame // 2 - 140, h_frame // 2),
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 120), 2)
                    cv2.imshow(WINDOW_NAME, frame)
                    cv2.waitKey(1)

            # (d) Small pause between samples so user can reposition
            time.sleep(0.3)

            # Check for S (skip) or Q (quit) between samples
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('s'), ord('S')):
                print(f"    [SKIP] Saving {len(samples)} samples for {word}")
                skip_word = True
                break
            elif key in (ord('q'), ord('Q')):
                # Save what we have and quit
                if samples:
                    np.save(save_path, np.array(samples, dtype=np.float32))
                    print(f"    [PARTIAL] Saved {len(samples)} samples for {word}")
                cap.release()
                cv2.destroyAllWindows()
                print("\nCollection stopped by user.")
                exit()

        # ══════════════════════════════════════════════════════════════════
        # PHASE 4 — SAVE completed word
        # ══════════════════════════════════════════════════════════════════
        if samples:
            data = np.array(samples, dtype=np.float32)
            np.save(save_path, data)
            print(f"    Saved {word}.npy — {len(samples)} sequences  "
                  f"(shape: {data.shape})")

cap.release()
cv2.destroyAllWindows()

# ── Final summary ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Motion Words Data Collection Complete!")
for w in WORD_MOTIONS:
    p = os.path.join(DATA_DIR, f"{w}.npy")
    if os.path.exists(p):
        data = np.load(p)
        print(f"  {w}: {data.shape[0]} sequences × "
              f"{data.shape[1]} frames × {data.shape[2]} landmarks")
    else:
        print(f"  {w}: MISSING")
print(f"{'='*60}")
print("\nAll motion words collected!")
print("Run python train_words_motion_model.py next")
