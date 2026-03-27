import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# ── Load Model & Labels ───────────────────────────────────────
model = tf.keras.models.load_model("signbridge_model.h5")
classes = np.load("label_encoder.npy", allow_pickle=True)
print(f"✅ Model loaded | Signs: {classes}")

# ── MediaPipe Setup ───────────────────────────────────────────
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

# ── Helper ────────────────────────────────────────────────────
def extract_landmarks(result):
    if result and result.hand_landmarks:
        hand = result.hand_landmarks[0]
        return [coord for lm in hand for coord in (lm.x, lm.y, lm.z)]
    return None

# ── Main Loop ─────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
timestamp = 0
predicted_letter = ""
confidence = 0.0

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_image, timestamp)

        # ── Predict ───────────────────────────────────────────
        landmarks = extract_landmarks(latest_result)
        if landmarks:
            input_data = np.array([landmarks])
            prediction = model.predict(input_data, verbose=0)
            idx = np.argmax(prediction)
            predicted_letter = classes[idx]
            confidence = prediction[0][idx] * 100

        # ── Draw Landmarks ────────────────────────────────────
        if latest_result and latest_result.hand_landmarks:
            for lm in latest_result.hand_landmarks[0]:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # ── Display Prediction ────────────────────────────────
        cv2.rectangle(frame, (0, 0), (300, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Sign: {predicted_letter}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 3)
        cv2.putText(frame, f"Confidence: {confidence:.1f}%",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        cv2.imshow("SignBridge AI - Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()