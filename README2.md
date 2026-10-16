# ✋ SignBridge AI — Complete User Guide

> **Real-time ASL (American Sign Language) Alphabet Detection**  
> Detects 24 static hand signs (A–Y, excluding J & Z) using your webcam,  
> with full voice feedback, word building, and spelling support.

---

## 📋 Table of Contents

1. [What SignBridge AI Can Do](#-what-signbridge-ai-can-do)
2. [Project File Structure](#-project-file-structure)
3. [Requirements & Installation](#-requirements--installation)
4. [Step-by-Step: How to Run](#-step-by-step-how-to-run)
5. [Keyboard Controls (predict.py)](#️-keyboard-controls-predictpy)
6. [Voice Features Explained](#-voice-features-explained)
7. [Supported Signs (A–Y)](#-supported-signs-ay)
8. [How the Model Was Trained](#-how-the-model-was-trained)
9. [Tips for Best Accuracy](#-tips-for-best-accuracy)
10. [Known Limitations](#-known-limitations)

---

## 🚀 What SignBridge AI Can Do

| Feature | Details |
|---|---|
| **Real-time detection** | Detects ASL hand signs live via webcam at ~30 FPS |
| **24 sign support** | Recognizes A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y |
| **Both hands** | Works with your **left OR right hand** — auto-detected and mirrored |
| **Voice feedback** | Speaks each detected letter aloud as you sign it |
| **Word builder** | Build words letter by letter on screen |
| **Speak word** | Press `T` to hear the full word spoken |
| **Spell word** | Press `R` to spell the word letter-by-letter |
| **Voice toggle** | Press `V` to turn voice ON or OFF |
| **Confidence bar** | Visual bar shows how confident the model is |
| **Hand label** | Shows "Left Hand" or "Right Hand" on screen |
| **99.95% accuracy** | Trained on 50,000+ augmented samples |

---

## 📁 Project File Structure

```
Mega Project/
│
├── predict.py            ← MAIN PROGRAM — run this to use SignBridge AI
├── data_collection.py    ← Collect training data via webcam
├── train_model.py        ← Train the neural network model
├── utils.py              ← Shared utilities (normalization + augmentation)
│
├── hand_landmarker.task  ← MediaPipe hand tracking model (pre-downloaded)
├── signbridge_model.h5   ← Trained Keras model (best weights)
├── signbridge_model.tflite ← Optimized TFLite model (used by predict.py)
├── label_encoder.npy     ← Saved class labels (A–Y)
├── training_results.png  ← Training accuracy/loss curves + confusion matrix
│
└── dataset/
    ├── A.npy             ← 100–150 raw landmark samples per letter
    ├── B.npy
    ├── ... (one file per sign)
    └── Y.npy
```

---

## 🛠 Requirements & Installation

### Python Version
Requires **Python 3.10 or higher**.

### Install All Dependencies

Open a terminal in the project folder and run:

```bash
pip install opencv-python mediapipe tensorflow numpy scikit-learn matplotlib seaborn pyttsx3
```

### Dependency Reference

| Package | Purpose |
|---|---|
| `opencv-python` | Webcam capture & drawing on screen |
| `mediapipe` | Hand landmark detection (21 keypoints) |
| `tensorflow` | Neural network training & TFLite inference |
| `numpy` | Array/math operations |
| `scikit-learn` | Label encoding, train/test split, metrics |
| `matplotlib` + `seaborn` | Training graphs & confusion matrix |
| `pyttsx3` | Offline text-to-speech (no internet needed) |

---

## 🏃 Step-by-Step: How to Run

### Option A — Just Use It (Model Already Trained)

If `signbridge_model.tflite` and `label_encoder.npy` already exist:

```bash
python predict.py
```

That's it! The camera will open and you can start signing.

---

### Option B — Full Pipeline (Collect Data → Train → Predict)

#### Step 1: Collect Training Data

```bash
python data_collection.py
```

- Webcam opens automatically
- For each sign (A → Y), the on-screen tip tells you the correct hand shape
- Press **SPACE** when your hand is ready to start collecting
- **150 samples** are captured per sign automatically
- Already-collected signs are **skipped automatically**
- J and Z are excluded (they require hand movement — Phase 2)

#### Step 2: Train the Model

```bash
python train_model.py
```

- Loads all `.npy` files from the `dataset/` folder
- Normalizes landmarks (wrist-relative, scale-invariant)
- Augments dataset ×15 (~50,000+ samples total) using:
  - Gaussian noise, scale jitter, 2D rotation
  - Perspective warp, finger dropout, **mirror hand** (both hands!)
- Trains a 5-layer deep neural network (512 → 256 → 128 → 64 → 32 → 24)
- Saves best model automatically via `ModelCheckpoint`
- Exports `signbridge_model.tflite` automatically after training
- Saves `training_results.png` with accuracy, loss, and confusion matrix

#### Step 3: Run Real-Time Detection

```bash
python predict.py
```

---

## ⌨️ Keyboard Controls (predict.py)

| Key | Action | Voice Response |
|---|---|---|
| `SPACE` | **Add** the currently detected letter to your word | Says the added letter aloud |
| `BACKSPACE` | **Delete** the last letter from the word | Says *"Deleted X"* |
| `C` | **Clear** the entire word | Says *"Cleared"* |
| `T` | **Speak** the current word as a whole word | e.g., says *"MOM"* |
| `R` | **Spell** the current word letter by letter | e.g., says *"M... O... M"* |
| `V` | **Toggle Voice** ON or OFF | Says *"Voice on"* when re-enabled |
| `Q` | **Quit** the program | Says *"Goodbye! Session ended."* |

---

## 🎙️ Voice Features Explained

SignBridge AI has a **fully offline** voice system powered by `pyttsx3` (Windows SAPI).  
The voice runs in a **background thread** — the camera never lags due to speech.

### 1. Auto Letter Announcement
- When the **same letter is detected for 12 consecutive frames** (~0.4 seconds), it is spoken aloud automatically
- If you switch to a different sign, the counter resets and the new letter is announced when it stabilises
- Old announcements are discarded if you move too fast — no backlog of stale speech

### 2. Letter Confirmation on SPACE
- Every time you press `SPACE` to save a letter, it is spoken immediately as confirmation

### 3. Speak Word (`T`)
- Reads the current word as a complete word
- Example: word = `HELLO` → voice says *"HELLO"*

### 4. Spell Word (`R`)
- Spells the word one letter at a time with pauses between
- Example: word = `HELLO` → voice says *"H... E... L... L... O"*

### 5. Voice ON/OFF (`V`)
- Silences all voice output while keeping detection active
- HUD shows **"Voice: ON"** (green) or **"Voice: OFF"** (grey)

---

## 🤟 Supported Signs (A–Y)

The following 24 static ASL signs are supported:

| Sign | Hand Shape |
|---|---|
| A | Fist with thumb beside index finger |
| B | Four fingers straight up, thumb tucked |
| C | Curved fingers forming a 'C' |
| D | Index up, other fingers curve to thumb |
| E | All fingers curled, thumb tucked under |
| F | Index + thumb circle, others spread out |
| G | Index & thumb point sideways (like a gun) |
| H | Index & middle finger point sideways together |
| I | Pinky straight up, fist closed |
| K | Index up, middle finger angled out, thumb between them |
| L | Index up, thumb out (like letter L) |
| M | Three fingers fold over tucked thumb |
| N | Two fingers fold over tucked thumb |
| O | All fingers + thumb form an 'O' |
| P | Like K but hand pointing down |
| Q | Like G but hand pointing down |
| R | Index and middle fingers crossed |
| S | Fist with thumb over fingers |
| T | Thumb between index and middle fingers (fist) |
| U | Index and middle fingers together, pointing up |
| V | Index and middle fingers in a V |
| W | Index, middle, ring fingers spread (W shape) |
| X | Index finger bent/hooked |
| Y | Pinky and thumb out, other fingers curled |

> ⚠️ **J and Z** are motion-based signs (require drawing letters in the air) and are not yet supported. They will be added in Phase 2.

---

## 🧠 How the Model Was Trained

### Architecture — SignBridge_v3

```
Input (63 values: 21 landmarks × x,y,z)
    ↓
Dense(512) → BatchNorm → ReLU → Dropout(0.40)
    ↓
Dense(256) → BatchNorm → ReLU → Dropout(0.35)
    ↓
Dense(128) → BatchNorm → ReLU → Dropout(0.30)
    ↓
Dense(64)  → BatchNorm → ReLU → Dropout(0.25)
    ↓
Dense(32)  → BatchNorm → ReLU → Dropout(0.20)
    ↓
Dense(24, softmax)  ← one output per sign
```

**Total parameters:** ~212,000

### Training Configuration

| Setting | Value |
|---|---|
| Raw samples per sign | 150 (100 for A–E) |
| Augmentation factor | ×15 |
| Total augmented samples | ~56,700 |
| Train / test split | 85% / 15% |
| Batch size | 64 |
| Max epochs | 400 |
| Early stopping patience | 30 |
| Optimizer | Adam (lr=1e-3) |
| Regularization | L2 (1e-4) on all Dense layers |

### Augmentation Pipeline (6 types)

| Type | Effect |
|---|---|
| Gaussian noise | Simulates sensor jitter |
| Scale jitter | Simulates hand closer/farther from camera |
| 2D rotation | Simulates wrist tilt (±15°) |
| Perspective warp | Simulates camera angle shift |
| Finger dropout | Simulates partial occlusion |
| **Mirror hand** | **Simulates opposite hand (left ↔ right)** |

### Final Accuracy

| Metric | Result |
|---|---|
| **Test Accuracy** | **99.95%** |
| Training accuracy | 99.77% |
| Best epoch | 114 / 400 |
| Model size (TFLite) | 224 KB |

---

## 💡 Tips for Best Accuracy

1. **Lighting** — Make sure your hand is well-lit. Avoid strong backlight.
2. **Background** — Plain or simple backgrounds work best.
3. **Distance** — Keep your hand 30–60 cm from the camera.
4. **Hold steady** — Hold each sign still for ~0.5 seconds before pressing SPACE.
5. **Both hands work** — The app auto-detects left or right hand via MediaPipe.
6. **Confusable pairs** — M/N, U/V, R/U, G/Q look similar. Hold the sign more deliberately.
7. **Speed** — If voice lags behind, it is catching up from the queue. Press `V` twice to reset.

---

## ⚠️ Known Limitations

| Limitation | Detail |
|---|---|
| **J and Z not supported** | These are motion signs requiring temporal/video analysis (Phase 2) |
| **Single hand only** | Only one hand is tracked at a time |
| **No GPU acceleration** | TensorFlow on native Windows ≥ 2.11 does not use CUDA. Use WSL2 for GPU. |
| **Static signs only** | Dynamic/moving signs are not in scope for this version |
| **English ASL only** | Trained on American Sign Language alphabet |

---

## 👥 Project Info

**Project Name:** SignBridge AI  
**Purpose:** College Mega Project — ASL Alphabet Recognition System  
**Detection:** MediaPipe Hand Landmarker → 63 normalized landmark features → Deep Neural Network  
**Voice:** pyttsx3 (offline, Windows SAPI) via non-blocking background thread  
**Model Format:** TFLite (fast, 224 KB) with Keras .h5 fallback  

---

*For training details, see `Team_Code_Explanation.txt`. For prediction pipeline details, see `predict_explanation.txt`.*
