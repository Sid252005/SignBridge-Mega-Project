# SignBridge AI

> **Real-Time Bilingual Sign Language Translation, Word Building & Speech Synthesis System**  
> An intelligent computer vision and deep learning assistive platform bridging the communication gap between the Deaf and Hard-of-Hearing (DHH) community and the hearing world.

SignBridge AI runs a modern client-server architecture consisting of a **FastAPI WebSocket backend** performing real-time MediaPipe Hand Landmarker detection and multi-model neural network inference, coupled with a responsive **React (Vite) frontend** featuring a dark-themed camera workspace, live confidence telemetry, word formulation, predictive suggestions, text-to-speech synthesis, and multi-tier emergency (SOS) detection.

---

## 1. Project Overview

### The Problem SignBridge AI Solves
Over 70 million individuals worldwide use sign language as their primary means of communication. However, the vast majority of hearing individuals lack sign language comprehension. This disparity creates severe communication barriers in healthcare, banking, education, and daily personal interactions. 

Traditional assistive solutions suffer from severe drawbacks:
* **Costly and intrusive sensor gloves** that are fragile, tethered, and impractical for day-to-day use.
* **Cloud-dependent video translation APIs** that introduce high network latency (200–800 ms), require high-bandwidth connections, and expose intimate webcam video streams to remote cloud servers.
* **Static-only classifiers** that ignore dynamic gestures (such as 'J', 'Z', and motion-based words like "HELLO" or "THANKYOU").
* **Screen-only text output** that fails to provide voice feedback, hindering natural two-way conversation.

### Who It Is Designed For
* **Primary Users**: Deaf, hard-of-hearing, and speech-impaired individuals seeking a sensorless, non-intrusive translation bridge.
* **Secondary Users**: Hearing individuals, educators, service workers, and families seeking immediate auditory translation of gestures.
* **Healthcare & Emergency Responders**: Personnel requiring fast recognition of critical distress phrases (e.g., "HELP", "DOCTOR", "PAIN") and automated emergency signals.

### What the Application Does
SignBridge AI captures live video from any standard USB or integrated laptop webcam. Each frame is processed locally using **Google MediaPipe Tasks API (LIVE_STREAM mode)** to extract 21 three-dimensional hand landmarks (63 coordinates per hand). These coordinates undergo wrist-relative translation and palm-scale normalization, followed by automated bilateral hand mirroring. The normalized feature vectors are fed into specialized neural network models:
* **ASL Alphabet Mode**: Classifies 24 static American Sign Language letters (A–Y, excluding J and Z) using a lightweight Feed-Forward Neural Network (TFLite).
* **ISL Alphabet Mode**: Classifies 26 Indian Sign Language dual-hand letters (A–Z) using a dual-hand 126-dimensional classifier (TFLite).
* **Common Words Static Mode**: Classifies 23 critical static emergency and conversational signs (HELP, STOP, WATER, DOCTOR, SICK, etc.) from dual-hand postures.
* **Motion Letters (J / Z)**: Captures 30-frame temporal sequences (~1.2s) classified via a Long Short-Term Memory (LSTM) recurrent network.
* **Motion Common Words**: Captures 30-frame temporal sequences classified into 7 dynamic conversational signs (HELLO, GOODBYE, THANKYOU, SORRY, YES, NO, AMBULANCE) via a 2-layer LSTM recurrent network.
* **Speech Synthesis (TTS)**: Utters stabilized letters, completed words, and motion signs aloud via the browser Web Speech API.
* **Automated Emergency Engine**: Continuously monitors for a repeated 'S' sign (performed 3 distinct times), rapid lateral distress waving, or sustained emergency signs to trigger high-priority SOS alerts.

---

## 2. Current Features

| Feature | Status | Description |
|---|:---:|---|
| **ASL Static Alphabet (A–Y)** | **Implemented** | Recognizes 24 static ASL letter poses via TFLite inference with bilateral left/right hand support. |
| **ISL Dual-Hand Alphabet (A–Z)** | **Implemented** | Recognizes all 26 Indian Sign Language letters using simultaneous two-hand 126-dim coordinate vectors. |
| **Common Words (23 Static Signs)** | **Implemented** | Classifies 23 high-utility static words (HELP, STOP, WATER, DOCTOR, SICK, PAIN, etc.) at 0.75 confidence threshold. |
| **Motion Letters (J & Z)** | **Implemented** | Key-triggered 30-frame LSTM sequence capture with visual progress and trail drawing. |
| **Motion Words (7 Dynamic Signs)** | **Implemented** | Recognizes HELLO, GOODBYE, THANKYOU, SORRY, YES, NO, and AMBULANCE using 2-layer LSTM with sequence interpolation. |
| **Dual-Hand Motion Evaluation** | **Implemented** | Automatically tests direct and X-mirrored orientations for motion sequences, enabling both left- and right-handed signers. |
| **Temporal Sequence Interpolation** | **Implemented** | Resamples variable sequence captures (15–45 frames) to exactly 30 frames using `np.interp` to ensure invariant classification. |
| **Webcam Capture via Browser** | **Implemented** | In-browser canvas streaming via `react-webcam` (10 FPS idle, 25 FPS during motion capture). |
| **MediaPipe Tasks API Integration** | **Implemented** | Asynchronous `HandLandmarker` running in `LIVE_STREAM` mode with sub-millisecond callback ingestion. |
| **FastAPI Backend Architecture** | **Implemented** | Asynchronous HTTP and WebSocket server handling image decoding, model routing, and state persistence. |
| **Full-Duplex WebSocket Protocol** | **Implemented** | Bi-directional `/ws` channel streaming JPEG frames from client and returning predictions, state, and landmarks. |
| **Weighted Temporal Smoothing** | **Implemented** | Rolling window (15 frames) with weighted temporal voting and confidence averaging to eliminate visual flicker. |
| **Detection Display Locking** | **Implemented** | Locks hero detection display during motion gesture recording and result display, suppressing static prediction overwrites. |
| **Word Builder Workspace** | **Implemented** | Workspace for assembling letters into phrases with Add, Backspace, Clear, and Set operations. |
| **Predictive Word Suggestions** | **Implemented** | Up to 5 interactive prediction pills matching prefix queries from a 100+ word dictionary in letter mode. |
| **Browser Speech Synthesis (TTS)** | **Implemented** | Non-blocking voice readback for completed words and motion gestures with Chromium queue-clearing fixes. |
| **Conversation History Log** | **Implemented** | Scrollable session log with timestamps, individual entry deletion, and text file export. |
| **Save / Export Conversation** | **Implemented** | Exports session history to a server-side timestamped text file (`conversation_YYYYMMDD_HHMMSS.txt`). |
| **Manual SOS Trigger** | **Implemented** | Dedicated emergency button triggering audible alarms, HUD warning accents, and logging distress alerts. |
| **Automatic SOS (S-Sign x3)** | **Implemented** | Multi-state detector tracking 3 distinct repetitions of ASL letter 'S' within a 10-second window. |
| **Automatic SOS (Distress Wave)** | **Implemented** | Optical wrist-tracking engine detecting rapid horizontal distress waving (>= 4 reversals with high amplitude). |
| **Automatic SOS (Sustained HELP)** | **Implemented** | Triggers automated distress alert if static HELP sign is held for >= 20 frames at >= 85% confidence. |
| **Keyboard Shortcuts** | **Implemented** | Comprehensive keyboard bindings: Space (Add), Backspace (Delete), Esc (Clear), S (Speak), V (Voice), T (Lang), P (Mode), N (Motion). |
| **WebSocket Auto-Reconnect** | **Implemented** | Client hook featuring exponential backoff reconnection logic (3s, 6s, 12s, max 15s). |
| **Responsive Dark UI** | **Implemented** | Modern design system built with Vanilla CSS variables, glassmorphism, HUD overlays, and confidence tier coloring. |

---

## 3. System Architecture

SignBridge AI separates high-throughput video ingestion from low-latency computer vision and machine learning inference through a full-duplex WebSocket bridge:

```mermaid
flowchart TD
    subgraph Client ["Client Browser (React 19 + Vite)"]
        Webcam["Webcam Feed (react-webcam)"]
        FrameSender["Frame Dispatcher (10 FPS idle / 25 FPS motion)"]
        UI["UI State & Telemetry (useSignBridge)"]
        TTS["Web Speech API (TTS Engine)"]
        Webcam --> FrameSender
        UI --> TTS
    end

    subgraph Network ["Transport Layer"]
        WS["WebSocket Channel (/ws)"]
        FrameSender -. "JSON {type: 'frame', data: base64}" .-> WS
        WS -. "JSON {type: 'prediction' / 'state'}" .-> UI
    end

    subgraph Server ["Server (FastAPI + Uvicorn)"]
        Decoder["Image Decoder (cv2.imdecode)"]
        MP["MediaPipe HandLandmarker (LIVE_STREAM)"]
        Norm["Landmark Preprocessing (utils.py)"]
        Router{"Mode & Action Router"}

        WS --> Decoder
        Decoder --> MP
        MP --> Norm
        Norm --> Router

        subgraph Models ["Inference Engine"]
            ASL["ASL Static Model (TFLite)"]
            ISL["ISL Static Model (TFLite)"]
            Words["Words Static Model (TFLite)"]
            MotionL["Motion J/Z Model (Keras .h5)"]
            MotionW["Motion Words Model (Keras .h5)"]
        end

        Router -->|Mode: Letters, Lang: ASL| ASL
        Router -->|Mode: Letters, Lang: ISL| ISL
        Router -->|Mode: Words, Single Frame| Words
        Router -->|Action: motion_sequence (Letters)| MotionL
        Router -->|Action: motion_sequence (Words)| MotionW

        ASL --> Smooth["Temporal Smoothing & State Update"]
        ISL --> Smooth
        Words --> Smooth
        MotionL --> Smooth
        MotionW --> Smooth
        Smooth --> WS
    end
```

---

## 4. Technology Stack

| Layer | Technology | Purpose | Version Verified |
|---|---|---|:---:|
| **Frontend Framework** | React | Component-driven user interface architecture | `^19.2.8` |
| **Frontend DOM** | ReactDOM | React DOM rendering engine | `^19.2.8` |
| **Frontend Build Tool** | Vite | Rapid ESM development server and production bundler | `^8.3.0` |
| **Vite React Plugin** | `@vitejs/plugin-react` | Babel/SWC Fast Refresh support | `^6.1.1` |
| **Webcam Ingestion** | `react-webcam` | Canvas capture and camera access abstraction | `^7.2.0` |
| **Linter** | `oxlint` | High-performance JavaScript/JSX static analysis | `^1.81.0` |
| **Frontend Styling** | Vanilla CSS (CSS3) | Design tokens, glassmorphism, responsive grid, animations | Native |
| **Speech Engine** | Web Speech API | Client-side, zero-latency text-to-speech engine | Browser Native |
| **Backend Framework** | FastAPI | High-performance asynchronous web framework | `0.136.1` |
| **ASGI Server** | Uvicorn | High-throughput ASGI server implementation | `0.46.0` |
| **Deep Learning** | TensorFlow | Neural network evaluation, Keras runtime, TFLite interpreter | `2.21.0` |
| **Computer Vision** | OpenCV (`opencv-python`) | Image decoding, color transforms, mirror operations | `4.13.0` |
| **Hand Tracking** | Google MediaPipe | Hand Landmarker Tasks API (`LIVE_STREAM` mode) | `0.10.32` |
| **Mathematical Compute** | NumPy | Coordinate vector manipulation, scaling, interpolation | `2.4.3` |
| **Runtime Environment** | Python | Primary backend programming environment | `3.12.10` |
| **Runtime Environment** | Node.js | Frontend runtime and package management | `v24.18.0` (npm `11.16.0`) |

---

## 5. Requirements

### Hardware Recommendations
* **Processor**: Intel Core i3 / AMD Ryzen 3 or higher (AVX2 support recommended for CPU TensorFlow optimization).
* **Memory (RAM)**: 8 GB minimum (16 GB recommended for running concurrent model workers).
* **Camera**: Standard USB 2.0 or integrated 720p/1080p webcam capable of >= 30 FPS.
* **GPU**: Optional. The entire pipeline is optimized for real-time CPU inference via TFLite XNNPACK delegates and lightweight LSTM layers.

### Software Requirements
* **Operating System**: Windows 10/11, macOS 12+, or Ubuntu 20.04+ LTS.
* **Python**: Python `3.10` to `3.12` (Python 3.12.10 verified in current workspace).
* **Node.js**: Node.js `18.x`, `20.x`, or `22.x+` (Node v24.18.0 verified in current workspace).
* **Package Manager**: `npm` (v10+ or v11+).
* **Web Browser**: Google Chrome, Microsoft Edge, Brave, or Safari (Chromium-based browsers recommended for optimal Web Speech API and `navigator.mediaDevices` camera support).

---

## 6. Complete Project Structure

```
Mega Project/
├── backend.py                        # FastAPI backend, WebSocket /ws, inference pipelines, SOS logic
├── utils.py                          # Landmark normalization, hand mirroring, synthetic augmentation
├── hand_landmarker.task              # Pre-downloaded MediaPipe HandLandmarker Tasks API model asset
│
├── signbridge_model.tflite           # ASL static alphabet model (TFLite, 24 classes, A-Y)
├── signbridge_model.h5               # ASL static alphabet model (Keras fallback)
├── label_encoder.npy                 # ASL class label array (24 classes)
│
├── isl_model.tflite                  # ISL static alphabet model (TFLite, 26 classes, A-Z)
├── isl_model.h5                      # ISL static alphabet model (Keras fallback)
├── isl_label_encoder.npy             # ISL class label array (26 classes)
│
├── words_model.tflite                # Common words static model (TFLite, 23 classes)
├── words_model.h5                    # Common words static model (Keras fallback)
├── words_label_encoder.npy           # Common words class label array (23 classes)
│
├── motion_model.h5                   # Motion letters LSTM model (Keras, J and Z)
├── motion_label_encoder.npy          # Motion letters class label array (['J', 'Z'])
│
├── words_motion_model.h5             # Motion words LSTM model (Keras, 7 classes)
├── words_motion_label_encoder.npy    # Motion words class label array (7 classes)
│
├── dataset/                          # Training datasets (.npy arrays)
│   ├── ISL/                          # ISL two-hand landmark numpy datasets
│   ├── words/                        # Static common words numpy datasets
│   ├── word_motions/                 # Dynamic common words 30-frame sequence datasets
│   └── *.npy                         # ASL static letters and J/Z motion datasets
│
├── data_collection.py                # Standalone ASL data collection script
├── isl_data_collection.py            # Standalone ISL two-hand data collection script
├── words_data_collection.py          # Standalone common words data collection script
├── motion_data_collection.py         # Standalone J/Z motion data collection script
├── words_motion_collection.py        # Standalone motion words data collection script
│
├── train_model.py                    # Training script for ASL static model
├── train_isl_model.py                # Training script for ISL dual-hand model
├── train_words_model.py              # Training script for static common words model
├── train_motion_model.py             # Training script for motion letters LSTM model
├── train_words_motion_model.py       # Training script for motion words LSTM model
│
├── predict.py                        # Standalone OpenCV desktop GUI application (v7)
├── test_hand.py                      # Diagnostic sanity test for MediaPipe hand tracking
│
├── README.md                         # Developer & user onboarding documentation
├── SIGNBRIDGE_PROJECT_REPORT.md      # Comprehensive technical report & specification
│
└── signbridge-ui/                    # React 19 Frontend Web Application
    ├── index.html                    # HTML entry point, fonts, title, skip link
    ├── package.json                  # Frontend dependencies and build scripts
    ├── vite.config.js                # Vite configuration with backend reverse proxy
    └── src/
        ├── main.jsx                  # Application bootstrap entry point
        ├── App.jsx                   # Primary layout coordinator & keyboard handler
        ├── App.css                   # Component stylesheets, animations, HUD layout
        ├── index.css                 # Design tokens, CSS variables, typography resets
        ├── hooks/
        │   └── useSignBridge.js      # WebSocket lifecycle, speech, and motion state hook
        └── components/
            ├── Header.jsx            # Top navigation bar, connection and language status
            ├── CameraPanel.jsx       # Webcam video preview, HUD indicators, countdown
            ├── PredictionPanel.jsx   # Hero detection panel displaying detected letter/word
            ├── WordBuilder.jsx       # Phrase composition row and character control buttons
            ├── Suggestions.jsx       # Predictive word suggestion pills
            ├── ConversationLog.jsx   # Scrollable history session log with deletion
            ├── ControlBar.jsx        # Bottom action toolbar (Speech, Mode, Motion, SOS)
            └── SOSAlert.jsx          # Full-screen emergency modal dialog
```

---

## 7. AI Models

SignBridge AI employs five distinct deep learning models trained specifically on MediaPipe landmark coordinate geometry:

| Model Name | Physical File | Format | Classes | Label Encoder | Input Shape | Primary Inference | Fallback |
|---|---|:---:|:---:|---|:---:|:---:|:---:|
| **ASL Static Alphabet** | `signbridge_model.tflite` | TFLite | 24 (A–I, K–Y) | `label_encoder.npy` | `(1, 63)` | `tf.lite.Interpreter` | `signbridge_model.h5` |
| **ISL Static Alphabet** | `isl_model.tflite` | TFLite | 26 (A–Z) | `isl_label_encoder.npy` | `(1, 126)` | `tf.lite.Interpreter` | `isl_model.h5` |
| **Common Words Static** | `words_model.tflite` | TFLite | 23 | `words_label_encoder.npy` | `(1, 126)` | `tf.lite.Interpreter` | `words_model.h5` |
| **Motion Letters (J/Z)** | `motion_model.h5` | Keras .h5 | 2 (J, Z) | `motion_label_encoder.npy` | `(1, 30, 63)` | `keras.models.load_model` | None (No .tflite) |
| **Motion Common Words** | `words_motion_model.h5` | Keras .h5 | 7 | `words_motion_label_encoder.npy` | `(1, 30, 63)` | `keras.models.load_model` | None (No .tflite) |

> **Format Notes**:
> 1. **ASL, ISL, and Static Words** are deployed in quantized **TensorFlow Lite (.tflite)** format, achieving sub-10ms CPU inference with automated fallback to Keras `.h5` if TFLite is unavailable.
> 2. **Motion Models (J/Z and Motion Words)** execute directly via **Keras (.h5)** LSTM inference because `.tflite` exports do not exist in the repository.

---

## 8. AI / Inference Pipeline

### Real-Time Static Sign Pipeline (Frame-by-Frame)
1. **Client Capture**: `react-webcam` captures a JPEG snapshot of the live video feed every 100ms (10 FPS).
2. **WebSocket Serialization**: Snapshot is converted to Base64 and sent over `/ws` as `{ type: "frame", data: "<base64>" }`.
3. **Image Decoding**: `backend.py` decodes the payload using `cv2.imdecode` into an OpenCV BGR numpy array.
4. **Color Conversion**: Converted to RGB format (`cv2.cvtColor`).
5. **MediaPipe Tracking**: `landmarker.detect_async` detects 21 3D hand landmarks asynchronously.
6. **Normalization**:
   * *Single-Hand (ASL)*: Coords are translated relative to wrist landmark 0 (`pts -= pts[0]`) and divided by Euclidean palm length (wrist to MCP 9).
   * *Dual-Hand (ISL & Words)*: 126-dim vector combining right hand and left hand. Missing hands are padded with zeros.
7. **Bilateral Mirroring**: If MediaPipe marks `category_name == "Left"`, X-coordinates are negated (`pts[:, 0] = -pts[:, 0]`), allowing left-handed users to use right-hand trained models.
8. **Inference**: Evaluates through TFLite interpreter (or Keras fallback).
9. **Temporal Smoothing**: Rolling deque (window = 15) calculates weighted majority voting (`1.0 + i * 0.15`) and weighted confidence averaging.
10. **State Dispatch**: Predicted letter, confidence, active word, and normalized landmarks are returned to the client as JSON.

### Motion Sequence Pipeline (Dynamic Gestures)
1. **Motion Trigger**: User clicks "Record Motion Sign" or presses `'N'`.
2. **Countdown**: 3-second preparation countdown (3 → 2 → 1 → GO).
3. **High-Speed Capture**: Frontend camera interval boosts from 100ms (10 FPS) to 40ms (~25 FPS).
4. **Client Buffering**: Valid 63-dim landmark frames are pushed into `motionBufferRef`.
5. **Dispatch**: Once 30 frames are collected (or upon 1.8s timeout with >= 15 frames), `{ type: "action", action: "motion_sequence", sequence: [...] }` is sent to the backend.
6. **Temporal Interpolation**: Backend accepts any sequence of length 15–45 frames and resamples it to exactly 30 frames using `np.interp` across all 63 dimensions.
7. **Dual-Hand Robustness**: Evaluates natural orientation; if confidence < 0.65, evaluates X-negated mirrored sequence.
8. **Prediction Display Lock**: Prediction panel locks to the detected motion word for 3.2 seconds, preventing subsequent static frames from flickering.
9. **Speech**: The recognized motion word is announced aloud via TTS.

---

## 9. MediaPipe Implementation

* **API Type**: Google MediaPipe **Tasks API** (`mediapipe.tasks.python.vision.HandLandmarker`).
* **Model Asset**: `hand_landmarker.task` (7.8 MB binary model located in project root).
* **Running Mode**: `VisionRunningMode.LIVE_STREAM` (asynchronous non-blocking streaming mode).
* **Concurrency Model**: MediaPipe processes frames on a background thread and returns landmark arrays via `_save_result_callback(result, output_image, timestamp_ms)`.
* **Hand Capacity**: Configured for `num_hands=2` to accommodate ISL and two-handed static signs.
* **Confidence Thresholds**: Detection: `0.5`, Presence: `0.5`, Tracking: `0.5`.

> [!IMPORTANT]
> The legacy `mediapipe.solutions.hands` API is NOT used in `backend.py`. The modern Tasks API is used exclusively.

---

## 10. Backend API Documentation

### REST HTTP Endpoints

#### `GET /health`
Returns system status, active modes, and model availability.
* **Request**: `GET http://localhost:8000/health`
* **Response (200 OK)**:
```json
{
  "status": "ok",
  "models": {
    "asl": true,
    "isl": true,
    "words": true,
    "motion": true,
    "words_motion": true
  },
  "current_language": "ASL",
  "current_mode": "letters",
  "active_word": ""
}
```

#### `GET /log`
Retrieves the session conversation history log.
* **Request**: `GET http://localhost:8000/log`
* **Response (200 OK)**:
```json
[
  { "time": "14:20", "text": "HELLO" },
  { "time": "14:21", "text": "THANKYOU" }
]
```

#### `DELETE /log`
Clears the session conversation log.
* **Request**: `DELETE http://localhost:8000/log`
* **Response (200 OK)**:
```json
{
  "status": "ok",
  "message": "Conversation log cleared"
}
```

---

### WebSocket Endpoint (`/ws`)

* **URL**: `ws://localhost:8000/ws`
* **Protocol**: JSON-encoded text messages.

#### Client Messages Sent to Backend

1. **Webcam Frame Streaming**:
```json
{
  "type": "frame",
  "data": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

2. **UI Actions**:
```json
{ "type": "action", "action": "add_letter", "letter": "A" }
{ "type": "action", "action": "backspace" }
{ "type": "action", "action": "clear" }
{ "type": "action", "action": "set_word", "word": "HELLO" }
{ "type": "action", "action": "speak" }
{ "type": "action", "action": "switch_language" }
{ "type": "action", "action": "switch_mode" }
{ "type": "action", "action": "save_log" }
{ "type": "action", "action": "sos" }
{ "type": "action", "action": "delete_log_entry", "index": 0 }
{ "type": "action", "action": "motion_sequence", "sequence": [[...], [...]] }
```

#### Server Messages Sent to Client

1. **Prediction Message**:
```json
{
  "type": "prediction",
  "letter": "A",
  "confidence": 98.4,
  "word": "HELL",
  "mode": "letters",
  "language": "ASL",
  "hand_detected": true,
  "landmarks": [0.0, 0.0, 0.0, ...],
  "sos_progress": 0,
  "motion_result": "HELLO",
  "motion_status": "success"
}
```

2. **State Snapshot Message**:
```json
{
  "type": "state",
  "current_word": "HELLO",
  "conversation_log": [{ "time": "14:20", "text": "HELLO" }],
  "current_language": "ASL",
  "current_mode": "letters",
  "sos": false,
  "sos_progress": 0
}
```

---

## 11. Frontend Architecture

| Component / Hook | File Path | Responsibility |
|---|---|---|
| **`App.jsx`** | `src/App.jsx` | Top-level state coordinator, keyboard shortcut listener, layout organizer. |
| **`useSignBridge.js`** | `src/hooks/useSignBridge.js` | Primary custom hook: manages WebSocket lifecycle, frame throttling, motion state machine, watchdog timers, speech synthesis, and SOS state. |
| **`Header.jsx`** | `src/components/Header.jsx` | Fixed header displaying branding, connection indicator, language badge, and mode badge. |
| **`CameraPanel.jsx`** | `src/components/CameraPanel.jsx` | Live webcam renderer (`react-webcam`), dynamic capture interval timer, SOS indicator, countdown overlay, and telemetry bar. |
| **`PredictionPanel.jsx`** | `src/components/PredictionPanel.jsx` | Hero detection panel displaying detected letter/word, scale-pop animation, and confidence progress bar. |
| **`WordBuilder.jsx`** | `src/components/WordBuilder.jsx` | Interactive word composer displaying current word with Add, Backspace, and Clear buttons. |
| **`Suggestions.jsx`** | `src/components/Suggestions.jsx` | Predictive dictionary pills displaying up to 5 matching words based on current prefix. |
| **`ConversationLog.jsx`** | `src/components/ConversationLog.jsx` | Session history panel rendering logged phrases with timestamps and individual deletion. |
| **`ControlBar.jsx`** | `src/components/ControlBar.jsx` | Bottom toolbar containing Speak, Speech ON/OFF, Save, Language toggle, Mode toggle, Motion trigger, and SOS button. |
| **`SOSAlert.jsx`** | `src/components/SOSAlert.jsx` | High-priority modal overlay displaying emergency alert banner, dismiss button, and audio siren. |

---

## 12. WebSocket Data Flow

### Frame Ingestion & Prediction Loop
```
React (CameraPanel)           WebSocket (/ws)              FastAPI (backend.py)
        |                            |                              |
        |--- {type: "frame", ...} -->|----------------------------->|  Decode JPEG
        |                            |                              |  MediaPipe Task
        |                            |                              |  Normalize + Mirror
        |                            |                              |  Model Forward Pass
        |                            |                              |  Weighted Smoothing
        |<-- {type: "prediction"} ---|<-----------------------------|  Send Result
        |                            |                              |
```

### Motion Sequence Capture & Display Locking
```
React (useSignBridge)         WebSocket (/ws)              FastAPI (backend.py)
        |                            |                              |
   [3-2-1 Countdown]                 |                              |
   [Record 30 frames]                |                              |
        |--- {action: "motion"} ---->|----------------------------->|  Resample 30 Frames
        |                            |                              |  LSTM Inference
        |                            |                              |  Clear Buffers
        |<-- {motion_status: "ok"} --|<-----------------------------|  Return Detected Word
        |                            |                              |
 [Lock Prediction Display]           |                              |
 [Speak Word via TTS]                |                              |
 [Show ✓ Result Badge]               |                              |
 [Auto-recover after 3.2s]           |                              |
```

---

## 13. Installation Guide

Follow these steps to install and run SignBridge AI on a fresh computer.

### Step 1 — Install Python
* Install **Python 3.10, 3.11, or 3.12** (Python 3.12.x 64-bit recommended).
* **Crucial for Windows**: Ensure **"Add Python to PATH"** is checked during installation.
* Verify in your terminal:
  ```bash
  python --version
  ```

### Step 2 — Install Node.js
* Install **Node.js LTS (v18, v20, or v22+)** from [nodejs.org](https://nodejs.org/).
* Verify in your terminal:
  ```bash
  node -v
  npm -v
  ```

### Step 3 — Open the Project Root
Open your terminal and navigate to the project directory:
```bash
cd "d:\Mega Project"
```

### Step 4 — Create Python Virtual Environment
Create a clean, isolated virtual environment:
```bash
# Windows
python -m venv signbridge_env
signbridge_env\Scripts\activate

# Linux / macOS
python3 -m venv signbridge_env
source signbridge_env/bin/activate
```

### Step 5 — Install Python Dependencies
Install all required packages for the backend and ML pipeline:
```bash
pip install fastapi uvicorn[standard] websockets opencv-python mediapipe tensorflow numpy scikit-learn matplotlib seaborn
```

### Step 6 — Install Frontend Dependencies
Navigate into the React UI folder and install packages:
```bash
cd signbridge-ui
npm install
cd ..
```

### Step 7 — Verify Model Files
Ensure that all required model assets exist in the project root:
* `hand_landmarker.task` (MediaPipe Tasks API asset)
* `signbridge_model.tflite` & `label_encoder.npy`
* `isl_model.tflite` & `isl_label_encoder.npy`
* `words_model.tflite` & `words_label_encoder.npy`
* `motion_model.h5` & `motion_label_encoder.npy`
* `words_motion_model.h5` & `words_motion_label_encoder.npy`

---

## 14. Running the Project

SignBridge AI uses a **two-terminal workflow**:

### Terminal 1: FastAPI Backend
In the project root directory:
```bash
# Activate environment if created
signbridge_env\Scripts\activate

# Start backend server
python backend.py
```
You should see:
```
==============================================================
           SignBridge AI — Loading Models...
==============================================================
Model Name               | Format     | Status     | Classes 
--------------------------------------------------------------
ASL Letters (A-Y)        | TFLite     | READY      | 24      
ISL Letters (A-Z)        | TFLite     | READY      | 26      
Words Static (23)        | TFLite     | READY      | 23      
Motion Letters (J/Z)     | Keras .h5  | READY      | 2       
Motion Words (7)         | Keras .h5  | READY      | 7       
==============================================================
[OK] MediaPipe HandLandmarker initialized (Tasks API - LIVE_STREAM).
>>> SignBridge AI Backend is running and ready for connections! <<<
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Terminal 2: React Frontend
In a second terminal:
```bash
cd "d:\Mega Project\signbridge-ui"
npm run dev
```
You should see:
```
  VITE v8.3.0  ready in 150 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### Step 3: Open Browser
Open your browser and navigate to:
```
http://localhost:5173
```
Allow camera permissions when prompted. The header status dot will turn green and show **"Connected"**.

---

## 15. First-Time Usage

1. **Allow Webcam**: Grant camera permissions in the browser prompt.
2. **ASL Letters (Default)**: Hold up a right or left hand signing letters A–Y. Notice the detection appear in the **Detected Letter** hero panel.
3. **Add to Word**: Press `Space` or click **Add Letter** to append the detected sign to your phrase.
4. **Switch to ISL**: Click **ASL → ISL** (or press `T`) to translate two-handed Indian Sign Language letters.
5. **Switch to Words Mode**: Click **Switch to Words** (or press `P`). Show emergency static signs like **HELP**, **STOP**, or **WATER**.
6. **Record Motion Sign**: In Words mode, click **Record Motion Sign** (or press `N`). Watch the 3-2-1 countdown, perform a wave for **HELLO** or nod for **YES**. The recognized word will lock onto the screen and be spoken aloud.
7. **Voice Feedback**: Ensure **Speech ON** is active. The system will speak words and recognized signs automatically.
8. **Emergency SOS**: In danger, perform the letter **'S' three distinct times**, wave rapidly from side to side, or click the red **SOS** button.

---

## 16. Keyboard Shortcuts

| Key | Action | Scope |
|:---:|---|---|
| `Space` | Append current detected letter to active word | Letters mode |
| `Backspace` / `Delete` | Remove last character from active word | Universal |
| `Escape` | Clear active word | Universal |
| `S` / `s` | Vocalize active phrase via TTS and log to history | Universal |
| `V` / `v` | Toggle automatic Text-to-Speech ON / OFF | Universal |
| `T` / `t` | Toggle Language: **ASL ↔ ISL** | Letters mode only |
| `P` / `p` | Toggle Mode: **Letters ↔ Words** | Universal |
| `N` / `n` | Start / Cancel **Motion Sign Recording** | Words / Letters mode |

---

## 17. Troubleshooting

### 1. `[Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000)`
* **Cause**: Another instance of Python or Uvicorn is already using port 8000.
* **Fix**: In PowerShell run:
  ```powershell
  Get-NetTCPConnection -LocalPort 8000 | Select-Object -ExpandProperty OwningProcess | Stop-Process -Force
  ```

### 2. Camera Shows "Camera Unavailable"
* **Cause**: Camera permission is blocked by browser or another application (e.g. Teams, Zoom, or `predict.py`) is using the webcam.
* **Fix**: Close any desktop Python camera windows or meeting software. Click the camera icon in your browser URL bar and set permission to "Always Allow".

### 3. "Connecting to SignBridge AI backend on localhost:8000…" Banner Persists
* **Cause**: Backend server is not running or crashed on startup.
* **Fix**: Check Terminal 1. Ensure `python backend.py` is running and `http://localhost:8000/health` returns status `ok`.

### 4. MediaPipe Model Asset Error
* **Error**: `[WARNING] MediaPipe model asset 'hand_landmarker.task' not found!`
* **Fix**: Download `hand_landmarker.task` directly using `python test_hand.py` or place it directly into `d:\Mega Project\hand_landmarker.task`.

### 5. Speech Audio Fails to Speak
* **Cause**: Browser audio autoplay restrictions or Chrome Web Speech cancellation bug.
* **Fix**: Click anywhere on the webpage to satisfy browser user-interaction policies. Ensure the **Speech ON** button is highlighted in green.

---

## 18. Known Limitations

1. **Motion Model Quantization**: The J/Z motion model (`motion_model.h5`) and motion words model (`words_motion_model.h5`) run via Keras `.h5` fallback rather than TFLite because `.tflite` exports do not exist for the LSTM recurrent networks.
2. **Lighting & Silhouette Sensitivity**: MediaPipe hand detection relies on RGB contrast. Low-light or back-lit environments degrade keypoint tracking accuracy.
3. **Facial/Body Occlusion on Signs**: Signs requiring hands near the face or chest (e.g., "SORRY" circular rub on chest) may experience hand landmark jitter if the palm overlaps with clothing or chin contours.
4. **Browser-Dependent TTS Voices**: The quality, accent, and availability of speech synthesis voices depend on the host operating system and browser Web Speech API implementation.

---

## 19. Development Guide

* **Modifying Frontend UI**: Edit components in `signbridge-ui/src/components/`. Design tokens, colors, and layout variables are located in `signbridge-ui/src/index.css` and `App.css`.
* **Modifying WebSocket & API Protocol**: Edit `backend.py` inside `websocket_endpoint()`.
* **Adding New Signs or Retraining**:
  1. Collect data: run `words_motion_collection.py` or `data_collection.py`.
  2. Train: run `train_words_motion_model.py` or `train_model.py`.
  3. Update classes and encoders in `backend.py`.
* **Updating Landmark Transformations**: Edit `utils.py` for normalization, coordinate scaling, or augmentation adjustments.

---

## 20. Testing & Verification

The project includes both automated and end-to-end integration tests:

1. **Model Loading & Verification**:
   ```bash
   python scratch/audit_models.py
   ```
   Validates existence, formats, and class mappings of all 5 neural models.
2. **WebSocket Pipeline Integration Test**:
   ```bash
   python scratch/test_motion_ws.py
   ```
   Simulates client frame sequence transmission, verifies 20-frame and 30-frame interpolation, and verifies failure-handling state recovery.
3. **Frontend Production Build**:
   ```bash
   cd signbridge-ui && npm run build
   ```
   Validates complete JSX compilation, asset bundling, and CSS integrity via Vite.

---

## 21. License / Academic Project

* **Project Nature**: Final Year Mega Project / Assistive Human-Computer Interaction (HCI) Engineering Project.
* **License**: Not currently specified. All rights reserved by project authors.

---

## 22. Credits & Acknowledgements

* **Google MediaPipe**: HandLandmarker Tasks API and landmark tracking pipelines.
* **TensorFlow / Keras**: Deep learning and TensorFlow Lite runtime engines.
* **FastAPI & Uvicorn**: High-performance asynchronous WebSocket backend infrastructure.
* **React & Vite**: Fast frontend component framework and tooling.
