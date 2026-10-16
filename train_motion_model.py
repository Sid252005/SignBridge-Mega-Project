"""
train_motion_model.py -- SignBridge AI Motion Model Training
=============================================================
Trains an LSTM-based classifier for J and Z motion gestures.

This is a SEPARATE model from the static sign classifier.
It does NOT modify or retrain the existing static model.

Input:  dataset/J_motion.npy, dataset/Z_motion.npy
        Each shape: (num_samples, 30, 63)  — 30-frame sequences of landmarks

Output: motion_model.h5              — Keras LSTM model
        motion_model.tflite          — TFLite for fast inference
        motion_label_encoder.npy     — Class labels ['J', 'Z']
        motion_training_results.png  — Training plots
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

from utils import normalize_landmarks

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR         = "dataset"
MOTION_SIGNS     = ['J', 'Z']
SEQUENCE_LENGTH  = 30           # frames per motion sample
NUM_LANDMARKS    = 63           # 21 landmarks × 3 coordinates
AUG_FACTOR       = 10           # augmentation multiplier for motion data
TEST_SPLIT       = 0.15
BATCH_SIZE       = 32
MAX_EPOCHS       = 300
PATIENCE         = 25
SEED             = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 60)
print("  SignBridge AI — Motion Model Training (J & Z)")
print("=" * 60)

# ─── 1. Load raw motion data ─────────────────────────────────────────────────
X_raw, y_raw = [], []
for sign in MOTION_SIGNS:
    path = os.path.join(DATA_DIR, f"{sign}_motion.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing motion dataset: {path}\n"
            f"Run motion_data_collection.py first to collect data for {sign}."
        )
    data = np.load(path)
    print(f"  [OK] Loaded {sign}: {data.shape}  ({data.shape[0]} sequences)")

    X_raw.append(data)
    y_raw.extend([sign] * len(data))

X_raw = np.concatenate(X_raw, axis=0).astype(np.float32)
y_raw = np.array(y_raw)
print(f"\n  Total raw sequences: {len(X_raw)}")
print(f"  Sequence shape: {X_raw.shape[1:]}  (frames × landmarks)")

# ─── 2. Normalize landmarks (per-frame) ──────────────────────────────────────
print("  Normalizing landmarks (per-frame, wrist-relative + palm-scale)...")
X_norm = np.zeros_like(X_raw)
for i in range(len(X_raw)):
    for f in range(SEQUENCE_LENGTH):
        X_norm[i, f] = normalize_landmarks(X_raw[i, f])

# ─── 3. Encode labels ────────────────────────────────────────────────────────
le = LabelEncoder()
y_int = le.fit_transform(y_raw)
print(f"  Classes: {le.classes_}")

# ─── 4. Motion-specific augmentation ─────────────────────────────────────────
print(f"\n  Augmenting motion dataset ×{AUG_FACTOR}...")

def _speed_jitter(seq, low=0.7, high=1.3):
    """
    Temporally stretch/compress a motion sequence to simulate
    faster or slower gesture performance.
    Resamples the sequence to the original length using interpolation.
    """
    n_frames, n_feat = seq.shape
    speed = np.random.uniform(low, high)
    new_len = max(2, int(n_frames * speed))

    # Interpolate each feature dimension
    old_indices = np.linspace(0, n_frames - 1, n_frames)
    new_indices = np.linspace(0, n_frames - 1, new_len)
    stretched = np.zeros((new_len, n_feat), dtype=np.float32)
    for j in range(n_feat):
        stretched[:, j] = np.interp(new_indices, old_indices, seq[:, j])

    # Resample back to original length
    if new_len != n_frames:
        resampled = np.zeros((n_frames, n_feat), dtype=np.float32)
        resample_indices = np.linspace(0, new_len - 1, n_frames)
        for j in range(n_feat):
            resampled[:, j] = np.interp(resample_indices,
                                         np.arange(new_len), stretched[:, j])
        return resampled
    return stretched


def _add_noise_seq(seq, sigma=0.012):
    """Add Gaussian noise to all frames."""
    return seq + np.random.randn(*seq.shape).astype(np.float32) * sigma


def _scale_jitter_seq(seq, low=0.85, high=1.15):
    """Scale all landmarks uniformly (simulate distance variation)."""
    return seq * np.random.uniform(low, high)


def _rotate_2d_seq(seq, max_angle_deg=12.0):
    """Rotate x-y plane across all frames (simulate wrist tilt)."""
    angle = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    seq = seq.copy().reshape(-1, 21, 3)
    seq[:, :, :2] = np.einsum('ijk,lk->ijl', seq[:, :, :2], R)
    return seq.reshape(-1, 63)


def _mirror_seq(seq):
    """Flip X-axis across all frames (simulate opposite hand)."""
    seq = seq.copy().reshape(-1, 21, 3)
    seq[:, :, 0] = -seq[:, :, 0]
    return seq.reshape(-1, 63)


def _time_shift(seq, max_shift=3):
    """Shift the sequence forward/backward by a few frames (circular)."""
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(seq, shift, axis=0)


def augment_motion_dataset(X, y, factor=10):
    """
    Augment motion sequences with motion-specific transformations.
    Each augmented sample randomly gets 1-3 of these operations:
      1. Speed jitter   — temporal stretch/compress
      2. Gaussian noise — sensor jitter
      3. Scale jitter   — distance variation
      4. 2-D rotation   — wrist tilt
      5. Mirror hand    — opposite hand
      6. Time shift     — slight temporal offset
    """
    X_list, y_list = [X.copy()], [y.copy()]

    # Always add mirrored versions
    X_mirror = np.array([_mirror_seq(s) for s in X], dtype=np.float32)
    X_list.append(X_mirror)
    y_list.append(y.copy())

    aug_ops = ['speed', 'noise', 'scale', 'rotate', 'mirror', 'time_shift']

    for _ in range(factor - 1):
        aug_batch = []
        for seq in X:
            seq = seq.copy()
            ops = np.random.choice(aug_ops,
                                   size=np.random.randint(1, 4),
                                   replace=False)
            for op in ops:
                if op == 'speed':
                    seq = _speed_jitter(seq)
                elif op == 'noise':
                    seq = _add_noise_seq(seq)
                elif op == 'scale':
                    seq = _scale_jitter_seq(seq)
                elif op == 'rotate':
                    seq = _rotate_2d_seq(seq)
                elif op == 'mirror':
                    seq = _mirror_seq(seq)
                elif op == 'time_shift':
                    seq = _time_shift(seq)
            aug_batch.append(seq)
        X_list.append(np.array(aug_batch, dtype=np.float32))
        y_list.append(y.copy())

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


X_aug, y_aug = augment_motion_dataset(X_norm, y_int, factor=AUG_FACTOR)
print(f"  Augmented dataset size: {len(X_aug)} sequences")

# ─── 5. Train / test split ───────────────────────────────────────────────────
X_train, X_test, y_train_int, y_test_int = train_test_split(
    X_aug, y_aug, test_size=TEST_SPLIT, random_state=SEED, stratify=y_aug
)
y_train = to_categorical(y_train_int, num_classes=len(MOTION_SIGNS))
y_test  = to_categorical(y_test_int,  num_classes=len(MOTION_SIGNS))
print(f"\n  Training sequences: {len(X_train)}")
print(f"  Testing  sequences: {len(X_test)}")

# ─── 6. Build LSTM model ─────────────────────────────────────────────────────
model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, NUM_LANDMARKS)),

    # Bidirectional LSTM captures patterns from both directions of the motion
    Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4))),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(64, kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.2),

    Dense(len(MOTION_SIGNS), activation='softmax'),
], name="SignBridge_Motion_LSTM")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─── 7. Callbacks ─────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor='val_accuracy', patience=PATIENCE,
        restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=8,
        min_lr=1e-6, verbose=1
    ),
    ModelCheckpoint(
        "motion_model.h5", monitor='val_accuracy',
        save_best_only=True, verbose=1
    ),
]

# ─── 8. Train ─────────────────────────────────────────────────────────────────
print("\n>> Training motion model...\n")
history = model.fit(
    X_train, y_train,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# ─── 9. Evaluate ──────────────────────────────────────────────────────────────
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n{'='*60}")
print(f"  [+] Test Accuracy : {accuracy * 100:.2f}%")
print(f"  [+] Test Loss     : {loss:.4f}")
print(f"{'='*60}\n")

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
print(classification_report(y_test_int, y_pred, target_names=le.classes_))

# ─── 10. Save labels ─────────────────────────────────────────────────────────
np.save("motion_label_encoder.npy", le.classes_)
print("[OK] Motion labels saved as motion_label_encoder.npy")

# ─── 11. Plot results ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("SignBridge AI — Motion Model Training Results (J & Z)",
             fontsize=14, fontweight='bold')

# Accuracy curve
ax = axes[0]
ax.plot(history.history['accuracy'],     label='Train', linewidth=2)
ax.plot(history.history['val_accuracy'], label='Validation', linewidth=2, linestyle='--')
ax.set_title('Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend()
ax.grid(alpha=0.3)

# Loss curve
ax = axes[1]
ax.plot(history.history['loss'],     label='Train', linewidth=2, color='tomato')
ax.plot(history.history['val_loss'], label='Validation', linewidth=2,
        linestyle='--', color='darkorange')
ax.set_title('Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(alpha=0.3)

# Confusion matrix
ax = axes[2]
cm = confusion_matrix(y_test_int, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax,
            annot_kws={"size": 14})
ax.set_title('Confusion Matrix (J vs Z)')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig("motion_training_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Graph saved as motion_training_results.png")

# ── Export TFLite ─────────────────────────────────────────────────────────────
print("\n>> Converting motion model to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("motion_model.tflite", "wb") as f:
    f.write(tflite_model)
print(f"[OK] Motion TFLite model saved — {len(tflite_model)//1024} KB")
print("\nMotion model training complete!")
print("The motion model will be automatically loaded by predict.py.")
