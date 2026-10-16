"""
train_isl_model.py -- SignBridge AI ISL Model Training
=======================================================
Trains a Dense neural network for Indian Sign Language (ISL).

Key differences from ASL training (train_model.py):
  - Input: 126 features (2 hands x 63 landmarks) instead of 63
  - Classes: All 26 letters (A-Z, ISL treats J and Z as static)
  - Data is already normalized during collection (no re-normalization)
  - Uses augment_isl_dataset() for 126-dim augmentation

Outputs:
  isl_model.h5               — Keras model
  isl_model.tflite           — TFLite for fast inference
  isl_label_encoder.npy      — Class labels
  isl_training_results.png   — Training plots
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore

from utils import augment_isl_dataset

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join("dataset", "ISL")
SIGNS       = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")   # All 26 ISL letters
AUG_FACTOR  = 15       # augmentation multiplier
TEST_SPLIT  = 0.20     # 80/20 train/test split
BATCH_SIZE  = 32
MAX_EPOCHS  = 50
PATIENCE    = 15
SEED        = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 60)
print("  SignBridge AI — ISL Model Training  (A-Z, 26 Signs)")
print("=" * 60)

# ─── 1. Load data (already normalized during collection) ─────────────────────
X_raw, y_raw = [], []
for sign in SIGNS:
    path = os.path.join(DATA_DIR, f"{sign}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing ISL dataset file: {path}\n"
            f"Run isl_data_collection.py first to collect data for {sign}."
        )
    data = np.load(path)
    X_raw.extend(data)
    y_raw.extend([sign] * len(data))
    print(f"  [OK] Loaded ISL {sign}: {len(data)} samples ({data.shape[-1]}-dim)")

X_raw = np.array(X_raw, dtype=np.float32)
y_raw = np.array(y_raw)
print(f"\n  Total raw samples  : {len(X_raw)}")
print(f"  Feature dimension  : {X_raw.shape[1]}")

# Verify feature dimension
assert X_raw.shape[1] == 126, (
    f"Expected 126-dim ISL features, got {X_raw.shape[1]}. "
    f"Re-collect data with isl_data_collection.py."
)

# ─── 2. Encode labels ────────────────────────────────────────────────────────
le = LabelEncoder()
y_int = np.array(le.fit_transform(y_raw), dtype=np.int32)
classes = list(le.classes_)
num_classes = len(classes)
print(f"  Classes ({num_classes}): {classes}")

# ─── 3. Augment dataset ──────────────────────────────────────────────────────
print(f"\n  Augmenting ISL dataset x{AUG_FACTOR}...")
X_aug, y_aug = augment_isl_dataset(X_raw, y_int, factor=AUG_FACTOR)
print(f"  Augmented dataset size: {len(X_aug)} samples")

# ─── 4. Train / test split ───────────────────────────────────────────────────
X_train, X_test, y_train_int, y_test_int = train_test_split(
    X_aug, y_aug, test_size=TEST_SPLIT, random_state=SEED, stratify=y_aug
)
y_train = to_categorical(y_train_int, num_classes=num_classes)
y_test  = to_categorical(y_test_int,  num_classes=num_classes)
print(f"\n  Training samples : {len(X_train)}")
print(f"  Testing  samples : {len(X_test)}")

# ─── 5. Build model ──────────────────────────────────────────────────────────
# Larger input (126-dim) for two-hand ISL classification
model = Sequential([
    Input(shape=(126,)),

    Dense(512, activation='relu'),
    Dropout(0.4),

    Dense(256, activation='relu'),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(64, activation='relu'),

    Dense(num_classes, activation='softmax'),
], name="SignBridge_ISL")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─── 6. Callbacks ────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor='val_accuracy', patience=PATIENCE,
        restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=7,
        min_lr=1e-6, verbose=1
    ),
    ModelCheckpoint(
        "isl_model.h5", monitor='val_accuracy',
        save_best_only=True, verbose=1
    ),
]

# ─── 7. Train ────────────────────────────────────────────────────────────────
print("\n>> Training ISL model...\n")
history = model.fit(
    X_train, y_train,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# ─── 8. Evaluate ─────────────────────────────────────────────────────────────
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n{'='*60}")
print(f"  [+] Test Accuracy : {accuracy * 100:.2f}%")
print(f"  [+] Test Loss     : {loss:.4f}")
print(f"{'='*60}\n")

# Per-class report
y_pred_probs = model.predict(X_test, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)
print(classification_report(y_test_int, y_pred, target_names=classes))

# ─── 9. Save labels ──────────────────────────────────────────────────────────
np.save("isl_label_encoder.npy", np.array(classes))
print("[OK] ISL labels saved as isl_label_encoder.npy")

# ─── 10. Plot results ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("SignBridge AI — ISL Training Results (A-Z, 26 Signs)",
             fontsize=16, fontweight='bold')

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

# Confusion matrix — 26x26 needs smaller font
ax = axes[2]
cm = confusion_matrix(y_test_int, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes, ax=ax,
            annot_kws={"size": 6})
ax.set_title('Confusion Matrix (26 ISL Signs)')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.tick_params(axis='both', labelsize=7)

plt.tight_layout()
plt.savefig("isl_training_results.png", dpi=150, bbox_inches='tight')
print("[OK] Graph saved as isl_training_results.png")

# ── Export TFLite (fast runtime model) ────────────────────────────────────────
print("\n>> Converting ISL model to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("isl_model.tflite", "wb") as f:
    f.write(tflite_model)
print(f"[OK] ISL TFLite model saved — {len(tflite_model)//1024} KB")
print("\nISL training complete! Run predict.py and press L to toggle ISL mode.")
