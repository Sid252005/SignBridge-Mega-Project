"""
train_model.py -- SignBridge AI Training Script (v3)
====================================================
Upgrades over v2:
  [+] Expanded from A-E (5 classes) to A-Y (24 static ASL signs)
  [+] Skips J & Z (motion-based) - handled in Phase 2
  [+] Wider entry layer (512) for better separation of similar signs
  [+] AUG_FACTOR raised to 15 => ~2250 samples per sign
  [+] PATIENCE raised to 30 for larger class space
  [+] Per-class accuracy report + full 24x24 Confusion Matrix
  [+] Auto TFLite export after training
  [+] All training plots saved to training_results.png
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
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

from utils import normalize_landmarks, augment_dataset

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR    = "dataset"
# 24 static ASL signs — J and Z are motion-based (Phase 2)
SIGNS       = list("ABCDEFGHIKLMNOPQRSTUVWXY")
AUG_FACTOR  = 15       # 150 raw x 15 = ~2250 samples per sign after augmentation
TEST_SPLIT  = 0.15
BATCH_SIZE  = 64
MAX_EPOCHS  = 400
PATIENCE    = 30       # higher patience for 24-class problem
SEED        = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 60)
print("  SignBridge AI — Model Training v3  (A-Y, 24 Signs)")
print("=" * 60)

# ─── 1. Load raw data ─────────────────────────────────────────────────────────
X_raw, y_raw = [], []
for sign in SIGNS:
    path = os.path.join(DATA_DIR, f"{sign}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset file: {path}")
    data = np.load(path)
    X_raw.extend(data)
    y_raw.extend([sign] * len(data))
    print(f"  [OK] Loaded {sign}: {len(data)} samples")

X_raw = np.array(X_raw, dtype=np.float32)
y_raw = np.array(y_raw)
print(f"\n  Total raw samples : {len(X_raw)}")

# ─── 2. Normalize landmarks ───────────────────────────────────────────────────
print("  Normalizing landmarks (wrist-relative + palm-scale)...")
X_norm = np.array([normalize_landmarks(row) for row in X_raw], dtype=np.float32)

# ─── 3. Encode labels ─────────────────────────────────────────────────────────
le = LabelEncoder()
y_int = le.fit_transform(y_raw)          # integer labels for augmentation
print(f"  Classes: {le.classes_}")

# ─── 4. Augment BEFORE splitting ─────────────────────────────────────────────
print(f"\n  Augmenting dataset ×{AUG_FACTOR}...")
X_aug, y_aug = augment_dataset(X_norm, y_int, factor=AUG_FACTOR)
print(f"  Augmented dataset size: {len(X_aug)} samples")

# ─── 5. Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train_int, y_test_int = train_test_split(
    X_aug, y_aug, test_size=TEST_SPLIT, random_state=SEED, stratify=y_aug
)
y_train = to_categorical(y_train_int, num_classes=len(SIGNS))
y_test  = to_categorical(y_test_int,  num_classes=len(SIGNS))
print(f"\n  Training samples : {len(X_train)}")
print(f"  Testing  samples : {len(X_test)}")

# ─── 6. Build model ───────────────────────────────────────────────────────────
# Wider entry layer (512) gives more capacity to separate 24 similar-looking signs
model = Sequential([
    Input(shape=(63,)),

    Dense(512, kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.4),

    Dense(256, kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.35),

    Dense(128, kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.3),

    Dense(64, kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.25),

    Dense(32, kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.2),

    Dense(len(SIGNS), activation='softmax'),
], name="SignBridge_v3")

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
        monitor='val_loss', factor=0.5, patience=10,
        min_lr=1e-6, verbose=1
    ),
    ModelCheckpoint(
        "signbridge_model.h5", monitor='val_accuracy',
        save_best_only=True, verbose=1
    ),
]

# ─── 8. Train ─────────────────────────────────────────────────────────────────
print("\n>> Training started...\n")
history = model.fit(
    X_train, y_train,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# ─── 9. Evaluate ──────────────────────────────────────────────────────────────
# Load best weights (already restored by EarlyStopping)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n{'='*60}")
print(f"  [+] Test Accuracy : {accuracy * 100:.2f}%")
print(f"  [+] Test Loss     : {loss:.4f}")
print(f"{'='*60}\n")

# Per-class report
y_pred_probs = model.predict(X_test, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)
print(classification_report(y_test_int, y_pred, target_names=le.classes_))

# ─── 10. Save model & labels ──────────────────────────────────────────────────
np.save("label_encoder.npy", le.classes_)
print("[OK] Labels saved as label_encoder.npy")

# ─── 11. Plot results ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("SignBridge AI — Training Results (v2)", fontsize=16, fontweight='bold')

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
ax.plot(history.history['val_loss'], label='Validation', linewidth=2, linestyle='--', color='darkorange')
ax.set_title('Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(alpha=0.3)

# Confusion matrix — 24x24 needs smaller font
ax = axes[2]
cm = confusion_matrix(y_test_int, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax,
            annot_kws={"size": 7})
ax.set_title('Confusion Matrix (24 Signs)')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.tick_params(axis='both', labelsize=8)

plt.tight_layout()
plt.savefig("training_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Graph saved as training_results.png")

# ── Export TFLite (fast runtime model) ───────────────────────────────────────
print("\n>> Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("signbridge_model.tflite", "wb") as f:
    f.write(tflite_model)
print(f"[OK] TFLite model saved — {len(tflite_model)//1024} KB")
print("\nTraining complete! Run predict.py to test.")