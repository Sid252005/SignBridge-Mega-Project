"""
train_words_model.py — SignBridge AI Common Words Model Training
================================================================
Trains a Dense NN to classify 23 common word signs from hand landmarks.
Uses TWO-HAND 126-dim input (63 per hand, same as ISL model).

Pipeline:
  1. Load pre-normalized .npy files from dataset/words/ (126-dim each)
  2. Augment with utils.augment_isl_dataset (factor=3)
  3. Train Dense model (512→256→128→64→23 softmax)
  4. Export .h5 + TFLite
  5. Generate training plots + confusion matrix

Signs: HELP, PAIN, DOCTOR, POLICE, STOP, WATER, FOOD, SICK, HURT,
       MEDICINE, HUNGRY, TIRED, DIZZY, FEVER, PLEASE, WAIT, MORE,
       HOME, PHONE, MOTHER, FATHER, BATHROOM, DANGER

Usage:
  python train_words_model.py
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (saves PNG without display issues)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from utils import augment_isl_dataset    # ISL-style augmentation for 126-dim

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join("dataset", "words")
SIGNS       = [
    "HELP", "PAIN", "DOCTOR", "POLICE", "STOP",
    "WATER", "FOOD", "SICK", "HURT", "MEDICINE",
    "HUNGRY", "TIRED", "DIZZY", "FEVER", "PLEASE",
    "WAIT", "MORE", "HOME", "PHONE", "MOTHER",
    "FATHER", "BATHROOM", "DANGER"
]
NUM_CLASSES   = len(SIGNS)       # 23
INPUT_DIM     = 126              # 63 per hand × 2 hands
AUG_FACTOR    = 3                # 150 raw × augment → ~600 samples/class
TEST_SPLIT    = 0.20             # 80/20 stratified split
BATCH_SIZE    = 32
MAX_EPOCHS    = 50
PATIENCE      = 15               # early stopping patience
SEED          = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 60)
print(f"  SignBridge AI — Common Words Model Training (2-Hand)")
print(f"  Classes  : {NUM_CLASSES} words")
print(f"  Input dim: {INPUT_DIM} (63 per hand × 2)")
print("=" * 60)

# ─── 1. Load pre-normalized data ─────────────────────────────────────────────
# Data was already normalized during collection (normalize_isl_landmarks called
# on every frame in words_data_collection.py). Do NOT re-normalize here.
X_raw, y_raw = [], []
for sign in SIGNS:
    path = os.path.join(DATA_DIR, f"{sign}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing dataset file: {path}\n"
            f"Run words_data_collection.py first."
        )
    data = np.load(path)
    X_raw.extend(data)
    y_raw.extend([sign] * len(data))
    print(f"  [OK] Loaded {sign:12s}: {len(data)} samples (shape: {data.shape})")

X_raw = np.array(X_raw, dtype=np.float32)
y_raw = np.array(y_raw)
print(f"\n  Total raw samples : {len(X_raw)}")
print(f"  Input shape       : {X_raw.shape[1]}  (126 = 63 × 2 hands)")

# Verify feature dimension
assert X_raw.shape[1] == INPUT_DIM, \
    f"Expected {INPUT_DIM}-dim features but got {X_raw.shape[1]}. " \
    f"Re-collect data with updated words_data_collection.py."

# ─── 2. Encode labels ────────────────────────────────────────────────────────
le = LabelEncoder()
y_int = le.fit_transform(y_raw)          # integer labels
print(f"  Classes: {le.classes_}")

# ─── 3. Augment BEFORE splitting ─────────────────────────────────────────────
# Use ISL-style augmentation since we have 126-dim (two-hand) features
print(f"\n  Augmenting dataset ×{AUG_FACTOR} (ISL-style, per-hand)...")
X_aug, y_aug = augment_isl_dataset(X_raw, y_int, factor=AUG_FACTOR)
print(f"  Augmented dataset size: {len(X_aug)} samples")

# ─── 4. 80/20 stratified train/test split ────────────────────────────────────
X_train, X_test, y_train_int, y_test_int = train_test_split(
    X_aug, y_aug, test_size=TEST_SPLIT, random_state=SEED, stratify=y_aug
)
y_train = to_categorical(y_train_int, num_classes=NUM_CLASSES)
y_test  = to_categorical(y_test_int,  num_classes=NUM_CLASSES)
print(f"\n  Training samples : {len(X_train)}")
print(f"  Testing  samples : {len(X_test)}")

# ─── 5. Build model ──────────────────────────────────────────────────────────
# 126-dim input for two-hand features
model = Sequential([
    Input(shape=(INPUT_DIM,)),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64,  activation='relu'),
    Dense(NUM_CLASSES, activation='softmax'),
], name="SignBridge_Words")

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─── 6. Callbacks ─────────────────────────────────────────────────────────────
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
        "words_model.h5", monitor='val_accuracy',
        save_best_only=True, verbose=1
    ),
]

# ─── 7. Train ─────────────────────────────────────────────────────────────────
print("\n>> Training started...\n")
history = model.fit(
    X_train, y_train,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# ─── 8. Evaluate ──────────────────────────────────────────────────────────────
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n{'='*60}")
print(f"  [+] Test Accuracy : {accuracy * 100:.2f}%")
print(f"  [+] Test Loss     : {loss:.4f}")
print(f"{'='*60}\n")

# Per-class classification report
y_pred_probs = model.predict(X_test, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)
print(classification_report(y_test_int, y_pred, target_names=le.classes_))

# ─── 9. Save label encoder ───────────────────────────────────────────────────
np.save("words_label_encoder.npy", le.classes_)
print("[OK] Labels saved as words_label_encoder.npy")

# ─── 10. Export TFLite ────────────────────────────────────────────────────────
print("\n>> Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("words_model.tflite", "wb") as f:
    f.write(tflite_model)
print(f"[OK] words_model.tflite saved — {len(tflite_model)//1024} KB")

# ─── 11. Plot training results ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("SignBridge AI — Common Words Training Results (2-Hand)",
             fontsize=16, fontweight='bold')

# Accuracy curve
ax = axes[0]
ax.plot(history.history['accuracy'],     label='Train',      linewidth=2)
ax.plot(history.history['val_accuracy'], label='Validation', linewidth=2, linestyle='--')
ax.set_title('Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend()
ax.grid(alpha=0.3)

# Loss curve
ax = axes[1]
ax.plot(history.history['loss'],     label='Train',      linewidth=2, color='tomato')
ax.plot(history.history['val_loss'], label='Validation', linewidth=2, linestyle='--', color='darkorange')
ax.set_title('Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(alpha=0.3)

# Confusion matrix heatmap (23×23)
ax = axes[2]
cm = confusion_matrix(y_test_int, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax,
            annot_kws={"size": 6})
ax.set_title(f'Confusion Matrix ({NUM_CLASSES} Words)')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.tick_params(axis='both', labelsize=7)

plt.tight_layout()
plt.savefig("words_training_results.png", dpi=150, bbox_inches='tight')
print("[OK] Graph saved as words_training_results.png")

print(f"\n{'='*60}")
print(f"  Training complete!")
print(f"  Files created:")
print(f"    words_model.h5")
print(f"    words_model.tflite")
print(f"    words_label_encoder.npy")
print(f"    words_training_results.png")
print(f"\n  Next step: python predict.py  (press P for words mode)")
print(f"{'='*60}")
