"""
train_words_motion_model.py -- SignBridge AI Motion Words Model Training
=========================================================================
Trains an LSTM-based classifier for 7 motion word signs:
  HELLO, GOODBYE, THANKYOU, SORRY, YES, NO, AMBULANCE

This is a SEPARATE model from the static words model (23 signs) and
the J/Z motion model (2 signs). It does NOT modify any existing models.

Input:  dataset/word_motions/*.npy
        Each shape: (80, 30, 63) — 80 sequences × 30 frames × 63 landmarks

Output: words_motion_model.h5              — Keras LSTM model
        words_motion_model.tflite          — TFLite for fast inference
        words_motion_label_encoder.npy     — Class labels (7 words)
        words_motion_training_results.png  — Training plots + confusion matrix
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
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from utils import augment_sequences

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR        = "dataset/word_motions"
SEQUENCE_LENGTH = 30           # frames per motion sample
NUM_LANDMARKS   = 63           # 21 landmarks × 3 coordinates
NUM_CLASSES     = 7            # exactly 7 motion words
AUG_FACTOR      = 2            # augmentation multiplier
TEST_SPLIT      = 0.20
BATCH_SIZE      = 16
MAX_EPOCHS      = 60
SEED            = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 60)
print("  SignBridge AI — Motion Words Model Training")
print("  7 signs: HELLO, GOODBYE, THANKYOU, SORRY, YES, NO, AMBULANCE")
print("=" * 60)

# ─── 1. Load motion word data ───────────────────────────────────────────────
X_all, y_all = [], []

for filename in sorted(os.listdir(DATA_DIR)):
    if not filename.endswith(".npy"):
        continue
    word = filename.replace(".npy", "")  # e.g. "HELLO"
    filepath = os.path.join(DATA_DIR, filename)
    data = np.load(filepath)
    print(f"  [OK] Loaded {word}: {data.shape}  ({data.shape[0]} sequences)")

    X_all.append(data)
    y_all.extend([word] * len(data))

if len(X_all) == 0:
    raise FileNotFoundError(
        f"No .npy files found in {DATA_DIR}!\n"
        f"Run words_motion_collection.py first to collect data."
    )

X_all = np.concatenate(X_all, axis=0).astype(np.float32)
y_all = np.array(y_all)

print(f"\n  Total raw sequences: {len(X_all)}")
print(f"  Sequence shape: {X_all.shape[1:]}  (frames × landmarks)")
print(f"  Unique classes: {np.unique(y_all)}")

# ─── 2. Encode labels ───────────────────────────────────────────────────────
le = LabelEncoder()
y_int = le.fit_transform(y_all)
print(f"  Classes: {le.classes_}")
print(f"  Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ─── 3. Augment sequences ───────────────────────────────────────────────────
print(f"\n  Augmenting motion dataset (factor={AUG_FACTOR})...")

# augment_sequences expects integer labels, but works with any label array.
# We pass integer labels and reconstruct one-hot after split.
X_aug, y_aug = augment_sequences(X_all, y_int, factor=AUG_FACTOR)
print(f"  Augmented dataset size: {len(X_aug)} sequences")

# ─── 4. Train / test split ──────────────────────────────────────────────────
X_train, X_test, y_train_int, y_test_int = train_test_split(
    X_aug, y_aug, test_size=TEST_SPLIT, random_state=SEED, stratify=y_aug
)

y_train = to_categorical(y_train_int, num_classes=NUM_CLASSES)
y_test  = to_categorical(y_test_int,  num_classes=NUM_CLASSES)

print(f"\n  Training sequences: {len(X_train)}")
print(f"  Testing  sequences: {len(X_test)}")

# ─── 5. Build LSTM model ────────────────────────────────────────────────────
# Larger than J/Z model (7 classes vs 2) — uses 128-unit first LSTM layer
model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, NUM_LANDMARKS)),

    LSTM(128, return_sequences=True),
    Dropout(0.3),

    LSTM(64, return_sequences=False),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(NUM_CLASSES, activation='softmax'),
], name="SignBridge_Words_Motion_LSTM")

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─── 6. Callbacks ───────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
]

# ─── 7. Train ───────────────────────────────────────────────────────────────
print("\n>> Training motion words model...\n")
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

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
print(classification_report(y_test_int, y_pred, target_names=le.classes_))

# ─── 9. Save model + labels ─────────────────────────────────────────────────
model.save("words_motion_model.h5")
print("[OK] Model saved as words_motion_model.h5")

np.save("words_motion_label_encoder.npy", le.classes_)
print("[OK] Labels saved as words_motion_label_encoder.npy")

# ─── 10. Export TFLite ───────────────────────────────────────────────────────
print("\n>> Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("words_motion_model.tflite", "wb") as f:
    f.write(tflite_model)
print(f"Saved words_motion_model.tflite — {len(tflite_model)//1024} KB")

# ─── 11. Plot results ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("SignBridge AI — Motion Words Training Results (7 Signs)",
             fontsize=14, fontweight='bold')

# Accuracy curve
ax = axes[0]
ax.plot(history.history['accuracy'],     label='Train', linewidth=2)
ax.plot(history.history['val_accuracy'], label='Validation', linewidth=2,
        linestyle='--')
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

# Confusion matrix heatmap
ax = axes[2]
cm = confusion_matrix(y_test_int, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax,
            annot_kws={"size": 11})
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig("words_motion_training_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Graph saved as words_motion_training_results.png")

print("\n" + "=" * 60)
print("  Motion words model training complete!")
print("  The model will be automatically loaded by predict.py.")
print("  Press N in words mode to detect motion words.")
print("=" * 60)
