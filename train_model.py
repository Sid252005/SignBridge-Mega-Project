import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ── Load Dataset ─────────────────────────────────────────────
DATA_DIR = "dataset"
SIGNS = list("ABCDE")

X, y = [], []

for sign in SIGNS:
    path = os.path.join(DATA_DIR, f"{sign}.npy")
    data = np.load(path)
    X.extend(data)
    y.extend([sign] * len(data))
    print(f"✅ Loaded {sign}: {len(data)} samples")

X = np.array(X)
y = np.array(y)

print(f"\nTotal samples: {len(X)}")
print(f"Each sample shape: {X[0].shape}")

# ── Encode Labels ─────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# ── Split Data ────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ── Build Model ───────────────────────────────────────────────
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(SIGNS), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── Train ─────────────────────────────────────────────────────
print("\n🚀 Training started...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# ── Evaluate ──────────────────────────────────────────────────
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")

# ── Save Model & Labels ───────────────────────────────────────
model.save("signbridge_model.h5")
np.save("label_encoder.npy", le.classes_)
print("✅ Model saved as signbridge_model.h5")
print("✅ Labels saved as label_encoder.npy")

# ── Plot Accuracy Graph ───────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_results.png")
plt.show()
print("✅ Graph saved as training_results.png")