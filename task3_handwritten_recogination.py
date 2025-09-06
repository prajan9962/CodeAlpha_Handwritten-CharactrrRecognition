# task3_handwritten_recognition.py
# CodeAlpha - Task 3: Handwritten Character Recognition (using MNIST)

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------
# 1) Load dataset
# -------------------------
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape
x_train = (x_train / 255.0).astype("float32")[..., None]  # shape: (60000, 28, 28, 1)
x_test  = (x_test / 255.0).astype("float32")[..., None]   # shape: (10000, 28, 28, 1)

print("Data shapes:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# -------------------------
# 2) Build CNN model
# -------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# -------------------------
# 3) Train model
# -------------------------
print("Training the model...")
history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=128,
    verbose=2
)

# -------------------------
# 4) Evaluate model
# -------------------------
print("Evaluating on test data...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# -------------------------
# 5) Generate reports
# -------------------------
print("Generating reports...")

# Predictions
y_pred = model.predict(x_test, verbose=0).argmax(axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,6))
plt.imshow(cm, cmap="Blues", interpolation="nearest")
plt.title("Confusion Matrix (MNIST)")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

# Create folders if not exist
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Save confusion matrix
plt.savefig("reports/mnist_confusion_matrix.png", dpi=150)
plt.close()

print("Confusion matrix saved in reports/mnist_confusion_matrix.png")

# -------------------------
# 6) Save model
# -------------------------
model.save("models/mnist_cnn.h5")
print("Model saved in models/mnist_cnn.h5")

print("\n🎉 Task 3 Completed Successfully!")
