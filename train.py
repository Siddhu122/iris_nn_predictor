import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# One-hot encoding
y = to_categorical(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build Neural Network
model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.1,
    verbose=1
)

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc*100:.2f}%")

# ---------------- SAVE MODEL IN OUTER FOLDER ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

# Create model folder if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Save trained model
MODEL_PATH = os.path.join(MODEL_DIR, "iris_nn_model.h5")
model.save(MODEL_PATH)

print("\nTrained model saved at:", MODEL_PATH)

model.save("models/iris_nn_model.h5")

import os
import joblib

# Get absolute path of current python file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create outer trained_data folder path
TRAINED_DATA_DIR = os.path.join(BASE_DIR, "..", "trained_data")

# Create folder if it doesn't exist
os.makedirs(TRAINED_DATA_DIR, exist_ok=True)

# Save trained model (.h5 format)
MODEL_PATH = os.path.join(TRAINED_DATA_DIR, "iris_nn_model.h5")
model.save(MODEL_PATH)

# Save scaler
SCALER_PATH = os.path.join(TRAINED_DATA_DIR, "scaler.save")
joblib.dump(scaler, SCALER_PATH)

print("Model and scaler saved successfully at:")