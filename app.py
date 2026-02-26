import torch
import torch.nn as nn
import joblib
import os
from flask import Flask, render_template, request

# -----------------------------
# Initialize Flask App
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Define Same NN Architecture
# -----------------------------
class IrisNN(nn.Module):
    def __init__(self):
        super(IrisNN, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------------
# Load Model & Scaler
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "trained_data", "iris_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "trained_data", "scaler.pkl")

model = IrisNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

scaler = joblib.load(SCALER_PATH)

classes = ["Setosa", "Versicolor", "Virginica"]

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        features = [[sepal_length, sepal_width, petal_length, petal_width]]

        # Apply same scaler used in training
        scaled_features = scaler.transform(features)

        input_tensor = torch.tensor(scaled_features, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)

        result = classes[predicted.item()]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Class: {result}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
    if __name__ == "__main__":
        import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)