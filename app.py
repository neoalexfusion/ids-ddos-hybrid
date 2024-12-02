from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the models and features
try:
    rf_model = joblib.load("hybrid_model_rf.pkl")
    isolation_forest = joblib.load("hybrid_model_isolation_forest.pkl")
    features = joblib.load("hybrid_model_features.pkl")
except Exception as e:
    print(f"Error loading models or features: {e}")

@app.route("/", methods=["GET"])
def home():
    """
    API Root Endpoint
    """
    return jsonify({"message": "Hybrid DDoS Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict DDoS attack or benign traffic based on input features.
    """
    try:
        # Parse input JSON
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Ensure the input has all required features
        input_features = np.array([data.get(feature, 0) for feature in features]).reshape(1, -1)

        # Isolation Forest Prediction (-1: Anomaly, 1: Normal)
        anomaly = isolation_forest.predict(input_features)[0]

        # Random Forest Prediction (0: Attack, 1: Benign)
        rf_probs = rf_model.predict_proba(input_features)[0][1]  # Probability of benign
        threshold = 0.7  # Set your decision threshold here
        rf_prediction = int(rf_probs >= threshold)

        # Hybrid Model Decision Logic
        if anomaly == -1:  # If flagged as anomaly
            final_prediction = 0 if rf_prediction == 0 else 1  # Attack if RF also predicts attack
        else:
            final_prediction = rf_prediction  # Use RF prediction for normal data

        # Return the prediction result
        result = {
            "anomaly_flag": int(anomaly == -1),
            "rf_prediction": rf_prediction,
            "final_prediction": "Attack" if final_prediction == 0 else "Benign",
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For local testing; Vercel will handle `app` internally
    app.run(host="0.0.0.0", port=5000)
