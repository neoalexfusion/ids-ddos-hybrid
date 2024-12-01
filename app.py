from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained models and feature list
rf_model = joblib.load("hybrid_model_rf.pkl")
isolation_forest = joblib.load("hybrid_model_isolation_forest.pkl")
features = joblib.load("hybrid_model_features.pkl")

@app.route("/", methods=["GET"])
def index():
    """
    Root endpoint to check if the API is running.
    """
    return jsonify({"message": "Hybrid DDoS Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to predict DDoS attacks.
    Accepts JSON payload containing feature values.
    """
    try:
        # Parse JSON payload
        data = request.get_json()

        # Convert data to DataFrame
        input_data = pd.DataFrame([data], columns=features)

        # Predict anomaly using Isolation Forest
        anomaly_prediction = isolation_forest.predict(input_data)

        # Predict class using Random Forest
        rf_probabilities = rf_model.predict_proba(input_data)[:, 1]  # Probability for benign class
        rf_prediction = (rf_probabilities >= 0.7).astype(int)  # Threshold is set to 0.7

        # Hybrid model logic
        if anomaly_prediction[0] == -1:  # Isolation Forest flags as anomaly
            if rf_prediction[0] == 0:  # Random Forest predicts attack
                final_prediction = "Attack"
            else:
                final_prediction = "Benign"
        else:
            final_prediction = "Attack" if rf_prediction[0] == 0 else "Benign"

        return jsonify({
            "anomaly_prediction": int(anomaly_prediction[0]),  # -1 for anomaly, 1 for normal
            "rf_prediction": int(rf_prediction[0]),
            "final_prediction": final_prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
