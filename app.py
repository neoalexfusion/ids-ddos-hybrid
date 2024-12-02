import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load pre-trained models and features
rf_model = joblib.load("hybrid_model_rf.pkl")
isolation_forest = joblib.load("hybrid_model_isolation_forest.pkl")
features = joblib.load("hybrid_model_features.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        input_data = pd.DataFrame([data])

        # Use Isolation Forest for anomaly detection
        anomaly_scores = isolation_forest.predict(input_data[features])
        if -1 in anomaly_scores:
            return jsonify({"prediction": "Anomaly Detected (Potential DDoS)"})

        # Use Random Forest for prediction
        input_data = input_data[features]
        prediction = rf_model.predict(input_data)
        result = "Attack" if prediction[0] == 0 else "Benign"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
#Yay