"""
app.py
======
Task 3: API Development
Flask REST API that loads the trained heart-disease model and serves
predictions as JSON. Also serves a small HTML form (templates/index.html)
for manual testing in a browser.
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
scaler = bundle["scaler"]
FEATURE_NAMES = bundle["feature_names"]
TRAIN_ACCURACY = bundle.get("accuracy")

# Human-friendly description of expected fields (used for input validation
# and shown on the "/" form page).
FEATURE_INFO = {
    "age": "Age in years",
    "sex": "1 = male, 0 = female",
    "cp": "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "Resting ECG results (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of the peak exercise ST segment (0-2)",
    "ca": "Number of major vessels colored by fluoroscopy (0-4)",
    "thal": "Thalassemia (0-3)",
}


def build_feature_vector(payload) -> pd.DataFrame:
    """Validates payload and returns an ordered feature DataFrame for the model."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object of feature: value pairs.")

    missing = [f for f in FEATURE_NAMES if f not in payload]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    try:
        values = [float(payload[f]) for f in FEATURE_NAMES]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"All fields must be numeric. Details: {exc}")

    non_finite = [
        FEATURE_NAMES[i] for i, v in enumerate(values) if not math.isfinite(v)
    ]
    if non_finite:
        raise ValueError(
            f"Field(s) must be finite numbers (NaN/Infinity not allowed): "
            f"{', '.join(non_finite)}"
        )

    return pd.DataFrame([values], columns=FEATURE_NAMES)


@app.route("/", methods=["GET"])
def home():
    """Simple HTML form for manual/browser testing."""
    return render_template("index.html", features=FEATURE_INFO)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts patient details as JSON, e.g.:
    {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }

    Returns:
    {
        "prediction": "Heart Disease Detected",
        "prediction_label": 1,
        "probability": 0.82
    }
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    try:
        features = build_feature_vector(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    features_scaled = scaler.transform(features)
    pred = int(model.predict(features_scaled)[0])
    proba = model.predict_proba(features_scaled)[0][pred]

    label = "Heart Disease Detected" if pred == 1 else "No Heart Disease Detected"

    return jsonify({
        "prediction": label,
        "prediction_label": pred,
        "probability": round(float(proba), 4),
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(Exception)
def handle_unexpected_error(e):
    # Make sure API consumers always get JSON back, even on unforeseen errors,
    # instead of Flask's default HTML error page.
    app.logger.exception("Unhandled exception")
    return jsonify({"error": "Internal server error.", "details": str(e)}), 500


if __name__ == "__main__":
    # For local development. On Render, gunicorn (see Procfile / start
    # command) serves the app instead of this development server.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
