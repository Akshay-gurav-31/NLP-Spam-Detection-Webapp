"""
app.py — Flask Web Server for Spam Detector AI
================================================
Loads the pre-trained Scikit-learn pipeline (model.pkl) and exposes two routes:
  GET  /          → Render the frontend UI (templates/index.html)
  POST /predict   → Accept a JSON message, classify it, return prediction + confidence
"""

from flask import Flask, render_template, request, jsonify
import joblib
import os
import numpy as np

# ── Initialise Flask application ──
app = Flask(__name__)

# ── Path to the serialised Scikit-learn pipeline ──
MODEL_PATH = "model.pkl"


def load_model():
    """
    Load the trained pipeline from disk.
    Returns the model object if found, otherwise None.
    The model is loaded once at startup to avoid repeated I/O on every request.
    """
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
    return None


# ── Load model into memory at startup ──
model = load_model()


@app.route("/")
def home():
    """Serve the main frontend page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts JSON: { "message": "<text>" }
    Returns JSON: { "prediction": "Spam" | "Not Spam", "confidence": <float> }

    Processing steps:
      1. Validate that the incoming message is not empty.
      2. Apply the same text preprocessing used during training.
      3. Run the message through the pipeline (TF-IDF → Naive Bayes).
      4. Return the class label and the probability of the predicted class.
    """

    # ── Model availability check ──
    if model is None:
        return jsonify({"error": "Model not found. Please run ml_training/train.py first."}), 500

    # ── Parse and validate request body ──
    data    = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Message is empty. Please provide some text."}), 400

    # ── Apply the same preprocessing as training (lowercase, no punctuation, no stopwords) ──
    from ml_training.train import preprocess_text
    clean_msg = preprocess_text(message)

    # ── Run prediction through the pipeline ──
    prediction = model.predict([clean_msg])[0]

    # ── Retrieve class probabilities and select the highest one as confidence ──
    proba      = model.predict_proba([clean_msg])[0]
    confidence = round(float(np.max(proba)) * 100, 2)

    return jsonify({
        "prediction": "Spam" if prediction == "spam" else "Not Spam",
        "confidence": confidence
    })


# ── Entry point ──
if __name__ == "__main__":
    if model is None:
        print("[WARNING] model.pkl not found. Run `cd ml_training && python train.py` first.")

    # Debug mode is enabled for development; disable in production.
    app.run(debug=True, port=5000)
