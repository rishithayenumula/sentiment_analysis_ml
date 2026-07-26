import re
import os
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "model.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please provide some text to analyze."}), 400

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        classes = list(model.classes_)
        confidence = float(max(proba))
        # Full breakdown, useful for the confidence bar in the UI
        breakdown = {cls: float(p) for cls, p in zip(classes, proba)}
    else:
        breakdown = {}

    return jsonify(
        {
            "sentiment": prediction,
            "confidence": confidence,
            "breakdown": breakdown,
        }
    )


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
