from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import pickle

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Build a row with all expected columns, defaulting to 0
        row = pd.DataFrame([{col: 0 for col in columns}])

        # Fill in submitted values
        for key, value in data.items():
            if key in row.columns:
                row[key] = float(value)
            else:
                # Handle one-hot encoded columns (e.g. sex_M, address_U)
                ohe_key = f"{key}_{value}"
                if ohe_key in row.columns:
                    row[ohe_key] = 1.0

        row_sc = scaler.transform(row)
        pred   = model.predict(row_sc)[0]
        proba  = model.predict_proba(row_sc)[0]

        result = {
            "prediction": encoder.inverse_transform([pred])[0],
            "probabilities": {
                cls: round(float(p), 4)
                for cls, p in zip(encoder.classes_, proba)
            }
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
