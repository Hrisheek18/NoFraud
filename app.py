from flask import Flask, request, jsonify
import numpy as np
import pickle

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("category_le.pkl", "rb") as f:
    category_le = pickle.load(f)

app = Flask(__name__)

feature_order = [
    "category", "amt", "gender", "state", "lat", "long",
    "city_pop", "merch_lat", "merch_long", "hour", "age"
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not all(feat in data for feat in feature_order):
        return jsonify({"error": "Missing features"}), 400

    input_data = []
    for feat in feature_order:
        input_data.append(data[feat])

    input_data[0] = category_le.transform([input_data[0]])[0]

    num_indices = [1, 4, 5, 6, 7, 8, 9, 10]
    input_arr = np.array(input_data, dtype=float).reshape(1, -1)
    input_arr[0, num_indices] = scaler.transform(input_arr[:, num_indices])

    pred = model.predict(input_arr)[0]
    proba = model.predict_proba(input_arr)[0, 1]

    return jsonify({
        "is_fraud": int(pred),
        "fraud_probability": float(proba)
    })

if __name__ == "__main__":
    app.run(debug=True)
